//! # UI - terminal user interface to visualize tree exploration for PUCT
//!
//! Usage: `cargo run --release --bin ui -- -c breakthrough -m alpha`
//!
//! Keyboard and mouse can be used to play the game step by step while
//! inspecting the tree search.

#![allow(non_snake_case)]

use ggpf::game::breakthrough::{ui::IBreakthrough, BreakthroughBuilder};
use ggpf::game::meta::{
    simulated::Simulated,
    with_history::{IWithHistory, WithHistoryGB},
};
use ggpf::game::openai::{Gym, GymBuilder};
use ggpf::game::*;
use ggpf::policies::mcts::MCTSTreeNode;
use ggpf::policies::{
    mcts::muz::{Muz, MuzEvaluators, MuzPolicy},
    mcts::puct::*,
    ppa::*,
    MultiplayerPolicy, MultiplayerPolicyBuilder,
};
use ggpf::settings;

use clap::{App, Arg};
use cursive::traits::*;
use cursive::view::SizeConstraint;
use cursive::views::ViewRef;
use cursive::views::{Button, Dialog, LinearLayout, NamedView, Panel, ResizedView};
use cursive::Cursive;
use cursive_flexi_logger_view::FlexiLoggerView;
use cursive_tree_view::{Placement, TreeView};
use flexi_logger::{LogTarget, Logger};
use ggpf::settings::{Config, Method};
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::RwLock;
use std::{error, fmt, fs};

#[derive(Clone)]
/// Entry for the tree view.
struct TreeEntry<G>
where
    G: Clone + Features,
{
    name: String,
    state: Arc<RwLock<MCTSTreeNode<G, PUCTPolicy_<G>>>>,
    probability: f32,
    value: f32,
    N_visits: f32,
    reward: f32,
}

impl<G> fmt::Display for TreeEntry<G>
where
    G: Clone + Features,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} | P{:^2.2} | V{:^2.2} | N{:^4} | R{:^2}",
            self.name, self.probability, self.value, self.N_visits, self.reward
        )
    }
}

impl<G> fmt::Debug for TreeEntry<G>
where
    G: Clone + Features,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} | P:{:^2.2} | V:{:^2.2} | N:{:^4} | R:{:^2}",
            self.name, self.probability, self.value, self.N_visits, self.reward
        )
    }
}

/// Expand the tree view by looking for the children and inserting new nodes.
fn expand_tree<G>(treeview: &mut TreeView<TreeEntry<G>>, parent_row: usize)
where
    G: Clone + Features,
{
    let content: TreeEntry<G> = treeview.borrow_item(parent_row).unwrap().clone();

    let tree_node = content.state.read().unwrap();

    let moves: Vec<&G::Move> = tree_node.moves.iter().map(|(a, _)| a).collect();

    for action in moves {
        let move_info = tree_node.info.moves.get(action).unwrap();
        let state = tree_node.moves.get(action).unwrap().clone();

        let item = TreeEntry {
            name: format!("{:?}", action),
            state,
            probability: move_info.pi,
            value: move_info.Q,
            N_visits: move_info.N_a,
            reward: move_info.reward,
        };

        if move_info.N_a == 0. {
            treeview.insert_item(item, Placement::LastChild, parent_row);
        } else {
            treeview.insert_container_item(item, Placement::LastChild, parent_row);
        }
    }
}

#[derive(Clone)]
/// Communication type from GUI to Simulator.
enum GuiToSimChannel {
    /// Play next move
    Next,
}

/// Communication channel from Simulator to GUI.
struct GuiEventSender {
    /// Channel sender.
    f: crossbeam_channel::Sender<Box<dyn FnOnce(&mut Cursive) + Send>>,
}

impl GuiEventSender {
    /// Send a closure that will remotely mutate the GUI.
    fn send<F, GV, G>(&self, f: F)
    where
        GV: GameView,
        G: Features + Clone + 'static,
        F: FnOnce(&mut GameDuelUI<GV, G>) + Send + 'static,
    {
        self.f
            .send(Box::new(move |cursive| f(&mut GameDuelUI::new(cursive))))
            .expect("gui is gone?");
    }
}

/// Interface for game matches.
struct GameDuelUI<'a, GV, G> {
    /// Cursive instance.
    siv: &'a mut Cursive,
    /// Storing game view and game types.
    _gv: PhantomData<(GV, G)>,
}

impl<'a, GV, G> GameDuelUI<'a, GV, G>
where
    GV: GameView,
    G: Features + Clone + 'static,
{
    /// Create a new UI manager given the cursive instance.
    fn new(siv: &'a mut Cursive) -> Self {
        Self {
            siv,
            _gv: PhantomData,
        }
    }

    /// Update game state.
    fn new_state(&mut self, state: GV::G) {
        let mut view: ViewRef<GV> = self.siv.find_name("game").unwrap();
        view.set_state(state);
    }

    /// Update policy tree state.
    fn new_policy_tree(
        &mut self,
        root_node: Arc<RwLock<MCTSTreeNode<G, PUCTPolicy_<G>>>>,
        root_value: f32,
        count: f32,
    ) {
        let mut treeview: ViewRef<TreeView<TreeEntry<G>>> = self.siv.find_name("tree").unwrap();

        treeview.clear();
        treeview.insert_container_item(
            TreeEntry {
                name: "root".to_string(),
                state: root_node,
                reward: 0.,
                probability: 1.,
                value: root_value,
                N_visits: count,
            },
            Placement::After,
            0,
        );
    }

    /// Render layout by creating views and adding them to cursive.
    ///
    /// Returns a structure to communicate with the GUI.
    fn render(
        &mut self,
        view: GV,
        game_simulator_sender: mpsc::Sender<GuiToSimChannel>,
    ) -> GuiEventSender {
        let left = LinearLayout::vertical().child(ResizedView::new(
            SizeConstraint::AtMost(100),
            SizeConstraint::Free,
            FlexiLoggerView::scrollable(),
        ));

        let middle = LinearLayout::vertical()
            .child(NamedView::new("game", view))
            .child(Button::new_raw("Next", move |_s| {
                game_simulator_sender.send(GuiToSimChannel::Next).unwrap()
            }));

        let mut treeview = TreeView::<TreeEntry<G>>::new();

        treeview.set_on_collapse(move |siv: &mut Cursive, row, is_collapsed, children| {
            if !is_collapsed && children == 0 {
                siv.call_on_name("tree", move |treeview: &mut TreeView<TreeEntry<G>>| {
                    expand_tree(treeview, row);
                });
            }
        });

        let right = LinearLayout::vertical().child(Panel::new(treeview.with_name("tree")));

        log::info!("Welcome to breakthrough!");

        let bt = LinearLayout::horizontal()
            .child(left)
            .weight(1)
            .child(middle)
            .weight(2)
            .child(ResizedView::with_min_width(60, right))
            .weight(1);

        self.siv
            .add_layer(Dialog::new().title("Breakthrough").content(bt));
        GuiEventSender {
            f: self.siv.cb_sink().clone(),
        }
    }
}

/// AlphaZero event loop, managing the game instance.
async fn event_loop_alpha<GV, PB2>(
    initial_state: GV::G,
    pb1: PUCT,
    pb2: PB2,
    rx: mpsc::Receiver<GuiToSimChannel>,
    tx: GuiEventSender,
) where
    PB2: MultiplayerPolicyBuilder<GV::G>,
    GV: GameView,
    GV::G: Game + SingleWinner + Features + Clone,
{
    let mut state = initial_state;

    let mut p1 = pb1.create(<GV::G as Game>::players()[0]);
    let mut p2 = pb2.create(<GV::G as Game>::players()[1]);

    while rx.recv().ok().is_some() {
        // at each step a Next is received
        if !state.is_finished() {
            let p1_to_play = state.turn() == <GV::G as Game>::players()[0];

            let action = if p1_to_play {
                let action = p1.play(&state).await;
                /* UPDATE TREE VIEW*/
                let root_node = p1.root.take().unwrap();
                let count = root_node.read().unwrap().info.node.count;
                let root_value: f32 = root_node
                    .read()
                    .unwrap()
                    .info
                    .moves
                    .iter()
                    .map(|(_, v)| (v.reward + 0.997 * v.Q * v.N_a / count)) // TODO: not hardcode discount.
                    .sum();

                tx.send(move |ui: &mut GameDuelUI<GV, GV::G>| {
                    ui.new_policy_tree(root_node, root_value, count)
                });

                /* UPDATE STATE*/
                action
            } else {
                p2.play(&state).await
            };
            log::info!("{:?}", action);
            state.play(&action).await;

            let state = state.clone();
            tx.send(move |ui: &mut GameDuelUI<GV, GV::G>| ui.new_state(state));
        };

        if state.is_finished() {
            log::info!("Game is finished! {:?} won.", state.winner());
        } else {
            log::info!("Turn to {:?}.", state.turn());
        }
    }
}

/// MuZero event loop, managing the game instance.
async fn event_loop_muz<GV, PB2>(
    initial_state: GV::G,
    pb1: Muz,
    pb2: PB2,
    rx: mpsc::Receiver<GuiToSimChannel>,
    tx: GuiEventSender,
) where
    PB2: MultiplayerPolicyBuilder<GV::G>,
    GV: GameView,
    GV::G: Game + SingleWinner + Features + Clone,
{
    let mut state = initial_state;

    let mut p1 = pb1.create(<GV::G as Game>::players()[0]);
    let mut p2 = pb2.create(<GV::G as Game>::players()[1]);

    while rx.recv().ok().is_some() {
        // at each step a Next is received
        if !state.is_finished() {
            let p1_to_play = state.turn() == <GV::G as Game>::players()[0];

            let action = if p1_to_play {
                let action = p1.play(&state).await;
                /* UPDATE TREE VIEW*/
                let mut muz_puct = p1.mcts.take().unwrap();
                let root_node = muz_puct.root.take().unwrap();
                let visit_count = root_node.read().unwrap().info.node.count;

                log::info!(
                    "Min/max: {}/{}",
                    muz_puct.base_mcts.min_tree,
                    muz_puct.base_mcts.max_tree
                );

                let root_value: f32 = root_node
                    .read()
                    .unwrap()
                    .info
                    .moves
                    .iter()
                    .map(|(_, v)| (v.reward + 0.997 * v.Q * v.N_a / visit_count)) // TODO: not hardcode discount.
                    .sum();

                tx.send(move |ui: &mut GameDuelUI<GV, Simulated<GV::G>>| {
                    ui.new_policy_tree(root_node, root_value, visit_count)
                });

                /* UPDATE STATE*/
                action
            } else {
                p2.play(&state).await
            };
            log::info!("{:?}", action);
            state.play(&action).await;

            let state = state.clone();
            tx.send(move |ui: &mut GameDuelUI<GV, Simulated<GV::G>>| ui.new_state(state));
        };

        if state.is_finished() {
            log::info!("Game is finished! {:?} won.", state.winner());
        } else {
            log::info!("Turn to {:?}.", state.turn());
        }
    }
}

type Result<T> = std::result::Result<T, Box<dyn error::Error>>;

/// Dispatch cursive instance according to the chosen method.
fn run_cursive<GV>(config: Config, initial_state: GV::G, view: GV, method: Method)
where
    GV: GameView,
    GV::G: Game + SingleWinner + Features + Clone + Eq + Hash + 'static,
{
    let mut siv = Cursive::default();
    siv.set_fps(0);

    Logger::with_env_or_str("info")
        .log_target(LogTarget::Writer(
            cursive_flexi_logger_view::cursive_flexi_logger(&siv),
        ))
        .suppress_timestamp()
        .format(flexi_logger::colored_with_thread)
        .start()
        .expect("failed to initialize logger!");

    let mut threaded_rt = tokio::runtime::Builder::new()
        .threaded_scheduler()
        .enable_all()
        .core_threads(2)
        .build()
        .unwrap();

    let ft = initial_state.get_features();
    let action_shape = <GV::G as Features>::action_dimension(&ft);
    let board_shape = <GV::G as Features>::state_dimension(&ft);

    let (tx, rx) = mpsc::channel();

    match method {
        Method::AlphaZero => {
            let gui_events = GameDuelUI::<GV, GV::G>::new(&mut siv).render(view, tx);

            if let Some(mut alpha_config) = config.get_alphazero(action_shape, board_shape) {
                std::thread::spawn(move || {
                    threaded_rt
                        .block_on(async {
                            alpha_config.watch_models = false;
                            alpha_config.batch_size = 1;

                            let alpha_evals = AlphaZeroEvaluators::new(alpha_config.clone(), true);

                            let puct = PUCT {
                                config: alpha_config.puct,
                                n_playouts: config.mcts.playouts,
                                prediction_channel: alpha_evals.get_channel(),
                            };

                            let pb2 = PPA::<GV::G, NoFeatures>::new(config.policies.ppa);

                            let b = tokio::spawn(event_loop_alpha::<GV, _>(
                                initial_state,
                                puct,
                                pb2,
                                rx,
                                gui_events,
                            ));
                            b.await
                        })
                        .unwrap();
                });
            } else {
                panic!("AlphaZero unsupported for this game.")
            }
        }
        Method::MuZero => {
            let gui_events = GameDuelUI::<GV, Simulated<GV::G>>::new(&mut siv).render(view, tx);

            if let Some(mut mu_config) = config.get_muzero(action_shape, board_shape) {
                std::thread::spawn(move || {
                    threaded_rt
                        .block_on(async {
                            mu_config.watch_models = false;
                            mu_config.batch_size = 1;

                            let mu_evals = MuzEvaluators::new(mu_config.clone(), true);

                            let muz = Muz {
                                muz: mu_config.muz,
                                n_playouts: config.mcts.playouts,
                                channels: mu_evals.get_channels(),
                            };

                            let pb2 = PPA::<GV::G, NoFeatures>::new(config.policies.ppa);

                            let b = tokio::spawn(event_loop_muz::<GV, _>(
                                initial_state,
                                muz,
                                pb2,
                                rx,
                                gui_events,
                            ));
                            b.await
                        })
                        .unwrap();
                });
            } else {
                panic!("AlphaZero unsupported for this game.")
            }
        }
    }

    siv.run();
}

/// Use MuZero with remote Gym.
async fn run_gym(config: Config, mut game_builder: GymBuilder, method: Method) {
    if let Method::MuZero = method {
        game_builder.render = false;
        let state = SingleplayerGameBuilder::create(&game_builder).await;
        game_builder.render = true;

        let ft = state.get_features();
        let action_shape = Gym::action_dimension(&ft);
        let board_shape = Gym::state_dimension(&ft);

        drop(state);

        if let Some(mut mu_config) = config.get_muzero(action_shape, board_shape) {
            mu_config.watch_models = false;
            mu_config.batch_size = 1;

            let mu_evals = MuzEvaluators::new(mu_config.clone(), true);

            let muz = Muz {
                muz: mu_config.muz,
                n_playouts: config.mcts.playouts,
                channels: mu_evals.get_channels(),
            };

            let mut muz_p: MuzPolicy<Gym> = muz.create(0);

            loop {
                let mut state = SingleplayerGameBuilder::create(&game_builder).await;

                while !state.is_finished() {
                    let action = muz_p.play(&state).await;
                    state.play(&action).await;
                    tokio::time::delay_for(std::time::Duration::from_millis(1000)).await;
                }
            }
        }
    } else {
        panic!("PUCT not supported.");
    }
}

/// Entry point.
fn main() -> Result<()> {
    let args = App::new("ggpf-generate")
        .arg(
            Arg::with_name("method")
                .short("m")
                .long("method")
                .takes_value(true)
                .possible_values(&["alpha", "mu"]),
        )
        .arg(
            Arg::with_name("config")
                .short("c")
                .long("config")
                .takes_value(true),
        )
        .get_matches();

    let config_file = format!("config/{}.toml", args.value_of("config").unwrap());
    let config = fs::read_to_string(config_file)?;

    let config: Config = toml::from_str(&config)?;

    let method: Method = match args.value_of("method").unwrap() {
        "alpha" => Method::AlphaZero,
        "mu" => Method::MuZero,
        _ => panic!("Unknown method"),
    };

    let mut threaded_rt = tokio::runtime::Builder::new().build()?;

    match config.game.clone() {
        settings::Game::Breakthrough { size, history } => {
            if let Some(history) = history {
                let game_builder = WithHistoryGB::new(BreakthroughBuilder { size }, history);
                let initial_state =
                    threaded_rt.block_on(game_builder.create(breakthrough::Color::Black));
                run_cursive(
                    config,
                    initial_state.clone(),
                    IWithHistory::new(IBreakthrough::new(initial_state.state)),
                    method,
                )
            } else {
                let game_builder = BreakthroughBuilder { size };
                let initial_state =
                    threaded_rt.block_on(game_builder.create(breakthrough::Color::Black));
                run_cursive(
                    config,
                    initial_state.clone(),
                    IBreakthrough::new(initial_state),
                    method,
                )
            }
        }
        settings::Game::Gym {
            name,
            remote,
            history,
        } => {
            let gb = GymBuilder {
                address: remote,
                game_name: name,
                render: true,
            };

            let mut threaded_rt = tokio::runtime::Builder::new()
                .threaded_scheduler()
                .enable_all()
                .core_threads(2)
                .build()
                .unwrap();

            if let Some(_history) = history {
                panic!("History not supported yet.");
            } else {
                let game_builder = gb;
                threaded_rt.block_on(run_gym(config, game_builder, method))
            }
        }
    };
    Ok(())
}
