#![allow(non_snake_case)]

use ggpf::game::breakthrough::*;
use ggpf::game::meta::with_history::{WithHistory, WithHistoryGB};
use ggpf::game::meta::simulated::Simulated;
use ggpf::game::*;
use ggpf::policies::mcts::MCTSTreeNode;
use ggpf::policies::{mcts::puct::*, mcts::muz::*, ppa::*, MultiplayerPolicy, MultiplayerPolicyBuilder};
use ggpf::settings;

use cursive::traits::*;
use cursive::view::SizeConstraint;
use cursive::views::ViewRef;
use cursive::views::{Button, Dialog, LinearLayout, NamedView, Panel, ResizedView};
use cursive::Cursive;
use cursive_flexi_logger_view::FlexiLoggerView;
use cursive_tree_view::{Placement, TreeView};
use flexi_logger::{LogTarget, Logger};
use std::fmt;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::RwLock;
use typenum::U2;

const MODEL_PATH: &str = "models/mu-breakthrough/";

type G  = WithHistory<Breakthrough, U2>;
type GV = ui::IBreakthrough;

type SG = Simulated<G>;

#[derive(Clone)]
struct TreeEntry {
    name: String,
    state: Arc<RwLock<MCTSTreeNode<SG, PUCTPolicy_<SG>>>>,
    probability: f32,
    value: f32,
    N_visits: f32,
    reward: f32,
}

impl fmt::Display for TreeEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} | P{:^2.2} | V{:^2.2} | N{:^4} | R{:^2}",
            self.name, self.probability, self.value, self.N_visits, self.reward
        )
    }
}

impl fmt::Debug for TreeEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} | P:{:^2.2} | V:{:^2.2} | N:{:^4} | R:{:^2}",
            self.name, self.probability, self.value, self.N_visits, self.reward
        )
    }
}

fn expand_tree(treeview: &mut TreeView<TreeEntry>, parent_row: usize) {
    let content: TreeEntry = treeview.borrow_item(parent_row).unwrap().clone();

    let tree_node = content.state.read().unwrap();

    let mut moves: Vec<&Move> = tree_node.moves.iter().map(|(a, _)| a).collect();
    moves.sort_by_key(|a| (a.x, a.y));
    for action in moves {
        let move_info = tree_node.info.moves.get(action).unwrap();
        let state = tree_node.moves.get(action).unwrap().clone();

        let item = TreeEntry {
            name: action.name(),
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
enum GuiToSimChannel {
    Next,
}

struct GuiEventSender {
    f: crossbeam_channel::Sender<Box<dyn FnOnce(&mut Cursive) + Send>>,
}
impl GuiEventSender {
    fn send<F>(&self, f: F)
    where
        F: FnOnce(&mut GameDuelUI) + Send + 'static,
    {
        self.f
            .send(Box::new(move |cursive| f(&mut GameDuelUI::new(cursive))))
            .expect("gui is gone?");
    }
}

struct GameDuelUI<'a> {
    siv: &'a mut Cursive,
}

impl<'a> GameDuelUI<'a> {
    fn new(siv: &'a mut Cursive) -> Self {
        Self { siv }
    }

    fn new_state(&mut self, state: G) {
        let mut view: ViewRef<GV> = self.siv.find_name("game").unwrap();
        view.set_state(state.state);
    }

    fn new_policy_tree(
        &mut self,
        root_node: Arc<RwLock<MCTSTreeNode<SG, PUCTPolicy_<SG>>>>,
        root_value: f32,
        count: f32,
    ) {
        let mut treeview: ViewRef<TreeView<TreeEntry>> = self.siv.find_name("tree").unwrap();


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

    fn render(
        &mut self,
        initial_state: G,
        game_simulator_sender: mpsc::Sender<GuiToSimChannel>,
    ) -> GuiEventSender {
        let state = GV::new(initial_state.state);

        let left = LinearLayout::vertical().child(ResizedView::new(
            SizeConstraint::AtMost(100),
            SizeConstraint::Free,
            FlexiLoggerView::scrollable(),
        ));

        let middle = LinearLayout::vertical()
            .child(NamedView::new("game", state))
            .child(Button::new_raw("Next", move |_s| {
                game_simulator_sender.send(GuiToSimChannel::Next).unwrap()
            }));

        let mut treeview = TreeView::<TreeEntry>::new();

        treeview.set_on_collapse(move |siv: &mut Cursive, row, is_collapsed, children| {
            if !is_collapsed && children == 0 {
                siv.call_on_name("tree", move |treeview: &mut TreeView<TreeEntry>| {
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

/*
start: <G as Game>::Player,
pb1: PUCT<G>,
pb2: P2,
*/
async fn event_loop<PB2>(
    initial_state: G,
    pb1: Muz,
    pb2: PB2,
    rx: mpsc::Receiver<GuiToSimChannel>,
    tx: GuiEventSender,
) where
    PB2: MultiplayerPolicyBuilder<G>,
{
    let mut state = initial_state;

    let mut p1 = pb1.create(G::players()[0]);
    let mut p2 = pb2.create(G::players()[1]);

    while rx.recv().ok().is_some() {
        if !state.is_finished() {
            let p1_to_play = state.turn() == <G as Game>::players()[0];

            let action = if p1_to_play {
                let action = p1.play(&state).await;
                /* UPDATE TREE VIEW*/
                let mut muz_puct = p1.mcts.take().unwrap();
                let root_node = muz_puct.root.take().unwrap();
                let visit_count = root_node.read().unwrap().info.node.count;

                log::info!("Min/max: {}/{}", muz_puct.base_mcts.min_tree, muz_puct.base_mcts.max_tree);

                let root_value: f32 = root_node
                    .read()
                    .unwrap()
                    .info
                    .moves
                    .iter()
                    .map(|(_, v)|  ( v.reward + 0.997 * v.Q * v.N_a / visit_count)) // TODO: not hardcode discount.
                    .sum();

                tx.send(move |ui: &mut GameDuelUI| ui.new_policy_tree(root_node, root_value, visit_count));

                /* UPDATE STATE*/
                action
            } else {
                p2.play(&state).await
            };
            log::info!("{}", action);
            state.play(&action).await;

            let state = state.clone();
            tx.send(move |ui| ui.new_state(state));
        };

        if state.is_finished() {
            log::info!("Game is finished! {:?} won.", state.winner());
        } else {
            log::info!("Turn to {:?}.", state.turn());
        }
    }
}
fn main() {
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

    let initial_state: G = threaded_rt.block_on((WithHistoryGB::new(&BreakthroughBuilder {})).create(Color::Black));
    let ft = initial_state.get_features();

    let (tx, rx) = mpsc::channel();

    let gui_events = GameDuelUI::new(&mut siv).render(initial_state.clone(), tx);

    std::thread::spawn(move || {

        threaded_rt
            .block_on(async {
                let muz_config = MuZeroConfig {
                    action_shape: G::action_dimension(&ft),
                    board_shape: G::state_dimension(&ft),
                    batch_size: 1,
                    networks_path: MODEL_PATH.into(),
                    repr_board_shape: ndarray::Dim(settings::MUZ_BT_SHAPE),
                    puct: PUCTSettings {
                        DECODE_VALUE: true,
                        ..PUCTSettings::default()
                    },
                    watch_models: false,
                };

                let muz_evals = MuzEvaluators::new(muz_config.clone(), true);

                let puct = Muz {
                    puct: muz_config.puct,
                    N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
                    channels: muz_evals.get_channels(),
                    repr_dimension: muz_config.repr_board_shape
                };

                let pb2 = PPA::<_, NoFeatures>::default();

                let b = tokio::spawn(event_loop(initial_state, puct, pb2, rx, gui_events));
                b.await
            })
            .unwrap();
    });

    siv.run();
}
