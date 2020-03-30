#![allow(non_snake_case)]

use ggpf::deep::evaluator::prediction_evaluator_single;
use ggpf::game::breakthrough::*;
use ggpf::game::meta::with_history::{IWithHistory, WithHistory};
use ggpf::game::{Base, Feature, Game, InteractiveGame, MoveTrait, NoFeatures, SingleWinner};
use ggpf::policies::mcts::MCTSTreeNode;
use ggpf::policies::{
    mcts::puct::{Evaluator, PUCTPolicy_, PUCTSettings, PUCT},
    ppa::*,
    MultiplayerPolicy, MultiplayerPolicyBuilder,
};
use ggpf::settings;

use cursive::traits::*;
use cursive::view::SizeConstraint;
use cursive::views::{Button, Dialog, LinearLayout, NamedView, Panel, ResizedView};
use cursive::Cursive;
use cursive_flexi_logger_view::FlexiLoggerView;
use cursive_tree_view::{Placement, TreeView};
use flexi_logger::{LogTarget, Logger};
use ndarray::Array;
use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;
use tensorflow::{Graph, Session, SessionOptions};
use typenum::U2;

const MODEL_PATH: &str = "models/alpha-breakthrough";

type G = WithHistory<Breakthrough, U2>;
type IG = IWithHistory<ui::IBreakthrough, U2>;

#[derive(Clone)]
struct TreeEntry<F>
where
    F: Evaluator<G>,
{
    name: String,
    state: Rc<RefCell<MCTSTreeNode<G, PUCTPolicy_<G, F>>>>,
    probability: f32,
    value: f32,
    N_visits: f32,
    reward: f32,
}

impl<F> fmt::Display for TreeEntry<F>
where
    F: Evaluator<G>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} | P{:^2.2} | V{:^2.2} | N{:^4} | R{:^2}",
            self.name, self.probability, self.value, self.N_visits, self.reward
        )
    }
}

impl<F> fmt::Debug for TreeEntry<F>
where
    F: Evaluator<G>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} | P:{:^2.2} | V:{:^2.2} | N:{:^4} | R:{:^2}",
            self.name, self.probability, self.value, self.N_visits, self.reward
        )
    }
}

fn expand_tree<F>(treeview: &mut TreeView<TreeEntry<F>>, parent_row: usize)
where
    F: Evaluator<G> + Clone,
{
    let content: TreeEntry<F> = treeview.borrow_item(parent_row).unwrap().clone();

    let tree_node = content.state.borrow();

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

struct GameDuelUI {}

impl GameDuelUI {
    fn render<P2, F>(
        start: <G as Game>::Player,
        pb1: PUCT<G, F>,
        pb2: P2,
    ) -> impl cursive::view::View
    where
        P2: MultiplayerPolicyBuilder<G>,
        P2::P: 'static,
        F: Fn(<G as Game>::Player, &G) -> (Array<f32, <G as Feature>::ActionDim>, f32)
            + Clone
            + 'static,
    {
        let p1 = RefCell::new(pb1.create(G::players()[0]));
        let p2 = RefCell::new(pb2.create(G::players()[1]));

        let state = IG::new(start);

        let left = LinearLayout::vertical().child(ResizedView::new(
            SizeConstraint::AtMost(100),
            SizeConstraint::Free,
            FlexiLoggerView::scrollable(),
        ));

        let middle = LinearLayout::vertical()
            .child(NamedView::new("game", state))
            .child(Button::new_raw("Next", move |s| {
                let state: &mut IG = &mut s.find_name("game").unwrap();
                if !state.get().is_finished() {
                    let mut p1 = p1.borrow_mut();
                    let mut p2 = p2.borrow_mut();
                    let p1_to_play = state.get().turn() == <G as Game>::players()[0];

                    let action = if p1_to_play {
                        let action = p1.play(&state.get());
                        /* UPDATE TREE VIEW*/
                        let root_node = p1.root.take().unwrap();
                        let count = root_node.borrow().info.node.count;

                        let treeview: &mut TreeView<TreeEntry<F>> =
                            &mut s.find_name("tree").unwrap();
                        treeview.clear();
                        treeview.insert_container_item(
                            TreeEntry {
                                name: "root".to_string(),
                                state: root_node,
                                reward: 0.,
                                probability: 1.,
                                value: 1.,
                                N_visits: count,
                            },
                            Placement::After,
                            0,
                        );
                        /* UPDATE STATE*/
                        action
                    } else {
                        p2.play(&state.get())
                    };
                    log::info!("{}", action);
                    state.play(&action);
                };

                if state.get().is_finished() {
                    log::info!("Game is finished! {:?} won.", state.get().winner());
                } else {
                    log::info!("Turn to {:?}.", state.get().turn());
                }
            }));

        let mut treeview = TreeView::<TreeEntry<F>>::new();

        treeview.set_on_collapse(move |siv: &mut Cursive, row, is_collapsed, children| {
            if !is_collapsed && children == 0 {
                siv.call_on_name("tree", move |treeview: &mut TreeView<TreeEntry<F>>| {
                    expand_tree(treeview, row);
                });
            }
        });

        let right = LinearLayout::vertical().child(Panel::new(treeview.with_name("tree")));

        log::info!("Welcome to breakthrough!");

        LinearLayout::horizontal()
            .child(left)
            .weight(1)
            .child(middle)
            .weight(2)
            .child(ResizedView::with_min_width(60, right))
            .weight(1)
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

    let mut graph = Graph::new();
    let session =
        Session::from_saved_model(&SessionOptions::new(), &["serve"], &mut graph, MODEL_PATH)
            .unwrap();

    let session = Rc::new(Box::new(session));
    let graph = Rc::new(Box::new(graph));

    let puct = PUCT {
        _g: PhantomData,
        config: PUCTSettings::default(),
        N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
        evaluate: move |pov, board: &G| {
            prediction_evaluator_single(&session, &graph, pov, board, false)
        },
    };

    let pb2 = PPA::<_, NoFeatures>::default();

    siv.add_layer(
        Dialog::new()
            .title("Breakthrough")
            .content(GameDuelUI::render(G::players()[1], puct, pb2)),
    );
    siv.run();
}
