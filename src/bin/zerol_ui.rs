#![allow(non_snake_case)]

use cursive::views::{Button, Dialog, ResizedView, LinearLayout, NamedView, Panel};
use cursive::view::SizeConstraint;
use cursive::Cursive;
use cursive::traits::*;
use std::cell::RefCell;


use zerol::game::breakthrough::*;
use zerol::game::{MoveTrait, InteractiveGame, BaseGame, NoFeatures, Feature, MultiplayerGame};
use zerol::policies::{
    ppa::*, mcts::puct::{PUCT, PUCTSettings, PUCTNodeInfo, PUCTMoveInfo}, MultiplayerPolicy, MultiplayerPolicyBuilder,
};
use ndarray::Array;

use std::marker::PhantomData;
use zerol::settings;
use zerol::misc::game_evaluator;

use tensorflow::{Graph, Session, SessionOptions};

use std::rc::Rc;
use cursive_flexi_logger_view::FlexiLoggerView;
use flexi_logger::{Logger, LogTarget};
use std::collections::HashMap;
use std::fmt;

use cursive_tree_view::{Placement, TreeView};

const MODEL_PATH: &str = "models/breakthrough";// todo: put in settings;

type G = Breakthrough;
type IG = ui::IBreakthrough;

#[derive(Debug, Clone)]
struct TreeEntry {
    name: String,
    state: usize,
    probability: f32,
    value: f32,
    N_visits: f32
}

impl fmt::Display for TreeEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} - {:^2.2} - {:^2.2} - {:^4}", self.name, self.probability, self.value, self.N_visits)
    }
}


fn expand_tree(tree: &HashMap<usize, PUCTNodeInfo<G>>, treeview: &mut TreeView<TreeEntry>, parent_row: usize, node: &TreeEntry) {
    if let Some(node) = tree.get(&node.state) {
        let mut moves: Vec<(&Move, &PUCTMoveInfo)> = node.moves.iter().collect();
        moves.sort_by_key(|(a,_)| (a.x, a.y));
        for (action, move_info) in moves {
            let item = TreeEntry {
                name: action.name(),
                state: move_info.target,
                probability: move_info.pi,
                value: move_info.Q,
                N_visits: move_info.N_a,
            };
            if move_info.N_a == 0. {
                treeview.insert_item(item, Placement::LastChild, parent_row);
            } else {
                treeview.insert_container_item(item, Placement::LastChild, parent_row);
            }
        }
    }
}

struct GameDuelUI {}

impl GameDuelUI {
    fn render<P2,F>(
        start: <G as MultiplayerGame>::Player,
        pb1: PUCT<G,F>,
        pb2: P2,
    ) -> impl cursive::view::View 
where
    P2: MultiplayerPolicyBuilder<G>,
    P2::P: 'static,
    F: Fn(<G as MultiplayerGame>::Player, &G) -> (Array<f32, <G as Feature>::ActionDim>, f32) + Clone + 'static
{
        
        let p1 = RefCell::new(pb1.create(G::players()[0]));
        let p2 = RefCell::new(pb2.create(G::players()[1]));

        let state = IG::new(start);


        let tree = Rc::new(RefCell::new(HashMap::new()));

        let left = LinearLayout::vertical()
            .child(ResizedView::new(SizeConstraint::AtMost(100), SizeConstraint::Free, FlexiLoggerView::scrollable()));

        let tree_1 = tree.clone();
        let middle = LinearLayout::vertical()
            .child(NamedView::new("game", state))
            .child(Button::new_raw("Next", move |s| {
                let state: &mut IG = &mut s.find_name("game").unwrap();
                if !state.get().is_finished() {
                    let mut p1 = p1.borrow_mut();
                    let mut p2 = p2.borrow_mut();
                    let p1_to_play = state.get().turn() == <G as MultiplayerGame>::players()[0];

                    let action = if p1_to_play {
                        let action = p1.play(&state.get());
                        /* UPDATE TREE VIEW*/
                        let mut tree = tree_1.borrow_mut();
                        *tree = p1.inner.b.tree.clone();

                        let treeview: &mut TreeView<TreeEntry> = &mut s.find_name("tree").unwrap();
                        treeview.clear();
                        treeview.insert_container_item(TreeEntry {
                            name: "root".to_string(),
                            state: state.get().hash(),
                            probability: 1.,
                            value: 1.,
                            N_visits: tree.get(&state.get().hash()).unwrap().count
                        }, Placement::After, 0);
                        /* UPDATE STATE*/
                        action
                    } else {
                        p2.play(&state.get())
                    };
                    log::info!("{}", action);
                    state.get_mut().play(&action);
                };

                if state.get().is_finished() {
                    log::info!("Game is finished! {:?} won.", state.get().winner());
                } else {
                    log::info!("Turn to {:?}.", state.get().turn());
                }
            }));

        let mut treeview = TreeView::<TreeEntry>::new();
        let tree_2 = tree.clone();
        treeview.set_on_collapse(move |siv: &mut Cursive, row, is_collapsed, children| {
            if !is_collapsed && children == 0 {
                let tree_3 = tree_2.clone();
                
                siv.call_on_name("tree", move |treeview: &mut TreeView<TreeEntry>| {
                    let content: TreeEntry = {
                        treeview.borrow_item(row).unwrap().clone()
                    };
                    expand_tree(&tree_3.borrow(), treeview, row, &content);
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


    Logger::with_env_or_str("info")
        .log_target(LogTarget::Writer(
            cursive_flexi_logger_view::cursive_flexi_logger(&siv)))
        .suppress_timestamp()
        .format(flexi_logger::colored_with_thread)
        .start()
        .expect("failed to initialize logger!");


    let mut graph = Graph::new();
    let session =
        Session::from_saved_model(&SessionOptions::new(), &["serve"], &mut graph, MODEL_PATH)
            .unwrap();

    let boxed_session = Rc::new(Box::new(session));
    let boxed_graph   = Rc::new(Box::new(graph));
    let session = boxed_session.clone();
    let graph   = boxed_graph.clone();

    let puct = PUCT {
        _g: PhantomData,
        s: PUCTSettings::default(),
        N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
        evaluate: move |pov, board: &Breakthrough| {
            game_evaluator(&session, &graph, pov, board)
        },
    };
    
    let pb2 = PPA::<_, NoFeatures>::default();

    siv.add_layer(
        Dialog::new()
            .title("Breakthrough")
            .content(GameDuelUI::render(
                G::players()[1],
                puct,
                pb2,
            )),
    );
    siv.run();
}