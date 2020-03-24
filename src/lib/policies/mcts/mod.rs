use crate::game;
use crate::game::Game;
use crate::policies::MultiplayerPolicy;

use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::FromIterator;
use std::marker::PhantomData;
//pub mod muz;
pub mod puct;
pub mod rave;
pub mod uct;

pub trait MCTSGame = Game + Clone;
/* ABSTRACT MCTS */

#[derive(Clone)]
pub struct MCTSTree<G: MCTSGame, MCTS: BaseMCTSPolicy<G>> {
    pub moves: HashMap<G::Move, Box<MCTSTree<G, MCTS>>>,
    pub info: MCTSNode<G, MCTS>,
}

impl<G, MCTS> Debug for MCTSTree<G, MCTS>
where
    G: MCTSGame,
    MCTS: BaseMCTSPolicy<G>,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        writeln!(fmt, "{:?} ===> {:?}", self.moves, self.info)
    }
}

#[derive(Clone)]
pub struct MCTSNode<G: MCTSGame, MCTS: BaseMCTSPolicy<G>> {
    pub state: G,
    pub node: MCTS::NodeInfo,
    pub moves: HashMap<G::Move, MCTS::MoveInfo>,
}

impl<G, MCTS> Debug for MCTSNode<G, MCTS>
where
    G: MCTSGame,
    MCTS: BaseMCTSPolicy<G>,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(fmt, "NODE: {:?}|| MOVES:{:?}", self.node, self.moves)
    }
}

pub trait BaseMCTSPolicy<G: MCTSGame>: Sized {
    type NodeInfo: Debug + Clone + Copy;
    type MoveInfo: Debug + Clone + Copy;
    type PlayoutInfo;

    /*
     * Get value associated to node
     */
    fn get_value(
        &self,
        board: &G,
        action: &G::Move,
        node_info: &Self::NodeInfo,
        move_info: &Self::MoveInfo,
        exploration: bool,
    ) -> f32;
    /*
     * Create a new node for given state
     */
    fn default_node(&self, board: &G) -> Self::NodeInfo;
    /*
     * Create a new node for given move.
     */
    fn default_move(&self, board: &G, action: &G::Move) -> Self::MoveInfo;

    fn backpropagate_new_node(
        &self,
        node: &mut MCTSNode<G, Self>,
        history: &[G::Move],
        playout: &Self::PlayoutInfo,
    );
    /*
     *
     */
    fn backpropagate(
        &self,
        index: usize,
        node: &mut MCTSNode<G, Self>,
        action: &G::Move,
        history: &[G::Move],
        playout: &Self::PlayoutInfo,
    );

    fn simulate(&self, board: &G) -> Self::PlayoutInfo;
}

use float_ord::FloatOrd;

pub struct WithMCTSPolicy<G: MCTSGame, MCTS: BaseMCTSPolicy<G>> {
    base_mcts: MCTS,
    N_PLAYOUTS: usize,
    pub root: Option<MCTSTree<G, MCTS>>,
    _g: std::marker::PhantomData<G>,
}

impl<G, MCTS> WithMCTSPolicy<G, MCTS>
where
    G: MCTSGame + Clone,
    MCTS: BaseMCTSPolicy<G>,
{
    fn select_move<'a>(&self, tree_node: &'a MCTSTree<G, MCTS>, exploration: bool) -> &'a G::Move {
        tree_node
            .info
            .moves
            .iter()
            .map(|(action, move_info)| {
                (
                    action,
                    self.base_mcts.get_value(
                        &tree_node.info.state,
                        action,
                        &tree_node.info.node,
                        &move_info,
                        exploration,
                    ),
                )
            })
            .max_by_key(|x| FloatOrd(x.1))
            .unwrap()
            .0
    }

    fn select<'a>(
        &self,
        root: &'a mut MCTSTree<G, MCTS>,
    ) -> (Vec<G::Move>, &'a mut MCTSTree<G, MCTS>) {
        let mut history: Vec<G::Move> = Vec::new();

        //let mut tree_pos = Some(root);
        let mut last_node = root;

        loop {
            if last_node.info.state.is_finished() {
                /* we're at a leaf node. */
                return (history, last_node);
            } else {
                /* play next move */
                let a = *self.select_move(last_node, true);
                history.push(a);
                let node_imm = last_node.moves.get(&a);
                if let Some(node) = node_imm {
                    if node.info.state.is_finished() {
                        return (history, last_node);
                    } else {
                        let node = last_node.moves.get_mut(&a).unwrap();
                        last_node = node.borrow_mut();
                    }
                } else {
                    return (history, last_node);
                }
            }
        }
    }

    fn expand<'a>(
        &mut self,
        tree_node: &'a mut MCTSTree<G, MCTS>,
        action: &G::Move,
    ) -> &'a mut MCTSTree<G, MCTS> {

        let mut new_state = tree_node.info.state.clone();
        new_state.play(action);


        let new_node = self.base_mcts.default_node(&new_state);

        let moves_info = HashMap::from_iter(
            new_state
                .possible_moves().iter()
                .map(|m| (*m, self.base_mcts.default_move(&new_state, &m))),
        );

        tree_node.moves.insert(
            *action,
            Box::new(MCTSTree {
                moves: HashMap::new(),
                info: MCTSNode {
                    moves: moves_info,
                    node: new_node,
                    state: new_state,
                },
            }),
        );
        tree_node.moves.get_mut(action).unwrap()
    }

    fn tree_search(&mut self, root: &mut MCTSTree<G, MCTS>) {
        let (history, last_node) = self.select(root);
        let created_node = self.expand(last_node, history.last().unwrap());
        let playout = self.base_mcts.simulate(&created_node.info.state);

        /* BACKUP */
        self.base_mcts
            .backpropagate_new_node(&mut created_node.info, &history, &playout);

        let mut tree_node = Some(root);
        for (index, action) in history.iter().enumerate() {
            self.base_mcts.backpropagate(
                index,
                &mut tree_node.as_mut().unwrap().info,
                &action,
                &history,
                &playout,
            );
            let new_tree_node = tree_node.and_then(|x| {
                x
                .moves
                .get_mut(&action)
                .map(|x| x.as_mut())
            });
            tree_node = new_tree_node;
        }
    }

    pub fn new(p: MCTS, N_PLAYOUTS: usize) -> Self {
        WithMCTSPolicy {
            base_mcts: p,
            N_PLAYOUTS,
            root: None,
            _g: PhantomData,
        }
    }
}

impl<G, MCTS> MultiplayerPolicy<G> for WithMCTSPolicy<G, MCTS>
where
    G: MCTSGame,
    MCTS: BaseMCTSPolicy<G>,
{
    fn play(&mut self, board: &G) -> G::Move {
        let mut root = MCTSTree {
            info: MCTSNode {
                state: board.clone(),
                node: self.base_mcts.default_node(board),
                moves: HashMap::from_iter(
                    board
                        .possible_moves().iter()
                        .map(|m| (*m, self.base_mcts.default_move(board, m))),
                ),
            },
            moves: HashMap::new(),
        };

        let playout = self.base_mcts.simulate(board);
        self.base_mcts.backpropagate_new_node(&mut root.info, &[], &playout);

        for _ in 0..self.N_PLAYOUTS {
            //println!("####> {} | {:?}", i, root);
            self.tree_search(&mut root)
        }

        let chosen_move = *self.select_move(&root, false);
        self.root = Some(root);

        chosen_move
    }
}
