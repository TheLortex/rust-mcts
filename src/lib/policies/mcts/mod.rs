use crate::game::{Game, Base};
use crate::policies::MultiplayerPolicy;

use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::FromIterator;
use std::marker::PhantomData;

pub mod muz;
pub mod puct;
pub mod rave;
pub mod uct;

pub trait MCTSGame = Game + Clone;
/* ABSTRACT MCTS */

use std::rc::{Rc, Weak};
use std::cell::RefCell;


pub type MCTSNodeParent<G, MCTS> = Option<(Weak<RefCell<MCTSTree<G, MCTS>>>, <G as Base>::Move)>;
pub type MCTSNodeChild<G, MCTS>  = Rc<RefCell<MCTSTree<G, MCTS>>>;

#[derive(Clone)]
pub struct MCTSTree<G, MCTS>
where
    G: MCTSGame, 
    MCTS: BaseMCTSPolicy<G> 
{
    pub parent: MCTSNodeParent<G, MCTS>,
    pub moves: HashMap<G::Move, MCTSNodeChild<G, MCTS>>,
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
    pub reward: f32,
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

    /*
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
    );*/

    fn backpropagate(
        &mut self,
        leaf: Rc<RefCell<MCTSTree<G, Self>>>,
        history: &[G::Move],
        playout: Self::PlayoutInfo
    );

    fn simulate(&self, board: &G) -> Self::PlayoutInfo;
}

use float_ord::FloatOrd;

pub struct WithMCTSPolicy<G: MCTSGame, MCTS: BaseMCTSPolicy<G>> {
    base_mcts: MCTS,
    N_PLAYOUTS: usize,
    pub root: Option<Rc<RefCell<MCTSTree<G, MCTS>>>>,
    _g: std::marker::PhantomData<G>,
}

impl<G, MCTS> WithMCTSPolicy<G, MCTS>
where
    G: MCTSGame + Clone,
    MCTS: BaseMCTSPolicy<G>,
{
    fn select_move(&self, tree_node: &MCTSTree<G, MCTS>, exploration: bool) -> G::Move {
        *tree_node
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

    fn select(
        &self,
        root: MCTSNodeChild<G, MCTS>,
    ) -> (Vec<G::Move>, MCTSNodeChild<G, MCTS>) {
        let mut history: Vec<G::Move> = Vec::new();

        //let mut tree_pos = Some(root);
        let mut last_node = root;

        loop {
            let last_node_clone = last_node.clone();
            let last_node_ref = last_node_clone.borrow();
            if last_node_ref.info.state.is_finished() {
                /* we're at a leaf node. */
                return (history, last_node);
            } else {
                /* play next move */
                let a = self.select_move(&last_node_ref, true);
                history.push(a);
                
                let node_imm = last_node_ref.moves.get(&a);
                if let Some(node) = node_imm {
                    if node.borrow().info.state.is_finished() {
                        return (history, last_node);
                    } else {
                        let node = last_node_ref.moves.get(&a).unwrap();
                        last_node = node.clone();
                    }
                } else {
                    return (history, last_node);
                }
            }
        }
    }

    fn expand(
        &mut self,
        tree_node: MCTSNodeChild<G, MCTS>,
        action: &G::Move,
    ) -> MCTSNodeChild<G, MCTS> {

        let mut new_state = tree_node.borrow().info.state.clone();
        let reward = new_state.play(action);


        let new_node = self.base_mcts.default_node(&new_state);

        let moves_info = HashMap::from_iter(
            new_state
                .possible_moves().iter()
                .map(|m| (*m, self.base_mcts.default_move(&new_state, &m))),
        );

        tree_node.borrow_mut().moves.insert(
            *action,
            Rc::new(RefCell::new(MCTSTree {
                parent: Some((Rc::downgrade(&tree_node), *action)),
                moves: HashMap::new(),
                info: MCTSNode {
                    reward,
                    moves: moves_info,
                    node: new_node,
                    state: new_state,
                },
            })),
        );
        tree_node.borrow().moves.get(action).unwrap().clone()
    }

    fn tree_search(&mut self, root: MCTSNodeChild<G, MCTS>) {
        let (history, last_node) = self.select(root);
        let created_node = self.expand(last_node, history.last().unwrap());
        let playout = self.base_mcts.simulate(&created_node.borrow().info.state);

        /* BACKUP */
        self.base_mcts.backpropagate(created_node, &history, playout);
        /*
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
        }*/
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
        let root = Rc::new(RefCell::new(MCTSTree {
            parent: None,
            info: MCTSNode {
                reward: 0.,
                state: board.clone(),
                node: self.base_mcts.default_node(board),
                moves: HashMap::from_iter(
                    board
                        .possible_moves().iter()
                        .map(|m| (*m, self.base_mcts.default_move(board, m))),
                ),
            },
            moves: HashMap::new(),
        }));

        let playout = self.base_mcts.simulate(board);
        self.base_mcts.backpropagate(root.clone(), &[], playout);

        for _ in 0..self.N_PLAYOUTS {
            //println!("####> {} | {:?}", i, root);
            self.tree_search(root.clone())
        }

        let chosen_move = self.select_move(&root.borrow(), false);
        self.root = Some(root);

        chosen_move
    }
}
