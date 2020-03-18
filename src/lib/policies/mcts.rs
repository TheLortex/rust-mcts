use std::collections::HashMap;
use std::f32;
use std::iter::*;

use super::super::game::MultiplayerGame;
use super::{MultiplayerPolicy, MultiplayerPolicyBuilder, N_PLAYOUTS};

use float_ord::FloatOrd;
use std::marker::PhantomData;

/* ABSTRACT MCTS */

pub trait MCTSPolicy<G: MultiplayerGame> {
    type NodeInfo;
    type PlayoutInfo;

    fn tree(&self) -> &HashMap<usize, Self::NodeInfo>;
    fn tree_mut(&mut self) -> &mut HashMap<usize, Self::NodeInfo>;

    fn select_move(&self, board: &G, exploration: bool) -> G::Move;

    fn select(&self, board: &mut G) -> Vec<(G, G::Move)> {
        let mut history: Vec<(G, G::Move)> = Vec::new();

        while !board.is_finished() {
            let s_t = board.hash();
            match self.tree().get(&s_t) {
                None => {
                    /* we're at a leaf node. */
                    return history;
                }
                Some(_node) => {
                    /* play next move */
                    let a = self.select_move(&board, true);
                    history.push((board.clone(), a));
                    board.play(&a)
                }
            };
        }
        history
    }

    fn default_node(&self, board: &G) -> Self::NodeInfo;

    fn expand(&mut self, board: &G) {
        let new_node = self.default_node(board);
        self.tree_mut().insert(board.hash(), new_node);
    }

    fn simulate(&self, history: &[G]) -> Self::PlayoutInfo;

    fn backpropagate(&mut self, history: Vec<(G, G::Move)>, playout: Self::PlayoutInfo);

    fn tree_search(&mut self, history: &[G]) {
        let mut b = history.last().unwrap().clone();
        let selection_history = self.select(&mut b);

        let mut game_history = Vec::from(history);
        game_history.extend(selection_history.iter().map(|(g,_)| g.clone()));
        self.expand(&b);
        let playout = self.simulate(&game_history);
        self.backpropagate(selection_history, playout);
    }
}

pub struct WithMCTSPolicy<G: MultiplayerGame, M: MCTSPolicy<G>>(pub M, std::marker::PhantomData<G>);

impl<G: MultiplayerGame, M: MCTSPolicy<G>> WithMCTSPolicy<G, M> {
    pub fn new(p: M) -> Self {
        WithMCTSPolicy(p, PhantomData)
    }
}

impl<G: MultiplayerGame, M: MCTSPolicy<G>> MultiplayerPolicy<G> for WithMCTSPolicy<G, M> {
    fn play(&mut self, history: &[G]) -> G::Move {
        let board = history.last().unwrap();
        for _ in 0..N_PLAYOUTS {
            self.0.tree_search(history)
        }

        self.0.select_move(board, false)
    }
}








/* UCT */

#[derive(Debug)]
pub struct UCTMoveInfo {
    pub Q: f32,
    pub N_a: f32,
}

#[derive(Debug)]
pub struct UCTNodeInfo<G: MultiplayerGame> {
    pub count: f32,
    pub moves: HashMap<G::Move, UCTMoveInfo>,
}

pub struct UCTPolicy_<G: MultiplayerGame> {
    color: G::Player,
    tree: HashMap<usize, UCTNodeInfo<G>>,
    UCT_WEIGHT: f32,
}

impl<G: MultiplayerGame> MCTSPolicy<G> for UCTPolicy_<G> {
    type NodeInfo = UCTNodeInfo<G>;
    type PlayoutInfo = bool;

    fn tree(&self) -> &HashMap<usize, Self::NodeInfo> {
        &self.tree
    }

    fn tree_mut(&mut self) -> &mut HashMap<usize, Self::NodeInfo> {
        &mut self.tree
    }

    fn select_move(&self, board: &G, exploration: bool) -> G::Move {
        let moves = board.possible_moves();
        let node_info = self.tree.get(&board.hash()).unwrap();
        let N = node_info.count;

        // select between optimism and pessimism in the confidence bound.
        let move_cb_multiplier = if board.turn() == self.color { 1. } else { -1. };

        let moves_scores = moves.iter().map(|action| {
            let v = node_info.moves.get(action).unwrap();
            let cb = self.UCT_WEIGHT * (N.ln() / (v.N_a + 1.)).sqrt();
            let value = if exploration {
                v.Q + move_cb_multiplier * cb
            } else {
                v.N_a
            };
            (value, action)
        });

        if board.turn() == self.color {
            *moves_scores.max_by_key(|x| FloatOrd(x.0)).unwrap().1
        } else {
            *moves_scores.min_by_key(|x| FloatOrd(x.0)).unwrap().1
        }
    }

    fn default_node(&self, board: &G) -> Self::NodeInfo {
        let moves = HashMap::from_iter(
            board
                .possible_moves()
                .into_iter()
                .map(|m| (*m, UCTMoveInfo { Q: 0., N_a: 0. })),
        );

        UCTNodeInfo { count: 0., moves }
    }

    fn simulate(&self, history: &[G]) -> Self::PlayoutInfo {
        history.last().unwrap().playout_board().has_won(self.color)
    }

    fn backpropagate(&mut self, history: Vec<(G, G::Move)>, playout: Self::PlayoutInfo) {
        let z = if playout { 1. } else { 0. };
        for (state, action) in history.iter() {
            let mut node = self.tree.get_mut(&state.hash()).unwrap();
            node.count += 1.;

            let mut v = node.moves.get_mut(action).unwrap();
            (*v).N_a += 1.;
            (*v).Q += (z - (*v).Q) / (*v).N_a;
        }
    }
}

pub type UCTPolicy<G> = WithMCTSPolicy<G, UCTPolicy_<G>>;

pub struct UCT {
    UCT_WEIGHT: f32,
}

impl Default for UCT {
    fn default() -> UCT {
        UCT { UCT_WEIGHT: 0.4 }
    }
}

use std::fmt;
impl fmt::Display for UCT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "UCT")?;
        writeln!(f, "|| UCT_WEIGHT: {}", self.UCT_WEIGHT)?;
        writeln!(f, "|| N_PLAYOUT: {}", N_PLAYOUTS)
    }
}

impl<G: MultiplayerGame> MultiplayerPolicyBuilder<G> for UCT {
    type P = UCTPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        WithMCTSPolicy::new(UCTPolicy_ {
            color,
            tree: HashMap::new(),
            UCT_WEIGHT: self.UCT_WEIGHT,
        })
    }
}

/* RAVE */

#[derive(Debug)]
struct MoveInfo {
    wins: f32,
    count: f32,
    wins_AMAF: f32,
    count_AMAF: f32,
}

#[derive(Debug)]
pub struct RAVENodeInfo<G: MultiplayerGame> {
    count: f32,
    moves: HashMap<G::Move, MoveInfo>,
}

pub struct RAVEPolicy_<G: MultiplayerGame> {
    color: G::Player,
    tree: HashMap<usize, RAVENodeInfo<G>>,
    UCT_WEIGHT: f32,
}

impl<G: MultiplayerGame> MCTSPolicy<G> for RAVEPolicy_<G> {
    type NodeInfo = RAVENodeInfo<G>;
    type PlayoutInfo = (bool, Vec<(usize, G::Move)>); // (has_won, history_default)

    fn tree(&self) -> &HashMap<usize, Self::NodeInfo> {
        &self.tree
    }

    fn tree_mut(&mut self) -> &mut HashMap<usize, Self::NodeInfo> {
        &mut self.tree
    }

    fn select_move(&self, board: &G, _exploration: bool) -> G::Move {
        let moves = board.possible_moves();
        // select between optimism and pessimism in the confidence bound.
        let optimistic = board.turn() == self.color;

        let moves_scores = moves.iter().map(|action| {
            let value = self.eval(&board.hash(), &action, optimistic);
            (value, action)
        });

        if board.turn() == self.color {
            *moves_scores.max_by_key(|x| FloatOrd(x.0)).unwrap().1
        } else {
            *moves_scores.min_by_key(|x| FloatOrd(x.0)).unwrap().1
        }
    }

    fn default_node(&self, board: &G) -> Self::NodeInfo {
        let moves = HashMap::from_iter(board.possible_moves().into_iter().map(|m| {
            (
                *m,
                MoveInfo {
                    wins: 0.,
                    wins_AMAF: 0.,
                    count: 0.,
                    count_AMAF: 0.,
                },
            )
        }));
        RAVENodeInfo { count: 0., moves }
    }

    fn simulate(&self, history: &[G]) -> Self::PlayoutInfo {
        let (s, default) = history.last().unwrap().playout_board_history();
        (s.has_won(self.color), default)
    }

    fn backpropagate(
        &mut self,
        history: Vec<(G, G::Move)>,
        (has_won, history_default): Self::PlayoutInfo,
    ) {
        let z = if has_won { 1. } else { 0. };
        let whole_history = [history.iter().map(|(a,b)| (a.hash(), *b)).collect(), history_default].concat();
        for (t, (state, action)) in history.iter().enumerate() {
            let mut node = self.tree.get_mut(&state.hash()).unwrap();
            node.count += 1.;

            let mut v = node.moves.get_mut(action).unwrap();
            (*v).count += 1.;
            (*v).wins += (z - (*v).wins) / (*v).count;

            // compute AMAF statistics
            for u in (t + 2..whole_history.len()).step_by(2) {
                let (_, action_u) = whole_history[u];
                if (t..u).step_by(2).all(|i| action_u != whole_history[i].1) {
                    if let Some(mut v_amaf) = node.moves.get_mut(&action_u) {
                        (*v_amaf).count_AMAF += 1.;
                        (*v_amaf).wins_AMAF += (z - (*v_amaf).wins_AMAF) / (*v_amaf).count_AMAF;
                    }
                }
            }
        }
    }
}

impl<G: MultiplayerGame> RAVEPolicy_<G> {
    fn beta(v: &MoveInfo) -> f32 {
        let b = 0.0001;
        let mut div = v.count_AMAF + v.count + 4. * v.count_AMAF * v.count * b * b;
        if div == 0. {
            div = 1.
        };
        v.count_AMAF / div
    }

    fn eval(self: &RAVEPolicy_<G>, state: &usize, action: &G::Move, optimistic: bool) -> f32 {
        let node_info = self.tree.get(state).unwrap();
        let v = node_info.moves.get(action).unwrap();

        let multiplier = if optimistic { 1. } else { -1. };
        let v_mean =
            v.wins + multiplier * self.UCT_WEIGHT * (node_info.count.ln() / (1. + v.count)).sqrt();
        let v_AMAF = v.wins_AMAF;

        let beta = Self::beta(v);
        (1. - beta) * v_mean + beta * v_AMAF
    }
}

pub type RAVEPolicy<G> = WithMCTSPolicy<G, RAVEPolicy_<G>>;

pub struct RAVE {
    UCT_WEIGHT: f32,
}

impl Default for RAVE {
    fn default() -> RAVE {
        RAVE { UCT_WEIGHT: 0.4 }
    }
}

impl fmt::Display for RAVE {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RAVE")?;
        writeln!(f, "|| UCT_WEIGHT: {}", self.UCT_WEIGHT)?;
        writeln!(f, "|| N_PLAYOUT: {}", N_PLAYOUTS)
    }
}

impl<G: MultiplayerGame> MultiplayerPolicyBuilder<G> for RAVE {
    type P = RAVEPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        WithMCTSPolicy::new(
            RAVEPolicy_ {
                color,
                tree: HashMap::new(),
                UCT_WEIGHT: self.UCT_WEIGHT,
            }
        )
    }
}
