use crate::game::{Game, MoveCode, SingleWinner};
use crate::policies::{MultiplayerPolicy, MultiplayerPolicyBuilder};
use crate::settings;

use rand::seq::SliceRandom;
use std::f32;
use std::iter::*;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::hash::Hash;
use std::fmt;

#[derive(Debug)]
struct PPAMoveInfo {
    Q: f32,
    N_a: f32,
}

#[derive(Debug)]
struct PPANodeInfo<G: Game> {
    count: f32,
    moves: HashMap<G::Move, PPAMoveInfo>,
}

/**
 *  PPA policy instance.
 */
pub struct PPAPolicy<G, M> 
where 
    G: Game,
    M: MoveCode<G>
{
    color: G::Player,
    s: PPA<G,M>,
    tree: HashMap<G, PPANodeInfo<G>>,
    playout_policy: HashMap<usize, f32>,

    _m: PhantomData<M>,
}

impl<G, M> PPAPolicy<G, M> 
where 
    G: Game + SingleWinner + Clone + Eq + Hash,
    M: MoveCode<G>
{
    fn next_move(self: &mut PPAPolicy<G, M>, board: &G) -> G::Move {
        let moves = board.possible_moves();

        let chosen_move = moves
            .choose_weighted(&mut rand::thread_rng(), |item| {
                let code = M::code(board, item);
                self.playout_policy.get(&code).unwrap_or(&0.).exp()
            })
            .unwrap();
        *chosen_move
    }

    fn simulate(self: &mut PPAPolicy<G, M>, root_board: &G) {
        let mut board = root_board.clone(); // COPY BOARD
        let history_uct = self.sim_tree(&mut board);

        let mut history_playout = vec![];


        while {!board.is_finished()} {
            let chosen_move = self.next_move(&board);
            board.play(&chosen_move);
            history_playout.push(chosen_move);
        }
        
        let z = board.winner() == Some(self.color);

        self.update(&history_uct, z);
        self.adapt(root_board, &history_uct, &history_playout, z);
    }

    fn adapt(self: &mut PPAPolicy<G,M>, board: &G, history_uct: &[(G, G::Move)], history_playout: &[G::Move], has_won: bool) {
        let mut board = board.clone();
        for (_, action) in history_uct {
            if (board.turn() == self.color) ^ (!has_won) {
                self.policy_update(&board, &action);
            }
            board.play(&action);
        }

        for action in history_playout {
            if (board.turn() == self.color) ^ (!has_won) {
                self.policy_update(&board, &action);
            }
            board.play(&action);
        }
    }

    fn policy_update(self: &mut PPAPolicy<G,M>, board: &G, action: &G::Move) {
        let node = self.playout_policy.entry(M::code(board, action)).or_insert(0.);
        *node += self.s.alpha;

        let z: f32 = board
            .possible_moves().iter()
            .map(|m| self.playout_policy.get(&M::code(board, &m)).unwrap_or(&0.).exp())
            .sum();
                
        for m in board.possible_moves() {
            let move_node = self.playout_policy.entry(M::code(board, &m)).or_insert(0.);
            let v = move_node.exp();
            *move_node -= self.s.alpha * v / z;
        }
    }

    fn update(
        self: &mut PPAPolicy<G, M>,
        history: &[(G, G::Move)],
        has_won: bool,
    ) {
        let z = if has_won { 1. } else { 0. };
        for (state, action) in history.iter() {
            let mut node = self.tree.get_mut(state).unwrap();
            node.count += 1.;
            let mut v = node.moves.get_mut(action).unwrap();
            (*v).N_a += 1.;
            (*v).Q += (z - (*v).Q) / (*v).N_a;
        }
    }

    fn sim_tree(self: &mut PPAPolicy<G, M>, b: &mut G) -> Vec<(G, G::Move)> {
        let mut history: Vec<(G, G::Move)> = Vec::new();

        while !b.is_finished() {
            match self.tree.get(&b) {
                None => {
                    //history.push((s_t, None));
                    self.new_node(&b);
                    return history;
                }
                Some(_node) => {
                    if let Some(a) = self.select_move(&b) {
                        history.push((b.clone(), a));
                        b.play(&a)
                    } else { // surely there was an available move
                        panic!("? {} {:?}", b.possible_moves().len(), b);
                        //history.push((s_t, None));
                        //return history;
                    }
                }
            };
        }
        history
    }

    fn select_move(self: &PPAPolicy<G, M>, board: &G) -> Option<G::Move> {
        let moves = board.possible_moves();
        let node_info = self.tree.get(&board).unwrap();

        let N = node_info.count;
        if board.turn() == self.color {
            let mut max_move = None;
            let mut max_value = 0.;
            for _move in moves {
                let v = node_info.moves.get(&_move).unwrap();
                let value = if v.N_a == 0. {
                    2.0
                } else {
                    v.Q + self.s.UCT_WEIGHT * (N.ln() / v.N_a).sqrt()
                };
                if value >= max_value {
                    max_value = value;
                    max_move = Some(_move);
                }
            }
            max_move
        } else {
            let mut min_move = None;
            let mut min_value = 1.;
            for _move in moves {
                let v = node_info.moves.get(&_move).unwrap();
                let value = if v.N_a == 0. {
                    0.
                } else {
                    v.Q - self.s.UCT_WEIGHT * (N.ln() / v.N_a).sqrt()
                };

                if value <= min_value {
                    min_value = value;
                    min_move = Some(_move);
                }
            }
            min_move
        }
    }

    fn new_node(self: &mut PPAPolicy<G, M>, board: &G) {
        let moves = HashMap::from_iter(
            board
                .possible_moves().iter()
                .map(|m| (*m, PPAMoveInfo { Q: 0., N_a: 0. })),
        );

        self.tree
            .insert(board.clone(), PPANodeInfo { count: 0., moves });
    }
}



impl<G, M> MultiplayerPolicy<G> for PPAPolicy<G, M> 
where 
    G: Game + SingleWinner + Clone + Eq + Hash, 
    M: MoveCode<G>
{
    fn play(self: &mut PPAPolicy<G, M>, board: &G) -> G::Move {
        for _ in 0..self.s.N_PLAYOUTS {
            self.simulate(board)
        }

        let info: &PPANodeInfo<G> = self.tree.get(&board).unwrap();

        let mut best_move = None;
        let mut max_visited = 0.;
        for m in board.possible_moves() {
            let x: &PPAMoveInfo = info.moves.get(&m).unwrap();
            if x.N_a >= max_visited {
                max_visited = x.N_a;
                best_move = Some(m);
            }
        }
        best_move.unwrap()
    }
}


// POLICY BUILDER

/**
 *  Playout Policy Adaptation policy builder.
 */
pub struct PPA<G, M> 
where
    G: Game,
    M: MoveCode<G>
{
    /// Weight for UCT formaula.
    pub UCT_WEIGHT: f32,
    /// Total number of playouts at each step.
    pub N_PLAYOUTS: usize,
    /// α value used in policy gradient.
    pub alpha: f32,
    /// PhantomData, storing move encoder type information.
    pub _m: PhantomData<fn() -> M>,
    /// PhantomData, storing game type information.
    pub _g: PhantomData<fn() -> G>
}

impl<G, M> Copy for PPA<G, M> 
where
    G: Game,
    M: MoveCode<G>
{}

impl<G, M> Clone for PPA<G, M> 
where
    G: Game,
    M: MoveCode<G>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<G, M> Default for PPA<G,M> 
where
    G: Game,
    M: MoveCode<G>
{
    fn default() -> PPA<G,M> {
        PPA::<G,M> {
            alpha: 0.1,
            UCT_WEIGHT: 0.4,
            N_PLAYOUTS: settings::DEFAULT_N_PLAYOUTS,
            _m: PhantomData,
            _g: PhantomData
        }
    }
}

impl<G, M> fmt::Display for PPA<G,M> 
where
    G: Game,
    M: MoveCode<G>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PPA")?;
        writeln!(f, "|| ALPHA: {}", self.alpha)?;
        writeln!(f, "|| UCT_WEIGHT: {}", self.UCT_WEIGHT)?;
        writeln!(f, "|| N_PLAYOUTS: {}", self.N_PLAYOUTS)
    }
} 

impl<G, M> MultiplayerPolicyBuilder<G> for PPA<G,M> 
where
    G: Game + SingleWinner + Clone + Eq + Hash,
    M: MoveCode<G>
{
    type P = PPAPolicy<G, M>;

    fn create(&self, color: G::Player) -> PPAPolicy<G, M> {
        PPAPolicy::<G,M> {
            color,
            s: *self,
            playout_policy: HashMap::new(),
            tree: HashMap::new(),
            _m: PhantomData
        }
    }
}
