use crate::game::{Game, Playout, SingleWinner};
use crate::policies::{
    mcts::{BaseMCTSPolicy, MCTSTreeNode, WithMCTSPolicy},
    MultiplayerPolicyBuilder,
};
use crate::settings;

use async_trait::async_trait;
use std::f32;
use std::fmt;
use std::sync::{Arc, RwLock};

/* UCT */

/// UCT move information.
#[derive(Debug, Clone, Copy)]
pub struct UCTMoveInfo {
    /// Node value.
    pub Q: f32,
    /// Number of times visited.
    pub N_a: f32,
}

/// UCT node information.
#[derive(Debug, Clone, Copy)]
pub struct UCTNodeInfo {
    /// Number of times visited
    pub count: f32,
}

/// UCT policy description.
pub struct UCTPolicy_<G: Game> {
    color: G::Player,
    uct_weight: f32,
}

#[async_trait]
impl<G> BaseMCTSPolicy<G> for UCTPolicy_<G>
where
    G::Move: Send,
    G: super::MCTSGame + SingleWinner,
{
    type NodeInfo = UCTNodeInfo;
    type MoveInfo = UCTMoveInfo;
    type PlayoutInfo = bool;

    fn get_value(
        &self,
        board: &G,
        _action: &G::Move,
        node_info: &Self::NodeInfo,
        move_info: &Self::MoveInfo,
        exploration: bool,
    ) -> f32 {
        let move_cb_multiplier = if board.turn() == self.color { 1. } else { -1. };

        let N = node_info.count;
        let cb = self.uct_weight * (N.ln() / (move_info.N_a + 1.)).sqrt();
        let value = if exploration {
            move_info.Q + move_cb_multiplier * cb
        } else {
            move_info.N_a
        };

        move_cb_multiplier * value
    }

    fn default_node(&self, _board: &G) -> Self::NodeInfo {
        UCTNodeInfo { count: 0. }
    }

    fn default_move(&self, _board: &G, _action: &G::Move) -> Self::MoveInfo {
        UCTMoveInfo { Q: 0., N_a: 0. }
    }

    fn backpropagate(
        &mut self,
        leaf: Arc<RwLock<MCTSTreeNode<G, Self>>>,
        _history: &[G::Move],
        playout: Self::PlayoutInfo,
    ) {
        let z = if playout { 1. } else { 0. };

        let mut current_node = leaf;
        while current_node.read().unwrap().parent.is_some() {
            // extract child
            let (tree_pointer, action) = current_node
                .read()
                .unwrap()
                .parent
                .as_ref()
                .map(|(t, a)| (t.upgrade().unwrap(), *a))
                .unwrap();

            current_node = tree_pointer;

            /* Store standard statistics */
            let mut node = current_node.write().unwrap();
            node.info.node.count += 1.;

            let move_info = node.info.moves.get_mut(&action).unwrap();
            move_info.N_a += 1.;
            move_info.Q += (z - move_info.Q) / move_info.N_a;
        }
    }
    /*
    fn backpropagate(
        &self,
        _index: usize,
        info: &mut MCTSNode<G, Self>,
        action: &G::Move,
        history: &[G::Move],
        playout: &Self::PlayoutInfo,
    ) {
        let z = if *playout { 1. } else { 0. };
        info.node.count += 1.;
        let move_info = info.moves.get_mut(action).unwrap();
        move_info.N_a += 1.;
        move_info.Q += (z - move_info.Q) / move_info.N_a;
    }

    fn backpropagate_new_node(
        &self,
        node: &mut MCTSNode<G, Self>,
        history: &[G::Move],
        playout: &Self::PlayoutInfo,
    ) {
    }*/

    async fn simulate(&self, board: &G) -> <Self as BaseMCTSPolicy<G>>::PlayoutInfo {
        board.playout_board(self.color).await.0.winner() == Some(self.color)
    }
}

/// UCT policy as an MCTS policy.
pub type UCTPolicy<G> = WithMCTSPolicy<G, UCTPolicy_<G>>;

/// UCT policy builder.
type UCT = settings::UCT;

impl fmt::Display for UCT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "UCT")?;
        writeln!(f, "|| uct_weight: {}", self.uct_weight)?;
        writeln!(f, "|| N_PLAYOUT: {}", self.playouts)
    }
}

impl<G> MultiplayerPolicyBuilder<G> for UCT
where
    G::Move: Send,
    G::Player: Send,
    G: super::MCTSGame + SingleWinner,
{
    type P = UCTPolicy<G>;

    fn create(&self, color: G::Player) -> Self::P {
        WithMCTSPolicy::new(
            UCTPolicy_ {
                color,
                uct_weight: self.uct_weight,
            },
            self.playouts,
        )
    }
}
