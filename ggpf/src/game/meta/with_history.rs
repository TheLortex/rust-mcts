use crate::game::*;
use async_trait::async_trait;
use std::sync::Arc;

/// A game with its history.
#[derive(Clone, Debug)]
pub struct WithHistory<G: Base> {
    prec: Option<Arc<Self>>,
    /// Current game state.
    pub state: G, // TODO: create accessor
    history_len: usize,
}

impl<G: Base + Clone> Base for WithHistory<G> {
    type Move = G::Move;

    fn possible_moves(&self) -> Vec<Self::Move> {
        self.state.possible_moves()
    }
}

#[async_trait]
impl<G: Playable + Clone + Sync + Send> Playable for WithHistory<G> {
    async fn play(&mut self, action: &<Self as Base>::Move) -> f32 {
        let prec = self.prec.take();
        let new_node = WithHistory {
            prec,
            state: self.state.clone(),
            history_len: self.history_len,
        };
        self.prec = Some(Arc::new(new_node));
        self.state.play(action).await
    }
}

impl<G: Game + Clone + Sync + Send> Game for WithHistory<G> {
    type Player = G::Player;

    fn players() -> Vec<Self::Player> {
        G::players()
    }

    fn player_after(player: Self::Player) -> Self::Player {
        G::player_after(player)
    }

    fn turn(&self) -> Self::Player {
        self.state.turn()
    }
}

impl<G: SingleWinner + Clone + Sync + Send> SingleWinner for WithHistory<G> {
    fn winner(&self) -> Option<G::Player> {
        self.state.winner()
    }
}

impl<G: Base + PartialEq> PartialEq for WithHistory<G> {
    fn eq(&self, other: &Self) -> bool {
        self.state.eq(&other.state)
    }
}
impl<G: Base + Eq> Eq for WithHistory<G> {}

impl<G: Base + Hash> Hash for WithHistory<G> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.state.hash(state)
    }
}
/* GAME BUILDER */
/// Builder for a game with history.
#[derive(Clone, Copy)]
pub struct WithHistoryGB<GB>(GB, usize);

impl<GB> WithHistoryGB<GB> {
    /// Creates a game builder with history, given a correspond standard game builder.
    pub fn new(gb: GB, history_len: usize) -> Self {
        Self(gb, history_len)
    }
}

#[async_trait]
impl<GB> GameBuilder for WithHistoryGB<GB>
where
    GB::G: Clone + Sync + Send + 'static,
    GB: GameBuilder + Send + Sync,
{
    type G = WithHistory<GB::G>;

    async fn create(&self, starting: <Self::G as Game>::Player) -> WithHistory<GB::G> {
        let state = self.0.create(starting).await;
        WithHistory {
            prec: None,
            state,
            history_len: self.1,
        }
    }
}

impl<G: Features + Clone + Sync + Send> Features for WithHistory<G> {
    // one dimension larger to store history
    type StateDim = <G::StateDim as Dimension>::Larger;
    type ActionDim = G::ActionDim;

    type Descriptor = (<G::StateDim as Dimension>::Larger, G::Descriptor);

    fn get_features(&self) -> Self::Descriptor {
        let ft = self.state.get_features();

        let state_dimension = {
            let game_state_dimension = G::state_dimension(&ft);
            let mut new_dim = game_state_dimension.insert_axis(Axis(0));
            new_dim[0] = self.history_len;
            new_dim
        };

        (state_dimension, ft)
    }

    fn state_dimension(descr: &Self::Descriptor) -> Self::StateDim {
        descr.0.clone()
    }

    fn action_dimension(descr: &Self::Descriptor) -> Self::ActionDim {
        G::action_dimension(&descr.1)
    }

    fn state_to_feature(&self, pov: Self::Player) -> Array<f32, Self::StateDim> {
        let mut states_ref = vec![];
        (0..self.history_len).fold(self, |current, _| {
            states_ref.push(&current.state);
            let res: &Self = current.prec.as_ref().map(|b| b.as_ref()).unwrap_or(current);
            res
        });
        let features_array: Vec<ndarray::Array<f32, Self::StateDim>> = states_ref
            .iter()
            .rev()
            .map(|g| g.state_to_feature(pov).insert_axis(Axis(0)))
            .collect();
        let features_array_view: Vec<ndarray::ArrayView<f32, Self::StateDim>> =
            features_array.iter().map(|x| x.view()).collect();
        ndarray::stack(Axis(0), &features_array_view)
            .expect("All features should have the same shape.")
    }

    fn moves_to_feature(
        descr: &Self::Descriptor,
        moves: &HashMap<Self::Move, f32>,
    ) -> Array<f32, Self::ActionDim> {
        G::moves_to_feature(&descr.1, moves)
    }

    fn feature_to_moves(&self, features: &Array<f32, Self::ActionDim>) -> HashMap<Self::Move, f32> {
        self.state.feature_to_moves(features)
    }

    fn all_feature_to_moves(
        descr: &Self::Descriptor,
        features: &Array<f32, Self::ActionDim>,
    ) -> HashMap<Self::Move, f32> {
        G::all_feature_to_moves(&descr.1, features)
    }

    fn all_possible_moves(descr: &Self::Descriptor) -> Vec<Self::Move> {
        G::all_possible_moves(&descr.1)
    }
}

/// Interface wrapper for WithHistory.
pub struct IWithHistory<GV>
where
    GV: GameView,
{
    view: GV,
}

use cursive::direction::Direction;
use cursive::event::{Event, EventResult};
use cursive::Printer;
use cursive::Vec2;

impl<GV> IWithHistory<GV>
where
    GV: GameView,
{
    /// Wraps a game view for a with history game.
    pub fn new(view: GV) -> Self {
        Self { view }
    }
}

impl<GV> cursive::view::View for IWithHistory<GV>
where
    GV: GameView,
{
    fn draw(&self, printer: &Printer) {
        self.view.draw(printer)
    }

    fn take_focus(&mut self, d: Direction) -> bool {
        self.view.take_focus(d)
    }

    fn on_event(&mut self, event: Event) -> EventResult {
        self.view.on_event(event)
    }

    fn required_size(&mut self, v: Vec2) -> Vec2 {
        self.view.required_size(v)
    }
}

impl<GV> GameView for IWithHistory<GV>
where
    GV: GameView,
    GV::G: Game + Clone,
{
    type G = WithHistory<GV::G>;

    fn set_state(&mut self, state: Self::G) {
        self.view.set_state(state.state)
    }
}
