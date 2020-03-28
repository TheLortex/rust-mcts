use crate::game::*;
use std::marker::PhantomData;
use std::sync::Arc;

pub struct WithHistory<G: Base, H> {
    prec: Option<Arc<Self>>,
    pub state: G, // TODO: create accessor
    _h: PhantomData<fn() -> H>,
}

use std::fmt;
impl<G: Base, H> fmt::Debug for WithHistory<G, H> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{:?}", self.state)
    }
}

impl<G: Base + Clone, H> Clone for WithHistory<G, H> {
    fn clone(&self) -> Self {
        WithHistory {
            prec: self.prec.clone(),
            state: self.state.clone(),
            _h: PhantomData,
        }
    }
}

impl<G: Base + Clone, H> Base for WithHistory<G, H> {
    type Move = G::Move;

    fn possible_moves(&self) -> Vec<Self::Move> {
        self.state.possible_moves()
    }
}




impl<G: Playable + Clone, H> Playable for WithHistory<G, H> {
    fn play(&mut self, action: &<Self as Base>::Move) -> f32 {
        let prec = self.prec.take();
        let new_node = WithHistory {
            prec,
            state: self.state.clone(),
            _h: PhantomData,
        };
        self.prec = Some(Arc::new(new_node));
        self.state.play(action)
    }
}

impl<G: Game + Clone, H> Game for WithHistory<G, H> {
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

impl<G: SingleWinner + Clone, H> SingleWinner for WithHistory<G, H> {
    fn winner(&self) -> Option<G::Player> {
        self.state.winner()
    }
}

impl<G: Base + PartialEq, H> PartialEq for WithHistory<G, H> {
    fn eq(&self, other: &Self) -> bool {
        self.state.eq(&other.state)
    }
}
impl<G: Base + Eq, H> Eq for WithHistory<G, H> {}

impl<G: Base + Hash, H_> Hash for WithHistory<G, H_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.state.hash(state)
    }
}
/* GUI */

pub struct IWithHistory<IG: InteractiveGame, H> {
    ig: IG, 
    state: WithHistory<IG::G, H>,
}

use cursive::direction::Direction;
use cursive::event::{Event, EventResult};
use cursive::Printer;
use cursive::Vec2;


impl<IG: InteractiveGame + cursive::view::View, H: 'static> cursive::view::View for IWithHistory<IG,H> 
where
    IG::G: Clone
{
    fn draw(&self, printer: &Printer<'_, '_>) {
        self.ig.draw(printer)
    }

    fn take_focus(&mut self, _source: Direction) -> bool {
        true
    }

    fn on_event(&mut self, evt: Event) -> EventResult {
        self.ig.on_event(evt)
    }

    fn required_size(&mut self, constraint: Vec2) -> Vec2 {
        self.ig.required_size(constraint)
    }
}

impl<IG: InteractiveGame, H: 'static> InteractiveGame for IWithHistory<IG,H>
where 
    IG::G: Game + Clone
{
    type G = WithHistory<IG::G, H>;

    fn new(turn: <Self::G as Game>::Player) -> Self {
        let ig = IG::new(turn);
        let state = WithHistory {
            prec: None,
            state: ig.get().clone(),
            _h: PhantomData
        };
        Self {
            ig,
            state,
        }
    }

    fn get(&self) -> &Self::G {
        &self.state
    }

    fn play(&mut self, action: &<Self::G as Base>::Move) {
        self.ig.play(action);
        
        let prec = self.state.prec.take();
        let new_node = WithHistory {
            prec,
            state: self.state.state.clone(),
            _h: PhantomData,
        };
        self.state.prec = Some(Arc::new(new_node));
        self.state.state = self.ig.get().clone();
    }
/*
    fn choose_move<'a>(&'a mut self, cb: Box<dyn 'a + FnOnce(<Self::G as Base>::Move, &mut Self)>) {
        self.ig.choose_move(Box::new(move |action, ig| cb(action, &mut self)))
    }*/
}


/* GAME BUILDER */
#[derive(Copy,Clone)]
pub struct WithHistoryGB<'a, GB, H> (&'a GB, PhantomData<H>);

impl<'a, GB, H> WithHistoryGB<'a, GB, H> {
    pub fn new(gb: &'a GB) -> Self {
        Self(gb, PhantomData)
    }
}

impl<'a, G: Game + Clone, GB: GameBuilder<G>,H> GameBuilder<WithHistory<G,H>> for WithHistoryGB<'a, GB,H> {
    fn create(&self, starting: G::Player) -> WithHistory<G,H> {
        WithHistory {
            prec: None,
            state: self.0.create(starting),
            _h: PhantomData
        }
    }
}

use typenum::Unsigned;

impl<G: Feature + Clone, H: Unsigned> Feature for WithHistory<G, H> {
    // one dimension larger to store history
    type StateDim = <G::StateDim as Dimension>::Larger;
    type ActionDim = G::ActionDim;

    fn state_dimension(&self) -> Self::StateDim {
        let game_state_dimension = G::state_dimension(&self.state);
        let mut new_dim = game_state_dimension.insert_axis(Axis(0));
        new_dim[0] = H::to_usize();
        new_dim
    }

    fn state_to_feature(&self, pov: Self::Player) -> Array<f32, Self::StateDim> {
        let mut states_ref = vec![];
        (0..H::to_usize()).fold(self, |current, _| {
            states_ref.push(&current.state);
            let res: &Self = current.prec.as_ref().map(|b| b.as_ref()).unwrap_or(current);
            res
        });
        let features_array: Vec<ndarray::Array<f32, Self::StateDim>> = states_ref
            .iter()
            .rev()
            .map(|g| {
                g.state_to_feature(pov).insert_axis(Axis(0))
            })
            .collect();
        let features_array_view: Vec<ndarray::ArrayView<f32, Self::StateDim>> = features_array.iter().map(|x| x.view()).collect();
        ndarray::stack(Axis(0), &features_array_view).expect("All features should have the same shape.")
    }

    fn action_dimension() -> Self::ActionDim {
        G::action_dimension()
    }

    fn moves_to_feature(moves: &HashMap<Self::Move, f32>) -> Array<f32, Self::ActionDim> {
        G::moves_to_feature(moves)
    }

    fn feature_to_moves(&self, features: &Array<f32, Self::ActionDim>) -> HashMap<Self::Move, f32> {
        self.state.feature_to_moves(features)
    }

    fn all_feature_to_moves(features: &Array<f32, Self::ActionDim>) -> HashMap<Self::Move, f32> {
        G::all_feature_to_moves(features)
    }

    fn all_possible_moves() -> Vec<Self::Move> {
        G::all_possible_moves()
    }
}
