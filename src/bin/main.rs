#![allow(dead_code)]
#![allow(non_snake_case)]
#![feature(associated_type_defaults)]
#![feature(test)]
#![feature(trait_alias)]
#![allow(unused_imports)]
#![feature(fn_traits)]
#![feature(drain_filter)]

use cursive::views::{Dialog,LinearLayout,NamedView,Button};
use cursive::Cursive;

use atomic_counter::{AtomicCounter,RelaxedCounter};

use std::collections::hash_map::DefaultHasher;
use std::convert::TryFrom;
use std::hash::{Hash, Hasher};

use rand::seq::SliceRandom;
use rand::Rng;
use std::cell::RefCell;
use std::sync::Arc;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;

extern crate zerol;


use zerol::game::breakthrough::*;
use zerol::game::misere_breakthrough::*;
use zerol::game::weak_schur::*;
use zerol::game::hashcode_20::*;
use zerol::game::{Game, InteractiveGame, MoveCode, NoFeatures};
use zerol::policies::{
    flat::*, mcts::*, nmcs::*, nrpa::*, ppa::*, puct::*, Policy, PolicyBuilder, SinglePolicy, SinglePolicyBuilder,
};

fn game_solo<G: Game, P: SinglePolicyBuilder<G>>(pb: &P, settings: G::Settings) -> f32 {
    let player = G::players()[0];

    let mut p = pb.create(player);
    let mut b = G::new(player, settings);
    let actions = p.solve(&b);
    for a in actions {
        println!("{:?}", b);
        println!("Action: {:?}", a);
        b.play(&a);
     //   println!("Score: {}", b.score(player));
    }
    println!("{:?}", b);
    b.score(player)
}

fn game_duel<G: Game, P1: Policy<G>, P2: Policy<G>>(
    start: G::Player,
    p1: &mut P1,
    p2: &mut P2,
    settings: G::Settings,
) -> G::Player {
    let mut b = G::new(start, settings);
    const DBG: bool = false;

    while {
        let action = if b.turn() == G::players()[0] {
            p1.play(&b)
        } else {
            p2.play(&b)
        };
        if DBG {
            println!("{:?} => {:?}", b, action);
        }
        b.play(&action);
        b.winner() == None
    } {}
    if DBG {
        println!("{:?}", b);
    };
    b.winner().unwrap()
}

pub fn monte_carlo_match<G: Game, P1: PolicyBuilder<G> + Sync, P2: PolicyBuilder<G> + Sync>(
    n: usize,
    pb1: &P1,
    pb2: &P2,
    settings: &G::Settings,
) -> usize {
    let pb = ProgressBar::new(n as u64);
    pb.set_style(
        ProgressStyle::default_bar().template(
            "{spinner:.green} [{elapsed_precise}] [{bar:70.cyan/blue}] {msg} (ETA {eta})",
        ),
    );

    let c1 = Arc::new(RelaxedCounter::new(0));
    let c2 = Arc::new(RelaxedCounter::new(0));

    let count_victory: usize = (0..n)
        .into_par_iter()
        .map(|_| {
            let c1 = c1.clone();
            let c2 = c2.clone();

            let mut p1 = pb1.create(G::players()[0]);
            let mut p2 = pb2.create(G::players()[1]);

            let result = if game_duel(
                *G::players().choose(&mut rand::thread_rng()).unwrap(),
                &mut p1,
                &mut p2,
                settings.clone()
            ) == G::players()[0]
            {
                c1.inc();
                1
            } else {
                c2.inc();
                0
            };
            pb.inc(1);
            let v1 = c1.get();
            let v2 = c2.get();
            pb.set_message(&format!(
                "{}/{} ({:.2}%)",
                v1,
                v2,
                (v1 as f32) * 100. / ((v1 + v2) as f32)
            ));
            result
        })
        .sum();
    pb.finish();
    count_victory
}

struct GameDuelUI {}

impl GameDuelUI {
    fn render<IG: InteractiveGame, P1: Policy<IG::G> + 'static, P2: Policy<IG::G> + 'static>(
        start: <IG::G as Game>::Player,
        _p1: P1,
        p2: P2,
    ) -> impl cursive::view::View {
        //let r_p1 = RefCell::new(p1);
        let r_p2 = RefCell::new(p2);

        LinearLayout::vertical()
            .child(NamedView::new("game", IG::new(start)))
            .child(Button::new_raw("Next", move |s| {
                let state: &mut IG = &mut s.find_name("game").unwrap();

                //let mut p1 = r_p1.borrow_mut();
                let mut p2 = r_p2.borrow_mut();
                let p1_to_play = state.get().turn() == <IG::G as Game>::players()[0];

                if p1_to_play {
                    //p1.play(&state)
                    state.choose_move(Box::new(|action, state| state.get_mut().play(&action)))
                } else {
                    let action = p2.play(state.get());
                    state.get_mut().play(&action);
                };
            }))
    }
}

type G = Breakthrough;

fn main_ui() {
    let mut siv = Cursive::default();

    let pb1 = Random::default();
    let p1: RandomPolicy<G> = pb1.create(G::players()[0]);
    let pb2 = UCT::default();
    let p2: UCTPolicy<G> = pb2.create(G::players()[1]);

    siv.add_layer(
        Dialog::new()
            .title("Breakthrough")
            .content(GameDuelUI::render::<IBreakthrough, _, _>(
                G::players()[0],
                p1,
                p2,
            )),
    );

    siv.run();
}

pub struct BTCapture {}
impl MoveCode<Breakthrough> for BTCapture {
    fn code(game: &Breakthrough, action: &zerol::game::breakthrough::Move) -> usize {
        let mut s = DefaultHasher::new();
        action.hash(&mut s);
        let capture = {
            let (tx, ty) = action.target(&game.content);
            if game.content[tx][ty] == zerol::game::breakthrough::Cell::Empty {
                0
            } else {
                1
            }
        };
        capture.hash(&mut s);
        usize::try_from(s.finish()).unwrap()
    }
}

use zerol::misc::evaluator;
use std::marker::PhantomData;

use tensorflow::{Code, Graph, Session, SessionOptions, Status};
const MODEL_PATH: &str = "models/sample";

fn main() {

    let mut graph = Graph::new();
    let session = Session::from_saved_model(
        &SessionOptions::new(),
        &["serve"],
        &mut graph,
        MODEL_PATH,
    ).unwrap();

    let p1 = Random {};
    let p2 = PUCT {
        _g: PhantomData,
        C_PUCT: 0.4,
        evaluate: &(|board| evaluator(&session, &graph, board)),
    };

    println!(
        "Result: {}",
        monte_carlo_match::<Breakthrough, _, _>(100, &p1, &p2, &())
    );
    //main_ui();
/*
    let pb = NRPA::<_, NoFeatures>::default();
    let res = game_solo::<WeakSchurNumber, _>(&pb, ());
    println!("=> {} ", res);*/
/*
    let pb = NRPA::<_, NoFeatures>::default();
    let config = Hashcode20Settings::new_from_file("./data/b_read_on.txt");
    let res = game_solo::<Hashcode20, _>(&pb, &config);
    println!("=> {} ", res);*/
}