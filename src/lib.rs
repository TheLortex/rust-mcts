#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(non_snake_case)]
#![feature(trait_alias)]

use numpy::{PyArray1, PyArray2};
//use ndarray::arr2;
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::wrap_pyfunction;

use std::collections::HashMap;
use std::iter::*;

use rand::seq::SliceRandom;
use rand::Rng;

mod game;
mod policies;

use game::breakthrough::{Breakthrough, Color, Move, K};
use game::Game;
use policies::puct::PUCTPolicy;
use policies::Policy;
use rayon::prelude::*;

fn self_play_match<F: Fn(&Breakthrough) -> (HashMap<Move, f32>, f32)>(
    evaluate: &F,
) -> (Color, Vec<(Breakthrough, HashMap<Move, f32>)>) {
    let mut game: Breakthrough = Breakthrough::new(Color::Black, ());
    let mut p1 = PUCTPolicy {
        C_PUCT: 4.,
        color: Color::Black,
        tree: HashMap::new(),
        evaluate,
    };
    let mut p2 = PUCTPolicy {
        C_PUCT: 4.,
        color: Color::White,
        tree: HashMap::new(),
        evaluate,
    };

    let mut history = vec![];

    while { game.winner().is_none() } {
        let policy = if game.turn() == Color::Black {
            &mut p1
        } else {
            &mut p2
        };

        let action = policy.play(&game);
        let game_node = policy.tree.get(&game.hash()).unwrap();
        let monte_carlo_distribution = HashMap::from_iter(
            game_node
                .moves
                .iter()
                .map(|(k, v)| (*k, v.N_a / game_node.count)),
        );
        history.push((game.clone(), monte_carlo_distribution));

        game.play(&action);
    }

    (game.winner().unwrap(), history)
}

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::sync::{Arc, Mutex};

fn evaluator(py: Python, evaluate: &PyObject, board: &Breakthrough) -> (HashMap<Move, f32>, f32) {
    let input: &PyArray2<f32> = PyArray2::new(py, [1, 2 * K * K + 1], false);
    for (j, item) in board.serialize().iter().enumerate() {
        input.as_array_mut()[[0, j]] = *item;
    }

    let arg: [PyObject; 1] = [input.into_py(py)];
    let result = match evaluate.call1(py, PyTuple::new(py, &arg)) {
        Ok(result) => result,
        Err(err) => {
            err.print(py);
            panic!("Python failed.")
        }
    }; // evaluator must return a policy-value tuple.
    let result: &PyTuple = result.cast_as(py).unwrap();
    let py_policy: &PyArray2<f32> = result.get_item(0).downcast_ref().unwrap();
    let py_value: &PyFloat = result.get_item(1).downcast_ref().unwrap();

    let policy = HashMap::from_iter(
        board
            .possible_moves()
            .into_iter()
            .map(|m| (*m, py_policy.as_array()[[0, m.serialize()]])),
    );
    let value = py_value.value();
    (policy, value as f32)
}

#[pymodule]
pub fn libzerol(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "puct")]
    pub fn puct(py: Python, game_batch: usize, evaluate: PyObject) -> &PyTuple {
        let evalutate_ref = &evaluate;//Arc::new(Mutex::new(evaluate));

        // TODO: take more samples
        let bar = ProgressBar::new(game_batch as u64);
        let histories = py.allow_threads(|| {
            (0..game_batch).into_par_iter().map(|_| {
                let gil = Python::acquire_gil();
                let py  = gil.python();
                let res = self_play_match(&(|board| evaluator(py, evalutate_ref, board)));
                bar.inc(1);
                res
            })
        });
        bar.finish();
        let samples: Vec<(Color, (Breakthrough, HashMap<Move, f32>))> = histories
            .map(|(winner, hist)| {
                let action = hist.choose(&mut rand::thread_rng()).cloned();
                (winner, action.unwrap())
            })
            .collect();
        let input: &PyArray2<f32> = PyArray2::new(py, [game_batch, 2 * K * K + 1], false);
        let policy: &PyArray2<f32> = PyArray2::zeros(py, [game_batch, 3 * K * K], false);
        let value: &PyArray1<f32> = PyArray1::new(py, [game_batch], false);
        for (i, (w, (b, a))) in samples.iter().enumerate() {
            for (j, item) in b.serialize().iter().enumerate() {
                input.as_array_mut()[[i, j]] = *item;
            }
            value.as_array_mut()[i] = if b.turn() == *w {
                // TODO; this should be in game ?
                1.0
            } else {
                0.0
            };
            for (action, proba) in a {
                policy.as_array_mut()[[i, action.serialize()]] = *proba;
            }
        }
        let res: [PyObject; 3] = [input.into_py(py), policy.into_py(py), value.into_py(py)];
        PyTuple::new(py, &res)
    }
    Ok(())
}
