#![allow(non_snake_case)]
#![feature(trait_alias)]
#![feature(type_alias_impl_trait)]
#![feature(deadline_api)]
#![warn(missing_docs)]

//! General Game Playing Framework.
//!
//! This crate introduces common APIs for games and policies
//! to explore general game playing algorithms.
//!
//! Among classical tree search-based methods there are attempts
//! to re-implement [AlphaZero](policies/mcts/puct/index.html) and [MuZero](policies/mcts/muz/index.html). AlphaZero has been
//! tested on a toy example, but MuZero is not yet successful.
//!
//! # Binaries
//!
//! Several binaries can be used to test the project:
//!
//! * [`evaluate`](../evaluate/index.html): test two policies against each other.
//! * [`generate`](../generate/index.html): generate self-play games for AlphaZero and MuZero.
//! * [`generate`](../generate/index.html): generate self-play games for AlphaZero and MuZero.
//! * [`ui`](../ui/index.html): visualize PUCT-based policies (AlphaZero/MuZero) in a duel against PPA.
//! * [`perf`](../perf/index.html): test raw games generation performance.

///
/// Features for neural network-based policies.
///
pub mod deep;
///
/// General game traits and implementations.
///
pub mod game;
///
/// General policy traits and implementations.
///
pub mod policies;
///
/// General game and playout settings.
///
pub mod settings;
