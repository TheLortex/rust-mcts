#![allow(non_snake_case)]
#![feature(trait_alias)]
#![feature(type_alias_impl_trait)]
#![feature(deadline_api)]
#![warn(missing_docs)]

//! General Game Playing Framework.
//! 
//! This crate introduces common APIs for games and policies 
//! to explore general game playing algorithms.

/**
 *  General game traits and implementations.
 */
pub mod game;
/**
 *  General policy traits and implementations.
 */
pub mod policies; 
/**
 *  Miscellanous features.
 */
pub mod misc;
/**
 *  Asynchronous execution.
 */
pub mod r#async;
/**
 *  General game and playout settings.
 */
pub mod settings;