# RGGPF: Rust General Game Playing Framework

This is a project for the IASD course on Monte-Carlo Search of M. Cazenave (https://www.lamsade.dauphine.fr/~cazenave/MonteCarloSearch.html). It introduces a general framework for game playing and agent policies for these games. Several games and policies have been implemented.

API documentation is [available here](https://www.lortex.org/rust-mcts/ggpf/).

## Setup

This project uses `rust` (nightly channel) and `python` with `tensorflow`. 

- Install `rustup` and launch `rustup default nightly` to enable the nightly compiler.
- Install `tensorflow` to enable PUCT/AlphaZero/MuZero policies.

## Usage

- Use `cargo run --bin <binary>` to execute binaries. Available binaries are:
  -  `ggpf_evaluate`: evaluate two policies on breakthrough
  -  `ggpf_ui`: interactive interface to inspect alphazero
  -  `ggpf_alphazero_generate`: self-play game generators for alphazero
  -  `ggpf_muzero_generate`: self-play game generators for muzero
- To perform traning, you need to launch both python and rust binaries:
  - `python alphazerol.py` to execute the training loop and generate the network model.
  - `cargo run --bin ggpf_alphazero_generate` to launch the self-play generator.

## Games

### Multi-player
- Breakthrough
- Misère Breakthrough

### Single-player
- Hashcode 2020
- Weak schur number
- OpenAI Gym

## Policies

- Random
- Flat UCB (Upper Confidence Bound)
- UCT (Upper Confidence Tree)
- RAVE (Rapid Action Value Estimation)
- NMCS (Nested Monte Carlo Search)
- NRPA (Nested Rollout Policy Adaptation)
- PPA  (Playout Policy Adaptation)
- PUCT (AlphaZero MCTS)
- MUZ  (MuZero MCTS)

## AlphaZero

It's possible to reproduce Deepmind's AlphaZero results on toy games.

## MuZero

