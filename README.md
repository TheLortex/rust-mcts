# RGGPF: Rust General Game Playing Framework

This is a project for the IASD course on Monte-Carlo Search of M. Cazenave (https://www.lamsade.dauphine.fr/~cazenave/MonteCarloSearch.html). It introduces a general framework for game playing and agent policies for these games. Several games and policies have been implemented.

API documentation is [available here](https://www.lortex.org/rust-mcts/ggpf/).

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

