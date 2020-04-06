# RGGPF: Rust General Game Playing Framework

This is a project for the IASD course on Monte-Carlo Search of M. Cazenave (https://www.lamsade.dauphine.fr/~cazenave/MonteCarloSearch.html). It introduces a general framework for game playing and agent policies for these games. Several games and policies have been implemented.

API documentation is [available here](https://www.lortex.org/rust-mcts/ggpf/).

## Setup

This project uses `rust` (nightly channel) and `python` with `tensorflow`. 

- Install `rustup` and launch `rustup default nightly` to enable the nightly compiler.
- Install `tensorflow` to enable PUCT/AlphaZero/MuZero policies.
- More generally, use `pip install -r requirements.txt` to install python dependencies (`tensorflow` is excluded from the list as either `tensorflow` or `tensorflow-gpu` works).

## Usage

*Cargo* is the Rust project manager.
Use `cargo run --release --bin <binary>` to execute binaries. Available binaries are:
  -  `evaluate`: evaluate two policies on breakthrough
  -  `ui`: interactive interface to inspect alphazero
  -  `generate`: self-play game generators
  -  `gym_server`: decoupled game executor for openai gym
  -  `perf`: benchmarking tests

### Configuration files

`evaluate`, `generate` and `ui` all use a configuration file located in the `config/` path. It is selected
by the `--config` option. 

### Training

To perform training, you need to launch both python and rust binaries:
  - `python training.py --config breakthrough --method <alpha|mu>` to execute the training loop and generate the network model.
  - `cargo run --release --bin generate -- -c breakthrough -m <alpha|mu>` to launch the self-play generator.

There shouldn't be any errors and the number of generated games should increase. 
Debugging information can be activated using `export RUST_DEBUG=info`. Models are saved in `data/<name>/model`. Tensorboard logs are saved in `data/<name>/logs` and training data is saved in `data/<name>/training_data`.

### Testing

Two policies can be tested using `evaluate`: 

`cargo run --release --bin evaluate -- -c breakthrough --policy <policy> --against <policy>`

### Visualizing

To launch the UI and visualize Alpha/Mu tree search live, use `ui`:

`cargo run --release --bin ui -- -c breakthrough --method <alpha|mu>`

(to avoid tensorflow logs, use `export TF_CPP_MIN_LOG_LEVEL=2`)



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

I haven't been able to have satisfactory results on MuZero, even on toy games, but the implementation is here.
