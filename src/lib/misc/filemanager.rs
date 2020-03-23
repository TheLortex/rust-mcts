use crate::game;
use crate::game::{meta::with_history::WithHistory, Feature};

use typenum::Unsigned;
use nix::unistd::mkfifo;
use nix::sys::stat;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::collections::HashMap;

pub struct FileManager {
    f: File,
}

impl FileManager {
    pub fn new(path: &str) -> Self {
        match mkfifo(path, stat::Mode::S_IRWXU) {
            Ok(_) => println!("Created FIFO {}.", path),
            Err(_) => println!("FIFO already exists."),
        };

        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .expect("Unable to open file.");
        FileManager { f }
    }

    pub fn append<G: game::Feature + Clone,N: Unsigned>(&mut self, board: &WithHistory<G,N>, policy: &HashMap<G::Move, f32>, value: f32) {
        let current_turn = board.state.turn();
        let vec_board = board.state_to_feature(current_turn).into_raw_vec();
        let vec_policy = G::moves_to_feature(policy).into_raw_vec();

        let toser = (vec_board, vec_policy, value);
        let ser = serde_pickle::to_vec(&toser, true).unwrap();
        self.f.write_all(&ser.len().to_be_bytes()).expect(":c");
        self.f.write_all(&ser).expect("Could not write file..");
    }
}
