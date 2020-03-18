

use nix::unistd::mkfifo;
use nix::sys::stat;

use std::fs::{File, OpenOptions};
use std::io::Write;

use super::game;

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

    pub fn append<G: game::Feature>(&mut self, history: &[G], policy: &HashMap<G::Move, f32>, value: &f32) {
        let mut vec_board: Vec<f32> = Vec::new();
        let current_turn = history.last().unwrap().turn();
        for board in &history[history.len()-2..history.len()] {
            vec_board.extend(board.state_to_feature(current_turn).into_raw_vec())
        }
        
        let vec_policy = G::moves_to_feature(policy).into_raw_vec();

        let toser = (vec_board, vec_policy, value);
        let ser = serde_pickle::to_vec(&toser, true).unwrap();
        self.f.write(&ser.len().to_be_bytes()).expect(":c");
        self.f.write_all(&ser).expect("Could not write file..");
    }
}
