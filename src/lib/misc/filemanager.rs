use crate::game;
use crate::r#async::GameHistoryEntry;

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

    pub fn append<G: game::Feature>(&mut self, game: GameHistoryEntry<G>) {
        let mut result = HashMap::new();

        result.insert("turn", game.turn);
        result.insert("state", game.state.into_raw_vec());
        result.insert("policy", game.policy.into_raw_vec());
        result.insert("value", game.value.into_raw_vec());
        result.insert("action", game.action.into_raw_vec());
        result.insert("reward", game.reward.into_raw_vec());
        

        let ser = serde_pickle::to_vec(&result, true).unwrap();
        self.f.write_all(&ser.len().to_be_bytes()).expect(":c");
        self.f.write_all(&ser).expect("Could not write file..");
    }
}
