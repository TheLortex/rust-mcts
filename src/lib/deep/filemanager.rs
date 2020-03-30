use crate::game;
use crate::deep::tf;
use crate::deep::self_play::GameHistoryEntry;

use nix::unistd::mkfifo;
use nix::sys::stat;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::collections::HashMap;

/// File manager.
pub struct FileManager {
    f: File,
}

impl FileManager {
    /// Instanciate file manager on given file.
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

    /// Add a new game history entry to file.
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


use notify::event::*;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tensorflow::{Session, Graph};
use std::sync::RwLock;

/// Watch a path for changes and reload the model when content has been modified.
pub fn watch_model(global_lock: Arc<AtomicBool>, tf_network: Arc<RwLock<(Graph, Session)>>, path: String) {
    let p = path.clone();
    let mut watcher: RecommendedWatcher = Watcher::new_immediate(move |x: Result<Event, _>| {
        if let Ok(x) = x {
            if x.kind == EventKind::Access(AccessKind::Close(AccessMode::Write)) {
                log::debug!("Updating model..");
                global_lock.store(true, Ordering::Relaxed);
                let mut tf_network = tf_network.write().unwrap();
                global_lock.store(false, Ordering::Relaxed);
                let (ref _graph, ref mut session_m) = *tf_network;
                session_m.close().expect("Unable to close the session.");

                *tf_network = tf::load_model(&p);
                log::debug!("Model successfully updated!");
            }
        }
    }).unwrap();

    watcher.watch(&path, RecursiveMode::NonRecursive).unwrap();
}