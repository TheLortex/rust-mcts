use crate::deep::self_play::GameHistoryEntry;
use crate::deep::tf;
use crate::game;

use nix::sys::stat;
use nix::unistd::mkfifo;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;

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

use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::mpsc::channel;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;
use tensorflow::{Graph, Session};

/// Watch a path for changes and reload the model when content has been modified.
pub fn watch_model(tf: Arc<(AtomicBool, RwLock<(Graph, Session)>)>, path: String) {
    thread::spawn(move || {
        // Create a channel to receive the events.
        let (tx, rx) = channel();

        let p = path.clone();

        let mut watcher: RecommendedWatcher = Watcher::new(tx, Duration::from_millis(500)).unwrap();

        log::info!("Watching path {}", path);
        watcher.watch(path, RecursiveMode::NonRecursive).unwrap();

        loop {
            match rx.recv() {
                Ok(_) => {
                    let (global_lock, tf_network) = tf.as_ref();
                    log::info!("Updating model.. {}", p);
                    global_lock.store(true, Ordering::Relaxed);
                    let mut tf_network = tf_network.write().unwrap();
                    global_lock.store(false, Ordering::Relaxed);
                    let (ref _graph, ref mut session_m) = *tf_network;
                    session_m.close().expect("Unable to close the session.");

                    *tf_network = tf::load_model(&p);
                    log::info!("Model successfully updated!");
                }
                Err(e) => println!("watch error: {:?}", e),
            }
        }
    });
}
