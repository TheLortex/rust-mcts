use crate::game::*;

use async_trait::async_trait;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::File;
use std::hash::Hash;
use std::hash::Hasher;
use std::io::{prelude::*, BufReader};
use std::iter::FromIterator;

/// Hashcode 2020 game settings
#[derive(Debug, Clone)]
pub struct Hashcode20Settings {
    B: usize,
    L: usize,
    D: usize,
    books: Vec<usize>,                               // book values
    libraries: Vec<(BTreeSet<usize>, usize, usize)>, // books, time to signup, number of books that can be scanned each day
}

impl Hashcode20Settings {
    /// Create a game settings instance given its file description.
    pub fn new_from_file(filename: &str) -> Hashcode20Settings {
        let file = File::open(filename).unwrap_or_else(|_| panic!("Failed to read {}", filename));
        let mut reader = BufReader::new(file);
        let mut line = String::new();

        reader.read_line(&mut line).unwrap();
        let settings: Vec<usize> = line
            .trim()
            .split(' ')
            .map(|word| word.parse::<usize>().unwrap())
            .collect();
        let B = settings[0];
        let L = settings[1];
        let D = settings[2];

        line.clear();
        reader.read_line(&mut line).unwrap();
        let books: Vec<usize> = line
            .trim()
            .split(' ')
            .map(|word| word.parse::<usize>().unwrap())
            .collect();
        let mut libraries = vec![];
        for _ in 0..L {
            line.clear();
            reader.read_line(&mut line).unwrap();
            let settings: Vec<usize> = line
                .trim()
                .split(' ')
                .map(|word| word.parse::<usize>().unwrap())
                .collect();
            //let N = settings[0];
            let T = settings[1];
            let M = settings[2];
            line.clear();
            reader.read_line(&mut line).unwrap();
            let l_books: Vec<usize> = line
                .trim()
                .split(' ')
                .map(|word| word.parse::<usize>().unwrap())
                .collect();

            let entry = (BTreeSet::from_iter(l_books), T, M);
            libraries.push(entry);
        }
        let res = Hashcode20Settings {
            B,
            L,
            D,
            books,
            libraries,
        };
        println!("Read game configuration:\n{:?}\n", res);
        res
    }
}

/// Hashcode 2020 Game
#[derive(Clone)]
pub struct Hashcode20 {
    pending_sign_up: Option<(usize, usize)>, // Library / days left

    scanned_books: BTreeSet<usize>,
    unscanned_books: BTreeSet<usize>,

    signedup_libraries: BTreeSet<usize>,
    unsignedup_libraries: BTreeSet<usize>,

    n_books_scanned: BTreeMap<usize, usize>,

    day: usize,
    rules: Hashcode20Settings,
}

impl fmt::Debug for Hashcode20 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Day: {}", self.day)?;
        writeln!(f, "Scanned books: {:?}", self.scanned_books)?;
        writeln!(f, "NScannedPerLib: {:?}", self.n_books_scanned)?;
        writeln!(f, "Signup pending: {:?}", self.pending_sign_up)?;
        writeln!(f, "Signed up: {:?}", self.signedup_libraries)
    }
}

/// Possible actions
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Move {
    /// Signup a library.
    Signup(usize),
    /// Scan a book in a library.
    Scan(usize, usize), //(book, library)
    /// Finish the day.
    Skip,
}

impl Hashcode20 {
    fn compute_possible_moves(&self) -> Vec<Move> {
        let mut result = vec![];

        if self.pending_sign_up.is_none() {
            for library in self.unsignedup_libraries.iter() {
                let (_, tts, _) = self.rules.libraries[*library];
                if tts < (self.rules.D - self.day) {
                    result.push(Move::Signup(*library))
                }
            }
        };

        for library in self.signedup_libraries.iter() {
            let (books, _, max_per_day) = &self.rules.libraries[*library];
            if self.n_books_scanned.get(&library).unwrap_or(&0) < max_per_day {
                for book in books {
                    if self.unscanned_books.contains(&book) {
                        result.push(Move::Scan(*book, *library))
                    }
                }
            }
        }

        if result.is_empty() {
            result.push(Move::Skip)
        };
        result
    }
}

#[async_trait]
impl SingleplayerGameBuilder<Hashcode20> for Hashcode20Settings {
    async fn create(&self) -> Hashcode20 {
        let scanned_books = BTreeSet::new();
        let signedup_libraries = BTreeSet::new();
        let unscanned_books = BTreeSet::from_iter(0..self.B);
        let unsignedup_libraries = BTreeSet::from_iter(0..self.L);
        let n_books_scanned = BTreeMap::new();

        Hashcode20 {
            pending_sign_up: None,
            scanned_books,
            unscanned_books,
            signedup_libraries,
            unsignedup_libraries,
            n_books_scanned,
            day: 0,
            rules: self.clone(),
        }
    }
}

impl Singleplayer for Hashcode20 {
    /*    fn score(&self) -> f32 {
        let score: usize = self.scanned_books.iter().map(|book| self.rules.books[*book]).sum();
        score as f32
    }*/
}

impl Base for Hashcode20 {
    type Move = Move;

    fn possible_moves(&self) -> Vec<Move> {
        self.compute_possible_moves()
    }
}

impl Hash for Hashcode20 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.scanned_books.hash(state);
        self.pending_sign_up.hash(state);
        self.signedup_libraries.hash(state);
        self.n_books_scanned.hash(state);
        self.day.hash(state);
    }
}

#[async_trait]
impl Playable for Hashcode20 {
    // assuming the move is valid.
    async fn play(&mut self, m: &<Self as Base>::Move) -> f32 {
        match m {
            Move::Skip => {
                self.day += 1;
                self.n_books_scanned.clear();
                if let Some((library, time_left)) = self.pending_sign_up {
                    if time_left == 1 {
                        self.pending_sign_up = None;
                        self.signedup_libraries.insert(library);
                        self.unsignedup_libraries.remove(&library);
                    } else {
                        self.pending_sign_up = Some((library, time_left - 1))
                    }
                };
                0.
            }
            Move::Scan(book, library) => {
                self.scanned_books.insert(*book);
                self.unscanned_books.remove(book);
                *self.n_books_scanned.entry(*library).or_insert(0) += 1;
                self.rules.books[*book] as f32
            }
            Move::Signup(library) => {
                self.pending_sign_up = Some((*library, self.rules.libraries[*library].1));
                0.
            }
        }
    }
}
