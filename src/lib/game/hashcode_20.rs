use crate::game::{Singleplayer, Base, SingleplayerGameBuilder};

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeSet, BTreeMap};
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::iter::FromIterator;
use std::fs::File;
use std::io::{prelude::*, BufReader};

#[derive(Debug, Clone)]
pub struct Hashcode20Settings {
    B: usize,
    L: usize,
    D: usize,
    books: Vec<usize>,                              // book values
    libraries: Vec<(BTreeSet<usize>, usize, usize)>, // books, time to signup, number of books that can be scanned each day
}

impl Hashcode20Settings {
    pub fn new_from_file(filename: &str) -> Hashcode20Settings {
        let file = File::open(filename).expect(&(format!("Failed to read {}", filename)));
        let mut reader = BufReader::new(file);
        let mut line = String::new();

        reader.read_line(&mut line).unwrap();
        let settings: Vec<usize> = line.trim().split(' ').map(|word| word.parse::<usize>().unwrap()).collect();
        let B = settings[0];
        let L = settings[1];
        let D = settings[2];

        line.clear();
        reader.read_line(&mut line).unwrap();
        let books: Vec<usize> = line.trim().split(' ').map(|word| word.parse::<usize>().unwrap()).collect();
        let mut libraries = vec![];
        for _ in 0..L {
            line.clear();
            reader.read_line(&mut line).unwrap();
            let settings: Vec<usize> = line.trim().split(' ').map(|word| word.parse::<usize>().unwrap()).collect();
            //let N = settings[0];
            let T = settings[1];
            let M = settings[2];
            line.clear();
            reader.read_line(&mut line).unwrap();
            let l_books: Vec<usize> = line.trim().split(' ').map(|word| word.parse::<usize>().unwrap()).collect(); 

            let entry = (BTreeSet::from_iter(l_books), T, M);
            libraries.push(entry);
        };
        let res = Hashcode20Settings {
            B, L, D,
            books,
            libraries
        };
        println!("Read game configuration:\n{:?}\n", res);
        res
    }
}

#[derive(Clone)]
pub struct Hashcode20 {
    pending_sign_up: Option<(usize, usize)>, // Library / days left

    scanned_books: BTreeSet<usize>,
    unscanned_books: BTreeSet<usize>,

    signedup_libraries: BTreeSet<usize>,
    unsignedup_libraries: BTreeSet<usize>,

    n_books_scanned: BTreeMap<usize, usize>,

    pub day: usize,
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


#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Move {
    Signup(usize),
    Scan(usize,usize), //(book, library)
    Skip,
}

impl Hashcode20 {
    fn compute_possible_moves(&self) -> Vec<Move> {
        let mut result = vec![];

        if self.pending_sign_up.is_none() {
            for library in self.unsignedup_libraries.iter() {
                let (_,tts,_) = self.rules.libraries[*library];
                if tts < (self.rules.D - self.day) {
                    result.push(Move::Signup(*library))
                }
            }
        };

        for library in self.signedup_libraries.iter() {
            let (books,_,max_per_day) = &self.rules.libraries[*library];
            if self.n_books_scanned.get(&library).unwrap_or(&0) < max_per_day {
                for book in books {
                    if self.unscanned_books.contains(&book) {
                        result.push(Move::Scan(*book,*library))
                    }
                }
            }
        };

        if result.is_empty() {
            result.push(Move::Skip)
        };
        result
    }
}

impl SingleplayerGameBuilder<Hashcode20> for Hashcode20Settings {
    fn create(&self) -> Hashcode20 {
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
    fn score(&self) -> f32 {
        let score: usize = self.scanned_books.iter().map(|book| self.rules.books[*book]).sum();
        score as f32
    }
}

type PossibleMovesIterator<'a> = impl Iterator<Item=Move> + 'a;

impl Base for Hashcode20 {
    type Move = Move;
    type MoveIterator<'a> = PossibleMovesIterator<'a>;


    fn possible_moves<'a>(&'a self) -> Self::MoveIterator<'a> {
        self.compute_possible_moves().into_iter()
    }

    // assuming the move is valid.
    fn play(&mut self, m: &Self::Move) {
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
                }
            }
            Move::Scan(book, library) => {
                self.scanned_books.insert(*book);
                self.unscanned_books.remove(book);
                *self.n_books_scanned.entry(*library).or_insert(0) += 1;
            },
            Move::Signup(library) => {
                self.pending_sign_up = Some((*library, self.rules.libraries[*library].1))
            }
        };
    }

    fn hash(&self) -> usize {
        let mut hasher = DefaultHasher::new();
        self.scanned_books.hash(&mut hasher);
        self.pending_sign_up.hash(&mut hasher);
        self.signedup_libraries.hash(&mut hasher);
        self.n_books_scanned.hash(&mut hasher);
        self.day.hash(&mut hasher);
        hasher.finish() as usize
    }
}
