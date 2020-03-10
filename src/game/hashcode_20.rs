use super::Game;

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeSet, BTreeMap};
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::rc::Rc;
use std::iter::FromIterator;
use std::fs::File;
use std::io::{self, prelude::*, BufReader};

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
pub struct Hashcode20<'a> {
    pending_sign_up: Option<(usize, usize)>, // Library / days left

    scanned_books: BTreeSet<usize>,
    unscanned_books: BTreeSet<usize>,

    signedup_libraries: BTreeSet<usize>,
    unsignedup_libraries: BTreeSet<usize>,

    n_books_scanned: BTreeMap<usize, usize>,

    pub day: usize,
    rules: &'a Hashcode20Settings,

    possible_moves: Vec<Move>,
}

impl<'a> fmt::Debug for Hashcode20<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Day: {}", self.day)?;
        writeln!(f, "Scanned books: {:?}", self.scanned_books)?;
        writeln!(f, "NScannedPerLib: {:?}", self.n_books_scanned)?;
        writeln!(f, "Possible moves: {:?}", self.possible_moves)?;
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

impl<'a> Hashcode20<'a> {
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

impl<'a> Game for Hashcode20<'a> {
    type Player = ();
    type Move = Move;
    type GameHash = usize;
    type Settings = &'a Hashcode20Settings;

    fn new(_: Self::Player, rules: &'a Hashcode20Settings) -> Hashcode20<'a> {
        let scanned_books = BTreeSet::new();
        let signedup_libraries = BTreeSet::new();
        let unscanned_books = BTreeSet::from_iter(0..rules.B);
        let unsignedup_libraries = BTreeSet::from_iter(0..rules.L);
        let n_books_scanned = BTreeMap::new();

        let mut res = Hashcode20 {
            pending_sign_up: None,
            scanned_books,
            unscanned_books,
            signedup_libraries,
            unsignedup_libraries,
            n_books_scanned,
            day: 0,
            rules,
            possible_moves: vec![]
        };
        res.possible_moves = res.compute_possible_moves();
        res
    }

    fn players() -> Vec<()> {
        vec![()]
    }

    fn score(&self, _: Self::Player) -> f32 {
        let score: usize = self.scanned_books.iter().map(|book| self.rules.books[*book]).sum();
        score as f32
    }

    fn possible_moves(&self) -> &Vec<Self::Move> {
        &self.possible_moves
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
        self.possible_moves = self.compute_possible_moves()
    }

    fn turn(&self) -> Self::Player {
        ()
    }

    fn hash(&self) -> Self::GameHash {
        let mut hasher = DefaultHasher::new();
        self.scanned_books.hash(&mut hasher);
        self.pending_sign_up.hash(&mut hasher);
        self.signedup_libraries.hash(&mut hasher);
        self.n_books_scanned.hash(&mut hasher);
        self.day.hash(&mut hasher);
        hasher.finish() as usize
    }

    fn winner(&self) -> Option<Self::Player> {
        if self.day == self.rules.D {
            Some(())
        } else {
            None
        }
    }

    fn pass(&mut self) {
        panic!("")
    }
}
