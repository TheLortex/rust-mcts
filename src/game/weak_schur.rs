use std::fmt;
use std::hash::*;

use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

use std::cmp::Ordering;

use super::Game;

const K: usize = 9;

#[derive(Clone, PartialEq, Eq)]
pub struct WeakSchurNumber {
    partitions: [Vec<usize>; K],
    last_value: usize,
}

impl fmt::Debug for WeakSchurNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Partitions")?;
        for i in 0..K {
            writeln!(f, "{:?}", self.partitions[i])?;
        }
        write!(f, "")
    }
}

impl WeakSchurNumber {
    /* assumes values is in increasing order. */
    fn is_valid(&self, values: &[usize]) -> bool {
        let mut begin = 0;
        let mut end = values.len() as isize - 1;
        let target = self.last_value + 1;

        while begin < end {
            let sum = values[begin as usize] + values[end as usize];
            match sum.cmp(&target) {
                Ordering::Equal => return false,
                Ordering::Less =>  begin += 1,
                Ordering::Greater =>  end -= 1,
            };
        }
        true
    }

    fn best_sequence(values: &[usize]) -> usize {
        let mut best_length = 0;
        let mut cur_length = 0;

        for i in 1..values.len() {
            if values[i] == values[i - 1] + 1 {
                cur_length += 1;
                if cur_length + 1 > best_length {
                    best_length = cur_length + 1
                };
            } else {
                cur_length = 0
            }
        }
        best_length
    }
}

impl Game for WeakSchurNumber {
    type Player = ();
    type Move = usize;
    type GameHash = usize;

    fn new(_: Self::Player) -> WeakSchurNumber {
        WeakSchurNumber {
            partitions: Default::default(),
            last_value: 0,
        }
    }

    fn players() -> Vec<()> {
        vec![()]
    }

    fn score(&self, _: Self::Player) -> f64 {
        self.partitions
            .iter()
            .map(|p| Self::best_sequence(&p))
            .max()
            .unwrap() as f64
    }

    fn play(&mut self, m: &Self::Move) {
        self.last_value += 1;
        self.partitions[*m].push(self.last_value);
    }

    fn turn(&self) -> Self::Player {}

    fn hash(&self) -> usize {
        let mut hasher = DefaultHasher::new();
        self.partitions.hash(&mut hasher);
        hasher.finish() as usize
    }

    fn possible_moves(&self) -> Vec<Self::Move> {
        self.partitions
            .iter()
            .enumerate()
            .filter(|(_, partition)| self.is_valid(partition))
            .map(|(i, _)| i)
            .collect()
    }

    fn winner(&self) -> Option<Self::Player> {
        if !self.possible_moves().is_empty() {
            None
        } else {
            Some(())
        }
    }

    fn pass(&mut self) {}
}
