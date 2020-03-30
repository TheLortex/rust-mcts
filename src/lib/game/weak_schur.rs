use crate::game::{Singleplayer,SingleplayerGameBuilder,Base,Playable};

use std::fmt;
use std::hash::*;
use std::hash::Hasher;
use std::cmp::Ordering;

const K: usize = 9;
const WS_RULE: bool = true;

/// Weak schur number game.
#[derive(Clone, Eq)]
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
    fn is_valid(last_value: usize, values: &[usize]) -> bool {
        let mut begin = 0;
        let mut end = values.len() as isize - 1;
        let target = last_value + 1;

        while begin < end {
            let sum = values[begin as usize] + values[end as usize];
            match sum.cmp(&target) {
                Ordering::Equal => return false,
                Ordering::Less => begin += 1,
                Ordering::Greater => end -= 1,
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

    fn score(&self) -> f32 {
        self.partitions
            .iter()
            .map(|p| Self::best_sequence(&p))
            .max()
            .unwrap() as f32
    }
}

impl Singleplayer for WeakSchurNumber {}

/// Weak schur number game builder
pub struct WeakSchurNumberBuilder {}

impl SingleplayerGameBuilder<WeakSchurNumber> for WeakSchurNumberBuilder {
    fn create(&self) -> WeakSchurNumber {
        let partitions = Default::default();
        let last_value = 0;

        WeakSchurNumber {
            partitions,
            last_value,
        }
    }
}

impl Base for WeakSchurNumber {
    type Move = usize;

    fn possible_moves(&self) -> Vec<Self::Move> {

        let valid_moves = self.partitions
            .iter()
            .enumerate()
            .filter(move |(_, partition)| WeakSchurNumber::is_valid(self.last_value + 1, partition));

        if WS_RULE {
            if let Some((idx, _)) = valid_moves.clone()
                .find(|(_, partition)| partition.last() == Some(&self.last_value))
            {
                return vec![idx];
            }
        }

        valid_moves.map(|(i,_)| i).collect()
    }
}


impl Hash for WeakSchurNumber {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.partitions.hash(state)
    }
}

impl PartialEq for WeakSchurNumber {
    fn eq(&self, other: &WeakSchurNumber) -> bool {
        self.partitions.eq(&other.partitions)
    }
}

impl Playable for WeakSchurNumber {
    fn play(&mut self, m: &Self::Move) -> f32 {
        self.last_value += 1;
        self.partitions[*m].push(self.last_value);
        if self.is_finished() {
            self.score()
        } else {
            0.
        }
    }
}