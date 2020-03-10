use std::fmt;
use std::hash::*;

use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

use std::cmp::Ordering;

use super::{SingleplayerGame,SingleplayerGameBuilder,BaseGame};

const K: usize = 9;
const WS_RULE: bool = true;

#[derive(Clone, PartialEq, Eq)]
pub struct WeakSchurNumber {
    partitions: [Vec<usize>; K],
    last_value: usize,
    possible_moves: Vec<usize>,
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

    fn compute_possible_moves(last_value: usize, partitions: &[Vec<usize>; K]) -> Vec<usize> {
        let valid_moves: Vec<(usize, &Vec<usize>)> = partitions
            .iter()
            .enumerate()
            .filter(|(_, partition)| WeakSchurNumber::is_valid(last_value + 1, partition))
            .collect();

        if WS_RULE {
            if let Some((idx, _)) = valid_moves
                .iter()
                .filter(|(_, partition)| partition.last() == Some(&last_value))
                .collect::<Vec<&(usize, &Vec<usize>)>>()
                .first()
            {
                return vec![*idx];
            }
        }
        valid_moves.iter().map(|(i, _)| *i).collect()
    }
}

pub struct WeakSchurNumberBuilder {

}

impl SingleplayerGameBuilder<WeakSchurNumber> for WeakSchurNumberBuilder {
    fn create(&self) -> WeakSchurNumber {
        let partitions = Default::default();
        let last_value = 0;
        let possible_moves = WeakSchurNumber::compute_possible_moves(last_value, &partitions);

        WeakSchurNumber {
            partitions,
            last_value,
            possible_moves,
        }
    }
}

impl SingleplayerGame for WeakSchurNumber {
    fn score(&self) -> f32 {
        self.partitions
            .iter()
            .map(|p| Self::best_sequence(&p))
            .max()
            .unwrap() as f32
    }

}

impl BaseGame for WeakSchurNumber {
    type Move = usize;

    fn play(&mut self, m: &Self::Move) {
        self.last_value += 1;
        self.partitions[*m].push(self.last_value);
        self.possible_moves =
            WeakSchurNumber::compute_possible_moves(self.last_value, &self.partitions);
    }

    fn hash(&self) -> usize {
        let mut hasher = DefaultHasher::new();
        self.partitions.hash(&mut hasher);
        hasher.finish() as usize
    }

    fn possible_moves(&self) -> &[Self::Move] {
        &self.possible_moves
    }
}
