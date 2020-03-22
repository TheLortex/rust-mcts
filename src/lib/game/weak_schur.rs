use crate::game::{SingleplayerGame,SingleplayerGameBuilder,BaseGame};

use std::fmt;
use std::hash::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::cmp::Ordering;

const K: usize = 9;
const WS_RULE: bool = true;

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
}

pub struct WeakSchurNumberBuilder {

}

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

impl SingleplayerGame for WeakSchurNumber {
    fn score(&self) -> f32 {
        self.partitions
            .iter()
            .map(|p| Self::best_sequence(&p))
            .max()
            .unwrap() as f32
    }

}


type PossibleMovesIterator<'a> = impl Iterator<Item=usize> + 'a;

impl BaseGame for WeakSchurNumber {
    type Move = usize;
    type MoveIterator<'a> = PossibleMovesIterator<'a>;

    fn play(&mut self, m: &Self::Move) {
        self.last_value += 1;
        self.partitions[*m].push(self.last_value);
    }

    fn hash(&self) -> usize {
        let mut hasher = DefaultHasher::new();
        self.partitions.hash(&mut hasher);
        hasher.finish() as usize
    }

    fn possible_moves<'a>(&'a self) -> Self::MoveIterator<'a> {

        let valid_moves = self.partitions
            .iter()
            .enumerate()
            .filter(move |(_, partition)| WeakSchurNumber::is_valid(self.last_value + 1, partition));

        if WS_RULE {
            panic!("Not implemented.");
            if let Some((idx, _)) = valid_moves
                .filter(|(_, partition)| partition.last() == Some(&self.last_value))
                .next()
            {
                panic!("Not implemented.");
                //return vec![idx];
            }
        }

        valid_moves.map(|(i,_)| i)
    }
}
