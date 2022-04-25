#![allow(clippy::many_single_char_names,clippy::explicit_counter_loop)]

use crate::btree::interface::{List};
use crate::btree::arraystack::{ArrayStack};

#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BlockStore<T: Clone> {
    blocks: ArrayStack<T>,
    free: ArrayStack<usize>,
}

impl<T: Clone> BlockStore<T> {
    pub fn new() -> Self {
        Self {
            blocks: ArrayStack::new(),
            free: ArrayStack::new(),
        }
    }
    pub fn new_block(&mut self, block: T) -> usize {
        if self.free.size() > 0 {
            self.free.remove(self.free.size() - 1).unwrap()
        } else {
            let id = self.blocks.size();
            self.blocks.add(id, block);
            id
        }
    }
    pub fn free_block(&mut self, id: usize) {
        self.blocks.take(id);
        self.free.add(self.free.size(), id);
    }
    pub fn read_block(&self, id: usize) -> Option<T> {
        self.blocks.get(id)
    }
    pub fn write_block(&mut self, id: usize, block: T) {
        self.blocks.set(id, block);
    }
}
