mod btree;

use rand::{thread_rng, Rng};
use btree::BTree;
use btree::interface::{SSet};

pub fn main() {
  let mut rng = thread_rng();
  let n = 200;
  let mut btree = BTree::<i32>::new(5);

  for i in 0..n {
    let x = i;
    btree.add(x);
  }

  for _ in 0..n {
      let x = rng.gen_range(0, 200);
      let y = btree.find(&x);
      assert_eq!(Some(x), y);
  }

  let x = rng.gen_range(0, 200);
  let y = btree.remove(&x);
  assert_eq!(Some(x), y);
}
