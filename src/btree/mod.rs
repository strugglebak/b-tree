#![allow(clippy::many_single_char_names,clippy::explicit_counter_loop)]

pub mod arraystack;
pub mod blockstore;
pub mod interface;

use blockstore::{BlockStore};
use interface::{SortedSet};

#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
struct Node<T: Clone + PartialOrd> {
    id: usize,
    keys: Box<[Option<T>]>,
    children: Box<[i32]>,
}

#[allow(non_snake_case)]
#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
// max_children_amount: 每个 node 的 children 的最大数量
// B: max_children_amount / 2
// n: 在 BTree 中存放有多少个节点
// root_index: 根节点索引
// bs: 存放 node 的store
pub struct BTree<T: Clone + PartialOrd> {
    max_children_amount: usize,
    B: usize,
    n: usize,
    root_index: usize,
    bs: BlockStore<Node<T>>,
}

impl<T: Clone + PartialOrd> Node<T> {
    fn new(t: &mut BTree<T>) -> Self {
        let b = t.max_children_amount;
        let mut obj = Self {
            id: 0,
            keys: vec![None; b].into_boxed_slice(),
            children: vec![-1i32; b + 1].into_boxed_slice(),
        };
        // 放入store并拿到 id
        // 后续可以根据这个 id 得到其对应的 Node
        // 每次调用这个 new，id 会自动增长
        obj.id = t.bs.new_block(obj.clone());
        obj
    }
    fn is_leaf(&self) -> bool {
        self.children[0] < 0
    }
    fn is_full(&self) -> bool {
        self.keys[self.keys.len() - 1].is_some()
    }
    fn size(&self) -> usize {
        let mut lo = 0;
        let mut hi = self.keys.len();
        while hi != lo {
            let m = (hi + lo) / 2;
            if self.keys[m].is_none() {
                hi = m;
            } else {
                lo = m + 1;
            }
        }
        lo
    }
    fn add(&mut self, x: T, ci: i32) -> bool {
        // 从 root node 遍历到 leaf node
        let i = BTree::<T>::find_it(&self.keys, &x);
        if i < 0 {
            return false;
        }

        // 强转
        let i = i as usize;

        // 对 key 进行处理
        let n = self.keys.len();
        if i >= n - 1 {
            // key 在最后一位插入
            self.keys[n - 1] = Some(x);
        } else {
            // key 在中间插入

            // i 到 第 n-1 的数先挨个向右移一位，腾出一个空间出来
            self.keys[i..(n - 1)].rotate_right(1);
            // 插入
            let end = self.keys[i].replace(x);
            self.keys[n - 1] = end;
        }

        // 对 children 进行处理
        // children 是比 key 要多一个的，所以这里是 i+1 的判断
        let n = self.children.len();
        if i + 1 >= n - 1 {
            // children 在最后一位插入
            self.children[n - 1] = ci;
        } else {
            // children 在中间插入
            self.children[(i + 1)..(n - 1)].rotate_right(1);

            // 插入
            self.children[n - 1] = ci;
            self.children.swap(i + 1, n - 1);
        }
        true
    }
    fn remove(&mut self, i: usize) -> Option<T> {
        let n = self.keys.len();
        // 把 keys[i] 变成 None
        let y = self.keys.get_mut(i)?.take();
        // keys 整体左移一位
        self.keys[i..n].rotate_left(1);
        y
    }
    // 对该 Node 砍掉一半的信息
    fn split(&mut self, t: &mut BTree<T>) -> Option<Node<T>> {
        // 先生成一个新的 Node 节点
        let mut w = Self::new(t);

        // 取中间的 key
        let j = self.keys.len() / 2;

        // 处理 key
        // 复制中间的 key 的后面的 key 给这个新 Node 节点
        for (i, key) in self.keys[j..].iter_mut().enumerate() {
            w.keys[i] = key.take();
        }

        // 处理 children
        // 复制中间的 children 的后面的 children 给这个新 Node 节点
        for (i, chd) in self.children[(j + 1)..].iter_mut().enumerate() {
            w.children[i] = *chd;
            *chd = -1;
        }

        // 将数据写入store
        t.bs.write_block(self.id, self.clone());

        // 返回这个被复制过的 Node 节点
        Some(w)
    }
}

impl<T: Clone + PartialOrd> BTree<T> {
    pub fn new(b: usize) -> Self {
        let mut tree = Self {
            max_children_amount: b | 1, // 变成奇数
            B: b / 2,
            n: 0,
            root_index: 0,
            bs: BlockStore::new(),
        };
        tree.root_index = Node::<T>::new(&mut tree).id;
        tree
    }
    fn find_it(a: &[Option<T>], x: &T) -> i32 {
        let mut lo = 0;
        let mut hi = a.len();
        // 二分查找
        while hi != lo {
            let m = (hi + lo) / 2;
            match &a[m] {
                // None 区域，继续找，直到 hi == lo
                None => hi = m,
                Some(v) if x < v => hi = m,
                Some(v) if x > v => lo = m + 1,
                // 如果 x 正好在数组 a 中，就返回 -m-1
                _ => return -(m as i32) - 1,
            }
        }
        // 找到需要插入的位置，返回 lo
        lo as i32
    }
    fn add_recursive(&mut self, mut x: T, ui: usize) -> Result<Option<Node<T>>, ()> {
        // 从store中的 ui 位置读取 u 节点
        if let Some(mut u) = self.bs.read_block(ui) {
            // 在 u 里面的 keys 中，查找 x 应该要放入到 u.keys 中的位置
            let i = Self::find_it(&u.keys, &x);

            // 如果 x 在 u 的 keys 中，返回 Err
            if i < 0 {
                return Err(());
            }

            // 如果要放 x 的地方是一个 leaf 节点
            if u.children[i as usize] < 0 {
                // 直接放入
                u.add(x, -1);
                // 更新这个 u 节点的信息
                self.bs.write_block(u.id, u.clone());
            } else {
            // 如果要放 x 的地方不是一个 leaf 节点，也就是放到 internal 节点上
                // 在 u 节点的某一个子节点 u' 递归地把 x 添加进去
                // 此时可能会生成一个新节点 w，上一次递归时 split 生成的 w
                let w = self.add_recursive(x, u.children[i as usize] as usize)?;
                // 如果真的生成了，说明 u' 被 split 过了
                // w 节点是 u' 节点的一半
                if let Some(mut w) = w {
                    // 这个时候 u 节点变成 w 节点 和已经 split 的 u' 节点的父节点，u 节点需要拿走 w 节点的第一个 key
                    x = w.remove(0).unwrap();
                    u.add(x, w.id as i32);

                    self.bs.write_block(w.id, w);
                    self.bs.write_block(u.id, u.clone());
                }
            }


            // 如果 u 满了，说明需要对 u 这个节点进行 split
            if u.is_full() {
                Ok(u.split(self))
            } else {
                Ok(None)
            }

        } else {
            Err(())
        }
    }
    fn merge(&mut self, u: &mut Node<T>, i: usize, v: &mut Node<T>, w: &mut Node<T>) {
        // 确保 v 和 w 的顺序为 [..., v, w, ...]
        // 逻辑上，合并成一个新的 v，删除 w
        assert_eq!(v.id, u.children[i] as usize);
        assert_eq!(w.id, u.children[i + 1] as usize);

        let sv = v.size();
        let sw = w.size();

        // w 的 key 都给 v
        for (i, key) in w.keys[0..sw].iter_mut().enumerate() {
            v.keys[sv + 1 + i] = key.take();
        }
        // w 的 children 都给 v
        for (i, chd) in w.children[0..(sw + 1)].iter_mut().enumerate() {
            v.children[sv + 1 + i] = *chd;
            *chd = -1;
        }


        // u 指向 v 位置的 key，给到 v 的 sv 这个位置
        v.keys[sv] = u.keys[i].take();


        // 那么 u 的 key 需要 rotate
        for j in (i + 1)..self.max_children_amount {
            u.keys.swap(j - 1, j);
        }
        u.keys[self.max_children_amount - 1].take();

        // 那么 u 的 children 需要 rotate
        for j in (i + 2)..(self.max_children_amount + 1) {
            u.children.swap(j - 1, j);
        }
        u.children[self.max_children_amount] = -1;


        // 更新 u v 的节点信息
        self.bs.write_block(u.id, u.clone());
        self.bs.write_block(v.id, v.clone());

        // 删除 w
        self.bs.free_block(w.id);
    }

    // w 的兄弟节点 v，节点顺序为 [..., v, w, ...]
    // 从 v -> w
    // u 是 v 和 w 的父节点
    // 这里 i 传的为 v 的位置
    fn shift_lr(&mut self, u: &mut Node<T>, i: usize, v: &mut Node<T>, w: &mut Node<T>) {
        let sv = v.size();
        let sw = w.size();

        // 要从 v -> w 移动的 key 数量，其实就是 (sv - sw) / 2
        let shift = (sw + sv) / 2 - sw;

        // w 要接收新的 key，这里需要腾出空间容纳这么多的 key
        w.keys.rotate_right(shift);
        w.children.rotate_right(shift);

        // 首先要更新 u 的 key
        // 在 w 的 shift - 1 的位置，u 需要把原来指向 v 的那个 key 给 w
        w.keys[shift - 1] = u.keys[i].take();
        // 更新 u 指向 v 的 key，把 v 的倒数第 shift 的 key 给 u
        u.keys[i] = v.keys[sv - shift].take();

        // 然后开始移动 key
        // 开始移动从 v -> w 移动 key
        for (i, key) in v.keys[(sv - shift + 1)..sv].iter_mut().enumerate() {
            w.keys[i] = key.take();
        }
        // children 也要移动
        for (i, chd) in v.children[(sv - shift + 1)..(sv + 1)].iter_mut().enumerate() {
            w.children[i] = *chd;
            *chd = -1;
        }


        // 更新 u v w 的节点信息
        self.bs.write_block(u.id, u.clone());
        self.bs.write_block(v.id, v.clone());
        self.bs.write_block(w.id, w.clone());
    }


    // w 的兄弟节点 v，节点顺序为 [w, v, ...]
    // 从 w <- v
    // u 是 v 和 w 的父节点
    // 这里 i 传的为 w 的位置
    // i = 0
    fn shift_rl(&mut self, u: &mut Node<T>, i: usize, v: &mut Node<T>, w: &mut Node<T>) {
        let sv = v.size();
        let sw = w.size();

        // 要从 v -> w 移动的 key 数量，其实就是 (sv - sw) / 2
        let shift = (sw + sv) / 2 - sw;

        // 这里就不需要腾出空间了，因为空间是有的

        // 在 w 的 sw 的位置，u 需要把原来指向 w 的那个 key 给 w
        w.keys[sw] = u.keys[i].take();

        // 然后开始移动 key
        // 开始移动从 v -> w 移动 key
        for (i, key) in v.keys[0..(shift - 1)].iter_mut().enumerate() {
            w.keys[sw + 1 + i] = key.take();
        }
        // children 也要移动
        for (i, chd) in v.children[0..shift].iter_mut().enumerate() {
            w.children[sw + 1 + i] = *chd;
            *chd = -1;
        }

        // 移动完成
        // 更新 u 的 key
        // 在 v 的 shift - 1 的位置，v 需要把剩下的第一个 key 给 u
        u.keys[i] = v.keys[shift - 1].take();


        // 移动完成后，shift 前面的位置
        // keys 都变成 None
        // children 都是 -1
        // 需要把后面的数据 rotate 到前面来
        for i in 0..(self.max_children_amount - shift) {
            v.keys.swap(i, shift + i);
        }
        // 然后为了安全起见，把剩下的再次设为 None
        for key in v.keys[(sv - shift)..self.max_children_amount].iter_mut() {
            key.take();
        }

        // children 的逻辑也是一样
        // 需要把后面的数据 rotate 到前面来
        for i in 0..(self.max_children_amount - shift + 1) {
            v.children.swap(i, shift + i);
        }
        // 然后为了安全起见，把剩下的再次设为 -1
        for chd in v.children[(sv - shift + 1)..(self.max_children_amount + 1)].iter_mut() {
            *chd = -1;
        }

        // 更新 u v w 的节点信息
        self.bs.write_block(u.id, u.clone());
        self.bs.write_block(v.id, v.clone());
        self.bs.write_block(w.id, w.clone());
    }
    fn check_underflow_zero(&mut self, u: &mut Node<T>, i: usize) {
        // i = 0
        // u 节点的第 0 个子节点，w
        if let Some(ref mut w) = self.bs.read_block(u.children[i] as usize) {
            // 如果 w 中拥有的 key 的数量 < B-1
            if w.size() < self.B - 1 {
                // w 的兄弟节点 v，节点顺序为 [w, v, ...]
                if let Some(ref mut v) = self.bs.read_block(u.children[i + 1] as usize) {
                    // 如果 v 节点有足够多的 key，至少 > B 个
                    if v.size() > self.B {
                        // 那么可以从 v 向 w 移动一些 key，保证 v 和 w 都至少有 B-1 个 key
                        // i 是 0，即为 w 所在位置
                        self.shift_rl(u, i, v, w);
                    } else {
                        // 如果 v 节点的 key 都不够，则合并 v 和 w 节点，保证 v 和 w 都至少有 B-1 个 key
                        // i 是 0，即为 w 所在位置
                        // 注意这里参数是反的
                        self.merge(u, i, w, v);
                    }
                }
            }
        }
    }
    fn check_underflow_nonzero(&mut self, u: &mut Node<T>, i: usize) {
        // u 节点的第 i 个子节点，w
        if let Some(ref mut w) = self.bs.read_block(u.children[i] as usize) {
            // 如果 w 中拥有的 key 的数量 < B-1
            if w.size() < self.B - 1 {
                // w 的兄弟节点 v，节点顺序为 [..., v, w, ...]
                if let Some(ref mut v) = self.bs.read_block(u.children[i - 1] as usize) {
                    // 如果 v 节点有足够多的 key，至少 > B 个
                    if v.size() > self.B {
                        // 那么可以从 v 向 w 移动一些 key，保证 v 和 w 都至少有 B-1 个 key
                        // i-1，即为 v 所在位置
                        self.shift_lr(u, i - 1, v, w);
                    } else {
                        // 如果 v 节点的 key 都不够，则合并 v 和 w 节点，保证 v 和 w 都至少有 B-1 个 key
                        // i-1，即为 v 所在位置
                        self.merge(u, i - 1, v, w);
                    }
                }
            }
        }
    }
    fn check_underflow(&mut self, u: &mut Node<T>, i: usize) {
        // 叶子节点没必要做 check
        if u.children[i] < 0 {
            return;
        }
        if i == 0 {
            // 确保 u 节点的 第 0 个子节点至少还有 B-1 个 key
            self.check_underflow_zero(u, i);
        } else {
            // 确保 u 节点的 第 i 个子节点至少还有 B-1 个 key
            self.check_underflow_nonzero(u, i);
        }
    }
    fn remove_smallest(&mut self, ui: i32) -> Option<T> {
        // 读取节点信息
        let mut u = self.bs.read_block(ui as usize);
        if let Some(ref mut u) = u {
            // 如果读取到的节点是 leaf 节点，直接删除即可
            if u.is_leaf() {
                let y = u.remove(0);
                self.bs.write_block(u.id, u.clone());
                y
            // 如果读取到的节点不是 leaf 节点，即是 internal 节点，需要做递归删除
            } else {
                // 注意这里要删除最小的，那么只能是删除 u 节点的第 0 个子节点
                let y = self.remove_smallest(u.children[0]);
                // 确保 u 节点的 第 0 个子节点至少还有 B-1 个 key
                self.check_underflow(u, 0);
                y
            }
        } else {
            None
        }
    }
    fn remove_recursive(&mut self, x: &T, ui: i32) -> Option<T> {
        if ui < 0 {
            return None;
        }
        let mut u = self.bs.read_block(ui as usize);
        match u {
            Some(ref mut u) => {
                // 首先要找到要删除的 x
                let mut i = Self::find_it(&u.keys, x);
                // i < 0 表示找到了要删除的 x
                if i < 0 {
                    // 找到的 x 的位置是 -m-1，这里先把它位置 abs
                    i = -(i + 1);
                    // 如果 x 是在 leaf 节点上
                    if u.is_leaf() {
                        // 直接删除即可
                        let y = u.remove(i as usize);
                        self.bs.write_block(u.id, u.clone());
                        y
                    // 如果 x 不是在 leaf 节点上，而是在 internal 节点上
                    } else {
                        // 先在 u 节点的某个子节点 u' 中删除最小的那个值 x'，位置是 i+1
                        // x' 是 比 x 大的最小值，所以是 i+1
                        // 如果选择 i 则是比 x 小的值
                        let x = self.remove_smallest(u.children[i as usize + 1]);

                        // 删除 x
                        let y = u.keys[i as usize].take();

                        // 原来要删除 x 的位置，替换为 x'
                        u.keys[i as usize] = x;

                        self.bs.write_block(u.id, u.clone());

                        // 确保 u 节点的 i+1 处的子节点至少还有 B-1 个 key
                        self.check_underflow(u, i as usize + 1);

                        y
                    }
                // i >= 0 表示没找到要删除的 x
                } else {
                    // 从 u 节点的 i 节点处递归删除
                    let y = self.remove_recursive(x, u.children[i as usize]);
                    if y.is_some() {
                        // 删除成功
                        // 确保 u 节点的 i 处的子节点至少还有 B-1 个 key
                        self.check_underflow(u, i as usize);
                        y
                    } else {
                        None
                    }
                }
            }
            None => None,
        }
    }
}

impl<T> SortedSet<T> for BTree<T>
where
    T: Clone + PartialOrd,
{
    fn size(&self) -> usize {
        self.n
    }
    fn add(&mut self, x: T) -> bool {
        match self.add_recursive(x, self.root_index) {
            Ok(w) => {
                if let Some(mut w) = w {
                    let x = w.remove(0);
                    let mut newroot = Node::new(self);

                    newroot.keys[0] = x;
                    newroot.children[0] = self.root_index as i32;
                    newroot.children[1] = w.id as i32;

                    // 更新 root 节点 index
                    self.root_index = newroot.id;

                    self.bs.write_block(w.id, w);
                    self.bs.write_block(self.root_index, newroot);
                }
                self.n += 1;
                true
            }
            Err(()) => false,
        }
    }
    fn remove(&mut self, x: &T) -> Option<T> {
        match self.remove_recursive(x, self.root_index as i32) {
            Some(y) => {
                self.n -= 1;
                let r = self.bs.read_block(self.root_index);
                if let Some(r) = r {
                    if r.size() == 0 && self.n > 0 {
                        // 删除成功

                        // 更新 root 节点 index
                        self.root_index = r.children[0] as usize;
                    }
                }
                Some(y)
            }
            None => None,
        }
    }
    fn find(&self, x: &T) -> Option<T> {
        let mut z = None;
        let mut ui = self.root_index as i32;
        while ui >= 0 {
            // 从根节点开始找
            let u = self.bs.read_block(ui as usize)?;
            let i = Self::find_it(&u.keys, &x);
            // 找到返回
            if i < 0 {
                return u.keys[(-(i + 1)) as usize].clone();
            }
            // 没找到继续找
            if u.keys[i as usize].is_some() {
                z = u.keys[i as usize].clone()
            }

            // 更新 children 指针
            ui = u.children[i as usize];
        }
        z
    }
}
