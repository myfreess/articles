# Myers diff 3

This article is the third in the [diff series](/example/myers-diff/index.md). In the [previous part](/example/myers-diff/myers-diff2.md), we explored the full Myers algorithm and its limitations. In this post, we'll learn how to implement a variant of the Myers algorithm that operates with linear space complexity.

## Divide and Conquer

The linear variant of Myers' diff algorithm used by Git employs a concept called the _Snake_ (sometimes referred to as the _Middle Snake_) to break down the entire search process. A Snake in the edit graph represents a diagonal movement of 0 to N steps after a single left or down move. The linear Myers algorithm finds the middle Snake on the optimal edit path and uses it to divide the entire edit graph into two parts. The subsequent steps apply the same technique to the resulting subgraphs, eventually producing a complete edit path.

```bash
    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
 0  o---o---o---o---o---o---o
    |   |   |   |   |   |   |
 1  o---o---o---o---o---o---o
    |   | \ |   |   |   |   |
 2  o---o---o---o---o---o---o
    |   |   |   |   |   |   |
 3  o---o---o---o---o---o---o
    |   |   |   |   | \ |   |
 4  o---o---o---o---o---o---o
    |   |   |   |   |   |   |
 5  o---o---o---o---o---o---o
                              \
 6                              @
                                  \
 7                                  @---o---o---o---o---o---o
                                        |   |   |   |   |   |
 8                                      o---o---o---o---o---o
                                        | \ |   |   |   |   |
 9                                      o---o---o---o---o---o
                                        |   |   |   |   |   |
10                                      o---o---o---o---o---o
                                        |   |   |   |   |   |
11                                      o---o---o---o---o---o
                                        |   |   | \ |   |   |
12                                      o---o---o---o---o---o
                                        |   |   |   |   |   |
13                                      o---o---o---o---o---o
                                        |   |   |   |   | \ |
14                                      o---o---o---o---o---o
```

> A quick recap: The optimal edit path is the one that has the shortest distance to the endpoint (a diagonal distance of zero), and there can be more than one such path.

Attentive readers may have noticed a chicken-and-egg problem: to find a Snake, you need an optimal edit path, but to get an optimal edit path, it seems like you need to run the original Myers algorithm first.

In fact, the idea behind the linear Myers algorithm is somewhat unconventional: it alternates the original Myers algorithm from both the top-left and bottom-right corners, but without storing the history. Instead, it simply checks if the searches from both sides overlap. When they do, the overlapping portion is returned as the Middle Snake.

This approach seems straightforward, but there are still some details to sort out.

When searching from the bottom-right, the diagonal coordinate can no longer be referred to as _k_. We need to define a new diagonal coordinate **c = k - delta**. This coordinate is the mirror image of _k_, perfectly suited for reverse direction search.

```bash
        x                       k
                                  0     1     2     3
        0     1     2     3         \     \     \     \
  y  0  o-----o-----o-----o           o-----o-----o-----o
        |     |     |     |      -1   |     |     |     | \
        |     |     |     |         \ |     |     |     |   2
     1  o-----o-----o-----o           o-----o-----o-----o
        |     | \   |     |      -2   |     | \   |     | \
        |     |   \ |     |         \ |     |   \ |     |   1
     2  o-----o-----o-----o           o-----o-----o-----o
                                        \     \     \     \
                                        -3    -2    -1      0
                                                              c
```

How do we determine if the searches overlap? Simply check if the position on a diagonal line in the forward search has an _x_ value greater than that in the reverse search. However, since the _k_ and _c_ coordinates differ for the same diagonal, the conversion can be a bit tricky.

### Code Implementation

We'll start by defining `Snake` and `Box` types, representing the middle snake and the sub-edit graphs (since they're square, we call them `Box`).

```moonbit
struct Box {
  left : Int
  right : Int
  top : Int
  bottom : Int
} derive(Show)

///|
struct Snake {
  start : (Int, Int)
  end : (Int, Int)
} derive(Show)

///|
fn Box::width(self : Self) -> Int {
  self.right - self.left
}

///|
fn Box::height(self : Self) -> Int {
  self.bottom - self.top
}

///|
fn Box::size(self : Self) -> Int {
  self.width() + self.height()
}

///|
fn Box::delta(self : Self) -> Int {
  self.width() - self.height()
}
```

To avoid getting bogged down in details too early, let's assume we already have a function `midpoint : (Box, Array[Line], Array[Line]) -> Snake?` to find the middle snake. Then, we can build the function `find_path` to search for the complete path.

```moonbit
fn Box::find_path(
  box : Self,
  old~ : Array[Line],
  new~ : Array[Line]
) -> Iter[(Int, Int)]? {
  guard box.midpoint(old~, new~) is Some(snake) else { None }
  let start = snake.start
  let end = snake.end
  let headbox = Box::{
    left: box.left,
    top: box.top,
    right: start.0,
    bottom: start.1,
  }
  let tailbox = Box::{
    left: end.0,
    top: end.1,
    right: box.right,
    bottom: box.bottom,
  }
  let head = headbox.find_path(old~, new~).or(Iter::singleton(start))
  let tail = tailbox.find_path(old~, new~).or(Iter::singleton(end))
  Some(head.concat(tail))
}
```

The implementation of `find_path` is straightforward, but `midpoint` is a bit more complex:

- For a `Box` of size 0, return `None`.
- Calculate the search boundaries. Since forward and backward searches each cover half the distance, divide by two. However, if the size of the `Box` is odd, add one more to the forward search boundary.
- Store the results of the forward and backward searches in two arrays.
- Alternate between forward and backward searches, returning `None` if no result is found.

```moonbit
fn Box::midpoint(self : Self, old~ : Array[Line], new~ : Array[Line]) -> Snake? {
  if self.size() == 0 {
    return None
  }
  let max = {
    let half = self.size() / 2
    if is_odd(self.size()) {
      half + 1
    } else {
      half
    }
  }
  let vf = BPArray::make(2 * max + 1, 0)
  vf[1] = self.left
  let vb = BPArray::make(2 * max + 1, 0)
  vb[1] = self.bottom
  for d = 0; d < max + 1; d = d + 1 {
    match self.forward(forward=vf, backward=vb, d, old~, new~) {
      None =>
        match self.backward(forward=vf, backward=vb, d, old~, new~) {
          None => continue
          res => return res
        }
      res => return res
    }
  } else {
    None
  }
}
```

The forward and backward searches have some modifications compared to the original Myers algorithm, which need a bit of explanation:

- Since we need to return the snake, the search process must calculate the previous coordinate (`px` stands for previous x).
- The search now works within a `Box` (not the global edit graph), so calculating `y` from `x` (or vice versa) requires conversion.
- The backward search minimizes `y` as a heuristic strategy, but minimizing `x` would also work.

```moonbit
fn Box::forward(
  self : Self,
  forward~ : BPArray[Int],
  backward~ : BPArray[Int],
  depth : Int,
  old~ : Array[Line],
  new~ : Array[Line]
) -> Snake? {
  for k = depth; k >= -depth; k = k - 2 {
    let c = k - self.delta()
    let mut x = 0
    let mut px = 0
    if k == -depth || (k != depth && forward[k - 1] < forward[k + 1]) {
      x = forward[k + 1]
      px = x
    } else {
      px = forward[k - 1]
      x = px + 1
    }
    let mut y = self.top + (x - self.left) - k
    let py = if depth == 0 || x != px { y } else { y - 1 }
    while x < self.right && y < self.bottom && old[x].text == new[y].text {
      x = x + 1
      y = y + 1
    }
    forward[k] = x
    if is_odd(self.delta()) &&
      (c >= -(depth - 1) && c <= depth - 1) &&
      y >= backward[c] {
      return Some(Snake::{ start: (px, py), end: (x, y) })
    }
  }
  return None
}

///|
fn Box::backward(
  self : Self,
  forward~ : BPArray[Int],
  backward~ : BPArray[Int],
  depth : Int,
  old~ : Array[Line],
  new~ : Array[Line]
) -> Snake? {
  for c = depth; c >= -depth; c = c - 2 {
    let k = c + self.delta()
    let mut y = 0
    let mut py = 0
    if c == -depth || (c != depth && backward[c - 1] > backward[c + 1]) {
      y = backward[c + 1]
      py = y
    } else {
      py = backward[c - 1]
      y = py - 1
    }
    let mut x = self.left + (y - self.top) + k
    let px = if depth == 0 || y != py { x } else { x + 1 }
    while x > self.left && y > self.top && old[x - 1].text == new[y - 1].text {
      x = x - 1
      y = y - 1
    }
    backward[c] = y
    if is_even(self.delta()) && (k >= -depth && k <= depth) && x <= forward[k] {
      return Some(Snake::{ start: (x, y), end: (px, py) })
    }
  }
  return None
}
```

Now we can implement `linear_diff`

```moonbit
fn linear_diff(old~ : Array[Line], new~ : Array[Line]) -> Array[Edit]? {
  let initial_box = Box::{
    left: 0,
    top: 0,
    right: old.length(),
    bottom: new.length(),
  }
  guard initial_box.find_path(old~, new~) is Some(path) else { None }
  // path length >= 2
  let xy = path.take(1).collect()[0] // (0, 0)
  let mut x1 = xy.0
  let mut y1 = xy.1
  let edits = Array::new(capacity=old.length() + new.length())
  path
  .drop(1)
  .each(fn(xy) {
    let x2 = xy.0
    let y2 = xy.1
    while x1 < x2 && y1 < y2 && old[x1].text == new[y1].text {
      edits.push(Equal(old=old[x1], new=new[y1]))
      x1 = x1 + 1
      y1 = y1 + 1
    }
    if x2 - x1 < y2 - y1 {
      edits.push(Insert(new=new[y1]))
      y1 += 1
    }
    if x2 - x1 > y2 - y1 {
      edits.push(Delete(old=old[x1]))
      x1 += 1
    }
    while x1 < x2 && y1 < y2 && old[x1].text == new[y1].text {
      edits.push(Equal(old=old[x1], new=new[y1]))
      x1 = x1 + 1
      y1 = y1 + 1
    }
    x1 = x2
    y1 = y2
  })
  return Some(edits)
}

fn pprint_diff(diff : Array[Edit]) -> String {
  let buf = StringBuilder::new(size_hint=100)
  for i = 0; i < diff.length(); i = i + 1 {
    buf.write_string(pprint_edit(diff[i]))
    buf.write_char('\n')
  } else {
    buf.to_string()
  }
}

///|
test "myers diff" {
  let old = lines("A\nB\nC\nA\nB\nB\nA")
  let new = lines("C\nB\nA\nB\nA\nC")
  let r = linear_diff(old~, new~).unwrap()
  inspect(
    pprint_diff(r),
    content=
      #|-    1         A
      #|-    2         B
      #|     3    1    C
      #|-    4         A
      #|     5    2    B
      #|+         3    A
      #|     6    4    B
      #|     7    5    A
      #|+         6    C
      #|
    ,
  )
}
```

## Conclusion

In addition to the default diff algorithm, Git also offers another diff algorithm called patience diff. It differs significantly from Myers diff in approach and sometimes produces more readable diff results.

## Appendix

```moonbit
///|
struct Line {
  number : Int // Line number
  text : String // Does not include newline
} derive(Show, ToJson)

///|
fn Line::new(number : Int, text : String) -> Line {
  Line::{ number, text }
}

///|
fn lines(str : String) -> Array[Line] {
  let lines = Array::new(capacity=50)
  let mut line_number = 0
  for line in str.split("\n") {
    line_number = line_number + 1
    lines.push(Line::new(line_number, line.to_string()))
  } else {
    return lines
  }
}

///|
test "lines" {
  @json.inspect(
    lines(""),
    content=
      [{"number":1,"text":""}]
    ,
  )
  @json.inspect(
    lines("\n"),
    content=
      [{"number":1,"text":""},{"number":2,"text":""}]
    ,
  )
  @json.inspect(
    lines("aaa"),
    content=
      [{"number":1,"text":"aaa"}]
    ,
  )
  @json.inspect(
    lines("aaa\nbbb"),
    content=
      [{"number":1,"text":"aaa"},{"number":2,"text":"bbb"}]
    ,
  )
}

///|
type BPArray[T] Array[T] // BiPolar Array

///|
fn[T] BPArray::make(capacity : Int, default : T) -> BPArray[T] {
  let arr = Array::make(capacity, default)
  BPArray(arr)
}

///|
fn copy(self : BPArray[Int]) -> BPArray[Int] {
  let BPArray(arr) = self
  let newarr = Array::make(arr.length(), 0)
  for i = 0; i < arr.length(); i = i + 1 {
    newarr[i] = arr[i]
  } else {
    BPArray(newarr)
  }
}

///|
fn[T] op_get(self : BPArray[T], idx : Int) -> T {
  let BPArray(arr) = self
  if idx < 0 {
    arr[arr.length() + idx]
  } else {
    arr[idx]
  }
}

///|
fn[T] op_set(self : BPArray[T], idx : Int, elem : T) -> Unit {
  let BPArray(arr) = self
  if idx < 0 {
    arr[arr.length() + idx] = elem
  } else {
    arr[idx] = elem
  }
}

///|
test "bparray" {
  let foo = BPArray::make(10, "foo")
  foo[9] = "bar"
  foo[8] = "baz"
  assert_eq(foo[-1], "bar")
  assert_eq(foo[-2], "baz")
}

enum Edit {
  Insert(new~ : Line)
  Delete(old~ : Line)
  Equal(old~ : Line, new~ : Line) // old, new
} derive(Show)

let line_width = 4

///|
fn pad_right(s : String, width : Int) -> String {
  String::make(width - s.length(), ' ') + s
}

///|
fn pprint_edit(edit : Edit) -> String {
  match edit {
    Insert(_) as edit => {
      let tag = "+"
      let old_line = pad_right("", line_width)
      let new_line = pad_right(edit.new.number.to_string(), line_width)
      let text = edit.new.text
      "\{tag} \{old_line} \{new_line}    \{text}"
    }
    Delete(_) as edit => {
      let tag = "-"
      let old_line = pad_right(edit.old.number.to_string(), line_width)
      let new_line = pad_right("", line_width)
      let text = edit.old.text
      "\{tag} \{old_line} \{new_line}    \{text}"
    }
    Equal(_) as edit => {
      let tag = " "
      let old_line = pad_right(edit.old.number.to_string(), line_width)
      let new_line = pad_right(edit.new.number.to_string(), line_width)
      let text = edit.old.text
      "\{tag} \{old_line} \{new_line}    \{text}"
    }
  }
}

///|
fn is_odd(n : Int) -> Bool {
  (n & 1) == 1
}

///|
fn is_even(n : Int) -> Bool {
  (n & 1) == 0
}
```