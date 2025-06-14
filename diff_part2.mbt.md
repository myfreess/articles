# Myers diff 2

This is the second post in the diff series. In the [previous one](https://www.moonbitlang.com/docs/examples/myers-diff), we learned how to transform the process of computing diffs into a graph search problem and how to search for the shortest edit distance. In this article, we will learn how to extend the search process from the previous post to obtain the complete edit sequence.

## Recording the Search Process

The first step to obtaining the complete edit sequence is to save the entire editing process. This step is relatively simple; we just need to save the current search depth `d` and the graph node with depth `d` at the beginning of each search round.

```moonbit
fn shortest_edit(
  old~ : Array[Line],
  new~ : Array[Line]
) -> Array[(BPArray[Int], Int)] {
  let n = old.length()
  let m = new.length()
  let max = n + m
  let v = BPArray::make(2 * max + 1, 0)
  let trace = []
  fn push(v : BPArray[Int], d : Int) -> Unit {
    trace.push((v, d))
  }
  // v[1] = 0
  for d = 0; d < max + 1; d = d + 1 {
    push(v.copy(), d)
    for k = -d; k < d + 1; k = k + 2 {
      let mut x = 0
      let mut y = 0
      // if d == 0 {
      //   x = 0
      // }
      if k == -d || (k != d && v[k - 1] < v[k + 1]) {
        x = v[k + 1]
      } else {
        x = v[k - 1] + 1
      }
      y = x - k
      while x < n && y < m && old[x].text == new[y].text {
        x = x + 1
        y = y + 1
      }
      v[k] = x
      if x >= n && y >= m {
        return trace
      }
    }
  } else {
    abort("impossible")
  }
}
```

## Backtracking the Edit Path

After recording the entire search process, the next step is to walk back from the endpoint to find the path taken. But before we do that, let's first define the `Edit` type.

```moonbit
enum Edit {
  Insert(new~ : Line)
  Delete(old~ : Line)
  Equal(old~ : Line, new~ : Line) // old, new
} derive(Show)
```

Next, let's perform the backtracking.

The method of backtracking is essentially the same as forward search, just in reverse.

- Calculate the current `k` value using `x` and `y`.

- Access the historical records and use the same judgment criteria as in forward search to find the `k` value at the previous search round.

- Restore the coordinates of the previous search round.

- Try free movement and record the corresponding edit actions.

- Determine the type of edit that caused the change in `k` value.

- Continue iterating.

```moonbit
fn backtrack(
  new~ : Array[Line],
  old~ : Array[Line],
  trace : Array[(BPArray[Int], Int)]
) -> Array[Edit] {
  let mut x = old.length()
  let mut y = new.length()
  let edits = Array::new(capacity=trace.length())
  // end backtrack_fst definition
  // start backtrack_snd definition
  for i = trace.length() - 1; i >= 0; i = i - 1 {
    let (v, d) = trace[i]
    let k = x - y
    let prev_k = if k == -d || (k != d && v[k - 1] < v[k + 1]) {
      k + 1
    } else {
      k - 1
    }
    let prev_x = v[prev_k]
    let prev_y = prev_x - prev_k
    while x > prev_x && y > prev_y {
      x = x - 1
      y = y - 1
      edits.push(Equal(old=old[x], new=new[y]))
    }
    if d > 0 {
      if x == prev_x {
        edits.push(Insert(new=new[prev_y]))
      } else if y == prev_y {
        edits.push(Delete(old=old[prev_x]))
      }
      x = prev_x
      y = prev_y
    }
  }
  // end backtrack_snd definition
  return edits
}
```

Combining the two functions, we get a complete `diff` implementation.

```moonbit
fn diff(old~ : Array[Line], new~ : Array[Line]) -> Array[Edit] {
  let trace = shortest_edit(old~, new~)
  backtrack(old~, new~, trace)
}
```

## Printing the Diff

To print a neat diff, we need to left-align the text. Also, due to the order issue during backtracking, we need to print from back to front.

```moonbit
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
fn pprint_diff(diff : Array[Edit]) -> String {
  let buf = StringBuilder::new(size_hint=100)
  for i = diff.length(); i > 0; i = i - 1 {
    buf.write_string(pprint_edit(diff[i - 1]))
    buf.write_char('\n')
  } else {
    buf.to_string()
  }
}
```

The result is as follows:

```moonbit
test "myers diff" {
  let old = lines("A\nB\nC\nA\nB\nB\nA")
  let new = lines("C\nB\nA\nB\nA\nC")
  inspect(
    pprint_diff(diff(old~, new~)),
    content=
      #|-    1         A
      #|-    2         B
      #|     3    1    C
      #|+         2    B
      #|     4    3    A
      #|     5    4    B
      #|-    6         B
      #|     7    5    A
      #|+         6    C
      #|
    ,
  )
}
```

## Conclusion

The Myers algorithm demonstrated above is complete, but due to the frequent copying of arrays, it has a very large space overhead. Therefore, most software like Git uses a linear variant of the diff algorithm (found in the appendix of the original paper). This variant may sometimes produce diffs of lower quality (harder for humans to read) than the standard Myers algorithm but can still ensure the shortest edit sequence.

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
```