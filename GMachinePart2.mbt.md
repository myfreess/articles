# G-Machine 2

This article is the second in the series on implementing lazy evaluation in MoonBit. In the first part, we explored the purposes of lazy evaluation and a typical abstract machine for lazy evaluation, the G-Machine, and implemented some basic G-Machine instructions. In this article, we will further extend the G-Machine implementation from the previous article to support `let` expressions and basic arithmetic, comparison, and other operations.

## let Expressions

The `let` expression in coreF differs slightly from that in MoonBit. A `let` expression can create multiple variables but can only be used within a limited scope. Here is an example:

```moonbit skip
{
  let x = n + m
  let y = x + 42
  x * y
}
```

Equivalent coreF expression:

```clojure
(let ([x (add n m)]
      [y (add x 42)])
  (mul x y)) ;; xy can only be used within let
```

It is important to note that coreF's `let` expressions must follow a sequential order. For example, the following is not valid:

```clojure
(let ([y (add x 42)]
      [x (add n m)])
  (mul x y))
```

In contrast, `letrec` is more complex as it allows the local variables defined to reference each other without considering the order of their definitions.

Before implementing `let` (and the more complex `letrec`), we first need to modify the current parameter passing method. The local variables created by `let` should intuitively be accessed in the same way as parameters, but the local variables defined by `let` do not correspond to `NApp` nodes. Therefore, we need to adjust the stack parameters before calling the supercombinator.

The adjustment is done in the implementation of the `Unwind` instruction. If the supercombinator has no parameters, it is the same as the original unwind. When there are parameters, the top address of the supercombinator node is discarded, and the `rearrange` function is called.

```moonbit
fn GState::rearrange(self : GState, n : Int) -> Unit {
  let appnodes = self.stack.take(n)
  let args = appnodes.map(fn(addr) {
    guard self.heap[addr] is NApp(_, arg)
    arg
  })
  self.stack = args + appnodes.drop(n - 1)
}
```

The `rearrange` function assumes that the first N addresses on the stack point to a series of `NApp` nodes. It keeps the bottommost one (used as Redex update), cleans up the top N-1 addresses, and then places N addresses that directly point to the parameters.

After this, both parameters and local variables can be accessed using the same command by changing the `PushArg` instruction to a more general `Push` instruction.

```moonbit
fn GState::push(self : GState, offset : Int) -> Unit {
  // Push(n) a0 : . . . : an : s
  //     =>  an : a0 : . . . : an : s
  let addr = self.stack.unsafe_nth(offset)
  self.put_stack(addr)
}
```

The next issue is that we need something to clean up. Consider the following expression:

```clojure
(let ([x1 e1]
      [x2 e2])
  expr)
```

After constructing the graph corresponding to the expression `expr`, the stack still contains addresses pointing to e1 and e2 (corresponding to variables x1 and x2), as shown below (the stack grows from bottom to top):

```
<Address pointing to expr>
       |
<Address pointing to x2>
       |
<Address pointing to x1>
       |
...remaining stack...
```

Therefore, we need a new instruction to clean up these no longer needed addresses. It is called `Slide`. As the name suggests, the function of `Slide(n)` is to skip the first address and delete the following N addresses.

```moonbit
fn GState::slide(self : GState, n : Int) -> Unit {
  let addr = self.pop1()
  self.stack = self.stack.drop(n).prepend(addr)
}
```

Now we can compile `let`. We will compile the expressions corresponding to local variables using the `compileC` function. Then, traverse the list of variable definitions (`defs`), compile and update the corresponding offsets in order. Finally, use the passed `comp` function to compile the main expression and add the `Slide` instruction to clean up the unused addresses.

> Compiling the main expression using the passed function makes it easy to reuse when adding subsequent features.

```moonbit
fn compileLet(
  comp : (RawExpr[String], List[(String, Int)]) -> List[Instruction],
  defs : List[(String, RawExpr[String])],
  expr : RawExpr[String],
  env : List[(String, Int)]
) -> List[Instruction] {
  let (env, codes) = loop env, @list.empty(), defs {
    env, acc, Empty => (env, acc)
    env, acc, More((name, expr), tail=rest) => {
      let code = expr.compileC(env)
      let env = argOffset(1, env).prepend((name, 0))
      continue env, acc + code, rest
    }
  }
  codes + comp(expr, env) + @list.of([Slide(defs.length())])
}
```

The semantics of `letrec` are more complex - it allows the N variables within the expression to reference each other, so we need to pre-allocate N addresses and place them on the stack. We need a new instruction: `Alloc(N)`, which pre-allocates N `NInd` nodes and pushes the addresses onto the stack sequentially. The addresses in these indirect nodes are negative and only serve as placeholders.

```moonbit
fn GState::alloc_nodes(self : GState, n : Int) -> Unit {
  let dummynode : Node = NInd(Addr(-1))
  for i = 0; i < n; i = i + 1 {
    let addr = self.heap.alloc(dummynode)
    self.put_stack(addr)
  }
}
```

The steps to compile letrec are similar to `let`:

- Use `Alloc(n)` to allocate N addresses.
- Use the `loop` expression to build a complete environment.
- Compile the local variables in `defs`, using the `Update` instruction to update the results to the pre-allocated addresses after compiling each one.
- Compile the main expression and use the `Slide` instruction to clean up.

```moonbit
fn compileLetrec(
  comp : (RawExpr[String], List[(String, Int)]) -> List[Instruction],
  defs : List[(String, RawExpr[String])],
  expr : RawExpr[String],
  env : List[(String, Int)]
) -> List[Instruction] {
  let mut env = env
  loop defs {
    Empty => ()
    More((name, _), tail=rest) => {
      env = argOffset(1, env).prepend((name, 0))
      continue rest
    }
  }
  let n = defs.length()
  fn compileDefs(
    defs : List[(String, RawExpr[String])],
    offset : Int
  ) -> List[Instruction] {
    match defs {
      Empty => comp(expr, env) + @list.of([Slide(n)])
      More((_, expr), tail=rest) =>
        expr.compileC(env) + compileDefs(rest, offset - 1).prepend(Update(offset))
    }
  }
  compileDefs(defs, n - 1).prepend(Alloc(n))
}
```

## Adding Primitives

From this step, we can finally perform basic integer operations such as arithmetic, comparison, and checking if two numbers are equal. First, modify the `Instruction` type to add related instructions.

```moonbit skip
  Add
  Sub
  Mul
  Div
  Neg
  Eq // ==
  Ne // !=
  Lt // <
  Le // <=
  Gt // >
  Ge // >=
  Cond(List[Instruction], List[Instruction])
```

At first glance, implementing these instructions seems simple. Take `Add` as an example: just pop two top addresses from the stack, retrieve the corresponding numbers from memory, perform the operation, and push the result address back onto the stack.

```moonbit skip
fn add(self : GState) -> Unit {
  let (a1, a2) = self.pop2() // Pop two top addresses
  match (self.heap[a1], self.heap[a2]) {
    (NNum(n1), NNum(n2)) => {
      let newnode = Node::NNum(n1 + n2)
      let addr = self.heap.alloc(newnode)
      self.putStack(addr)
    }
    ......
  }
}
```

However, the next problem we face is that this is a lazy evaluation language. The parameters of `add` are likely not yet computed (i.e., not `NNum` nodes). We also need an instruction that can force a computation to give a result or never stop computing. We call it `Eval` (short for Evaluation).

> In jargon, the result of such a computation is called Weak Head Normal Form (WHNF).

At the same time, we need to modify the structure of `GState` and add a state called `dump`. Its type is `List[(List[Instruction], List[Addr])]`, used by `Eval` and `Unwind` instructions.

The implementation of the `Eval` instruction is not complicated:

- Pop the top address of the stack.

- Save the current unexecuted instruction sequence and stack (by putting them into the dump).

- Clear the current stack and place the previously saved address.

- Clear the current instruction sequence and place the `Unwind` instruction.

> This is similar to how strict evaluation languages handle saving caller contexts, but practical implementations would use more efficient methods.

```moonbit
fn GState::eval(self : GState) -> Unit {
  let addr = self.pop1()
  self.put_dump(self.code, self.stack)
  self.stack = @list.of([addr])
  self.code = @list.of([Unwind])
}
```

This simple definition requires modifying the `Unwind` instruction to restore the context when `Unwind` in the `NNum` branch finds that there is a recoverable context (`dump` is not empty).

```moonbit
fn GState::unwind(self : GState) -> Unit {
  let addr = self.pop1()
  match self.heap[addr] {
    NNum(_) =>
      match self.dump {
        Empty => self.put_stack(addr)
        More((instrs, stack), tail=rest_dump) => {
          self.stack = stack
          self.put_stack(addr)
          self.dump = rest_dump
          self.code = instrs
        }
      }
    NApp(a1, _) => {
      self.put_stack(addr)
      self.put_stack(a1)
      self.put_code(@list.of([Unwind]))
    }
    NGlobal(_, n, c) =>
      if self.stack.length() < n {
        abort("Unwinding with too few arguments")
      } else {
        if n != 0 {
          self.rearrange(n)
        } else {
          self.put_stack(addr)
        }
        self.put_code(c)
      }
    NInd(a) => {
      self.put_stack(a)
      self.put_code(@list.of([Unwind]))
    }
  }
}
```

Next, we need to implement arithmetic and comparison instructions. We use two functions to simplify the form of binary operations. The result of the comparison instruction is a boolean value, and for simplicity, we use numbers to represent it: 0 for `false`, 1 for `true`.

```moonbit
fn GState::negate(self : GState) -> Unit {
  let addr = self.pop1()
  match self.heap[addr] {
    NNum(n) => {
      let addr = self.heap.alloc(NNum(-n))
      self.put_stack(addr)
    }
    otherwise =>
      abort("negate: wrong kind of node \{otherwise}, address \{addr}")
  }
}

fn GState::lift_arith2(self : GState, op : (Int, Int) -> Int) -> Unit {
  let (a1, a2) = self.pop2()
  match (self.heap[a1], self.heap[a2]) {
    (NNum(n1), NNum(n2)) => {
      let newnode = Node::NNum(op(n1, n2))
      let addr = self.heap.alloc(newnode)
      self.put_stack(addr)
    }
    (node1, node2) => abort("liftArith2: \{a1} = \{node1} \{a2} = \{node2}")
  }
}

fn GState::lift_cmp2(self : GState, op : (Int, Int) -> Bool) -> Unit {
  let (a1, a2) = self.pop2()
  match (self.heap[a1], self.heap[a2]) {
    (NNum(n1), NNum(n2)) => {
      let flag = op(n1, n2)
      let newnode = if flag { Node::NNum(1) } else { Node::NNum(0) }
      let addr = self.heap.alloc(newnode)
      self.put_stack(addr)
    }
    (node1, node2) => abort("liftCmp2: \{a1} = \{node1} \{a2} = \{node2}")
  }
}
```

Finally, implement branching:

```moonbit
fn GState::condition(
  self : GState,
  i1 : List[Instruction],
  i2 : List[Instruction]
) -> Unit {
  let addr = self.pop1()
  match self.heap[addr] {
    NNum(0) =>
      // false
      self.code = i2 + self.code
    NNum(1) =>
      // true
      self.code = i1 + self.code
    otherwise => abort("cond : \{addr} = \{otherwise}")
  }
}
```

No major adjustments are needed in the compilation part, just add some predefined programs:

```moonbit
let compiled_primitives : List[(String, Int, List[Instruction])] = @list.of([
    // Arith
    (
      "add",
      2,
      @list.of([
        Push(1),
        Eval,
        Push(1),
        Eval,
        Add,
        Update(2),
        Pop(2),
        Unwind,
      ]),
    ),
    (
      "sub",
      2,
      @list.of([
        Push(1),
        Eval,
        Push(1),
        Eval,
        Sub,
        Update(2),
        Pop(2),
        Unwind,
      ]),
    ),
    (
      "mul",
      2,
      @list.of([
        Push(1),
        Eval,
        Push(1),
        Eval,
        Mul,
        Update(2),
        Pop(2),
        Unwind,
      ]),
    ),
    (
      "div",
      2,
      @list.of([
        Push(1),
        Eval,
        Push(1),
        Eval,
        Div,
        Update(2),
        Pop(2),
        Unwind,
      ]),
    ),
    // Compare
    (
      "eq",
      2,
      @list.of([
        Push(1),
        Eval,
        Push(1),
        Eval,
        Eq,
        Update(2),
        Pop(2),
        Unwind,
      ]),
    ),
    (
      "neq",
      2,
      @list.of([
        Push(1),
        Eval,
        Push(1),
        Eval,
        Ne,
        Update(2),
        Pop(2),
        Unwind,
      ]),
    ),
    (
      "ge",
      2,
      @list.of([
        Push(1),
        Eval,
        Push(1),
        Eval,
        Ge,
        Update(2),
        Pop(2),
        Unwind,
      ]),
    ),
    (
      "gt",
      2,
      @list.of([
        Push(1),
        Eval,
        Push(1),
        Eval,
        Gt,
        Update(2),
        Pop(2),
        Unwind,
      ]),
    ),
    (
      "le",
      2,
      @list.of([
        Push(1),
        Eval,
        Push(1),
        Eval,
        Le,
        Update(2),
        Pop(2),
        Unwind,
      ]),
    ),
    (
      "lt",
      2,
      @list.of([
        Push(1),
        Eval,
        Push(1),
        Eval,
        Lt,
        Update(2),
        Pop(2),
        Unwind,
      ]),
    ),
    // MISC
    (
      "negate",
      1,
      @list.of([Push(0), Eval, Neg, Update(1), Pop(1), Unwind]),
    ),
    (
      "if",
      3,
      @list.of([
        Push(0),
        Eval,
        Cond(@list.of([Push(1)]), @list.of([Push(2)])),
        Update(3),
        Pop(3),
        Unwind,
      ]),
    ),
  ],
)
```

and modify the initial instruction sequence

```moonbit
fn run(codes : List[String]) -> Node {
  fn parse_then_compile(code : String) -> (String, Int, List[Instruction]) {
    let tokens = tokenize(code)
    let code = try tokens.parse_sc!() catch {
      ParseError(s) => abort(s)
    } else {
      expr => expr
    }
    let code = code.compileSC()
    return code
  }

  let codes = codes.map(parse_then_compile) + prelude_defs.map(ScDef::compileSC)
  let codes = compiled_primitives + codes
  let (heap, globals) = build_initial_heap(codes)
  let initialState : GState = {
    heap,
    stack: @list.empty(),
    code: @list.of([PushGlobal("main"), Eval]),
    globals,
    stats: 0,
    dump: @list.empty(),
  }
  GState::reify(initialState)
}

test "basic eval" {
  let main = "(defn main[] (let ([add1 (add 1)]) (add1 1)))"
  inspect!(run(@list.of([main])), content="NNum(2)")
  let main = "(defn main[] (let ([x 4] [y 5]) (sub x y)))"
  inspect!(run(@list.of([main])), content="NNum(-1)")
}
```

## Conclusion

In the next part, we will improve the code generation for primitives and add support for data structures.

## Appendix

```moonbit
enum Token {
  DefFn
  Let
  NIL
  CONS
  Case
  Letrec
  Open(Char) // { [ (
  Close(Char) // } ] )
  Id(String)
  Number(Int)
  EOF
} derive(Eq, Show)

fn between(this : Char, lw : Char, up : Char) -> Bool {
  this >= lw && this <= up
}

fn isDigit(this : Char) -> Bool {
  between(this, '0', '9')
}

fn isAlpha(this : Char) -> Bool {
  between(this, 'A', 'Z') || between(this, 'a', 'z')
}

fn isIdChar(this : Char) -> Bool {
  isAlpha(this) || isDigit(this) || this == '_' || this == '-'
}

fn isWhiteSpace(this : Char) -> Bool {
  this == ' ' || this == '\t' || this == '\n'
}

fn to_number(this : Char) -> Int {
  this.to_int() - 48
}

fn isOpen(this : Char) -> Bool {
  this == '(' || this == '[' || this == '{'
}

fn isClose(this : Char) -> Bool {
  this == ')' || this == ']' || this == '}'
}

struct Tokens {
  tokens : Array[Token]
  mut current : Int
} derive(Show)

fn Tokens::new(tokens : Array[Token]) -> Tokens {
  Tokens::{ tokens, current: 0 }
}

fn Tokens::peek(self : Tokens) -> Token {
  if self.current < self.tokens.length() {
    return self.tokens[self.current]
  } else {
    return EOF
  }
}

type! ParseError String

fn Tokens::next(self : Tokens, loc~ : SourceLoc = _) -> Unit {
  self.current = self.current + 1
  if self.current > self.tokens.length() {
    abort("Tokens::next(): \{loc}")
  }
}

fn Tokens::eat(self : Tokens, tok : Token, loc~ : SourceLoc = _) -> Unit!ParseError {
  let __tok = self.peek()
  // assert tok_ != EOF
  if __tok != tok {
    raise ParseError("\{loc} - Tokens::eat(): expect \{tok} but got \{__tok}")
  } else {
    self.next()
  }
}

fn tokenize(source : String) -> Tokens {
  let tokens : Array[Token] = Array::new(capacity=source.length() / 2)
  let mut current = 0
  let source = source.to_array()
  fn peek() -> Char {
    source[current]
  }

  fn next() -> Unit {
    current = current + 1
  }

  while current < source.length() {
    let ch = peek()
    if isWhiteSpace(ch) {
      next()
      continue
    } else if isDigit(ch) {
      let mut num = to_number(ch)
      next()
      while current < source.length() && isDigit(peek()) {
        num = num * 10 + to_number(peek())
        next()
      }
      tokens.push(Number(num))
      continue
    } else if isOpen(ch) {
      next()
      tokens.push(Open(ch))
      continue
    } else if isClose(ch) {
      next()
      tokens.push(Close(ch))
      continue
    } else if isAlpha(ch) {
      let identifier = @buffer.new(size_hint=42)
      identifier.write_char(ch)
      next()
      while current < source.length() && isIdChar(peek()) {
        identifier.write_char(peek())
        next()
      }
      let identifier = identifier.contents().to_unchecked_string()
      match identifier {
        "let" => tokens.push(Let)
        "letrec" => tokens.push(Letrec)
        "Nil" => tokens.push(NIL)
        "Cons" => tokens.push(CONS)
        "case" => tokens.push(Case)
        "defn" => tokens.push(DefFn)
        _ => tokens.push(Id(identifier))
      }
    } else {
      abort("error : invalid Character '\{ch}' in [\{current}]")
    }
  } else {
    return Tokens::new(tokens)
  }
}

test "tokenize" {
  inspect!(tokenize("").tokens, content="[]")
  inspect!(tokenize("12345678").tokens, content="[Number(12345678)]")
  inspect!(tokenize("1234 5678").tokens, content="[Number(1234), Number(5678)]")
  inspect!(
    tokenize("a0 a_0 a-0").tokens,
    content=
      #|[Id("a0"), Id("a_0"), Id("a-0")]
    ,
  )
  inspect!(
    tokenize("(Cons 0 (Cons 1 Nil))").tokens,
    content="[Open('('), CONS, Number(0), Open('('), CONS, Number(1), NIL, Close(')'), Close(')')]",
  )
}

fn Tokens::parse_num(self : Tokens) -> Int!ParseError {
  match self.peek() {
    Number(n) => {
      self.next()
      return n
    }
    other => raise ParseError("parse_num(): expect a number but got \{other}")
  }
}

fn Tokens::parse_var(self : Tokens) -> String!ParseError {
  match self.peek() {
    Id(s) => {
      self.next()
      return s
    }
    other => raise ParseError("parse_var(): expect a variable but got \{other}")
  }
}

fn Tokens::parse_cons(self : Tokens) -> RawExpr[String]!ParseError {
  match self.peek() {
    CONS => {
      self.next()
      let x = self.parse_expr!()
      let xs = self.parse_expr!()
      return App(App(Constructor(tag=1, arity=2), x), xs)
    }
    other => raise ParseError("parse_cons(): expect Cons but got \{other}")
  }
}

fn Tokens::parse_let(self : Tokens) -> RawExpr[String]!ParseError {
  self.eat!(Let)
  self.eat!(Open('('))
  let defs = self.parse_defs!()
  self.eat!(Close(')'))
  let exp = self.parse_expr!()
  Let(false, defs, exp)
}

fn Tokens::parse_letrec(self : Tokens) -> RawExpr[String]!ParseError {
  self.eat!(Letrec)
  self.eat!(Open('('))
  let defs = self.parse_defs!()
  self.eat!(Close(')'))
  let exp = self.parse_expr!()
  Let(true, defs, exp)
}

fn Tokens::parse_case(self : Tokens) -> RawExpr[String]!ParseError {
  self.eat!(Case)
  let exp = self.parse_expr!()
  let alts = self.parse_alts!()
  Case(exp, alts)
}

fn parse_alts(
  self : Tokens
) -> List[(Int, List[String], RawExpr[String])]!ParseError {
  let acc : List[(Int, List[String], RawExpr[String])] = @list.empty()
  loop self.peek(), acc {
    Open('['), acc => {
      self.next()
      self.eat!(Open('('))
      let (tag, variables) = match self.peek() {
        NIL => {
          self.next()
          (0, @list.empty())
        }
        CONS => {
          self.next()
          let x = self.parse_var!()
          let xs = self.parse_var!()
          (1, @list.of([x, xs]))
        }
        other =>
          raise ParseError("parse_alts(): expect NIL or CONS but got \{other}")
      }
      self.eat!(Close(')'))
      let exp = self.parse_expr!()
      let alt = (tag, variables, exp)
      self.eat!(Close(']'))
      continue self.peek(), acc.prepend(alt)
    }
    _, acc => acc.rev()
  }
}

fn Tokens::parse_defs(self : Tokens) -> List[(String, RawExpr[String])]!ParseError {
  let acc : List[(String, RawExpr[String])] = @list.empty()
  loop self.peek(), acc {
    Open('['), acc => {
      self.next()
      let var = self.parse_var!()
      let value = self.parse_expr!()
      self.eat!(Close(']'))
      continue self.peek(), acc.prepend((var, value))
    }
    _, acc => acc.rev()
  }
}

fn Tokens::parse_apply(self : Tokens) -> RawExpr[String]!ParseError {
  let mut res = self.parse_expr!()
  while self.peek() != Close(')') {
    res = App(res, self.parse_expr!())
  }
  return res
}

fn Tokens::parse_expr(self : Tokens) -> RawExpr[String]!ParseError {
  match self.peek() {
    EOF =>
      raise ParseError(
        "parse_expr() : expect a token but got a empty token stream",
      )
    Number(n) => {
      self.next()
      Num(n)
    }
    Id(s) => {
      self.next()
      Var(s)
    }
    NIL => {
      self.next()
      Constructor(tag=0, arity=0)
    }
    Open('(') => {
      self.next()
      let exp = match self.peek() {
        Let => self.parse_let!()
        Letrec => self.parse_letrec!()
        Case => self.parse_case!()
        CONS => self.parse_cons!()
        Id(_) | Open('(') => self.parse_apply!()
        other =>
          raise ParseError("parse_expr(): cant parse \{other} behind a '('")
      }
      self.eat!(Close(')'))
      return exp
    }
    other => raise ParseError("parse_expr(): cant parse \{other}")
  }
}

fn Tokens::parse_sc(self : Tokens) -> ScDef[String]!ParseError {
  self.eat!(Open('('))
  self.eat!(DefFn)
  let fn_name = self.parse_var!()
  self.eat!(Open('['))
  let args = loop self.peek(), @list.empty() {
    tok, acc =>
      if tok != Close(']') {
        let var = self.parse_var!()
        continue self.peek(), acc.prepend(var)
      } else {
        acc.rev()
      }
  }
  self.eat!(Close(']'))
  let body = self.parse_expr!()
  self.eat!(Close(')'))
  ScDef::{ name: fn_name, args, body }
}

test "parse scdef" {
  let test_ = fn!(s) { ignore(tokenize(s).parse_sc!()) }
  for p in programs {
    let (_, p) = p
    test_!(p)
  }
}

let programs : @hashmap.T[String, String] = {
  let programs = @hashmap.new(capacity=40)
  programs["square"] =
    #| (defn square[x] (mul x x))
  programs["fix"] =
    #| (defn fix[f] (letrec ([x (f x)]) x))
  programs["isNil"] =
    #| (defn isNil[x]
    #|   (case x [(Nil) 1] [(Cons n m) 0]))
  programs["tail"] =
    #| (defn tail[l] (case l [(Cons x xs) xs]))
  programs["fibs"] =
    // fibs = 0 : 1 : zipWith (+) fibs (tail fibs)
    #| (defn fibs[] (Cons 0 (Cons 1 (zipWith add fibs (tail fibs)))))
  programs["take"] =
    #| (defn take[n l]
    #|   (case l
    #|     [(Nil) Nil]
    #|     [(Cons x xs)
    #|        (if (le n 0) Nil (Cons x (take (sub n 1) xs)))]))
  programs["zipWith"] =
    #| (defn zipWith[op l1 l2]
    #|   (case l1
    #|     [(Nil) Nil]
    #|     [(Cons x xs)
    #|       (case l2
    #|         [(Nil) Nil]
    #|         [(Cons y ys) (Cons (op x y) (zipWith op xs ys))])]))
  programs["factorial"] =
    #| (defn factorial[n]
    #|   (if (eq n 0) 1 (mul n (factorial (sub n 1)))))
  programs["abs"] =
    #| (defn abs[n]
    #|   (if (lt n 0) (negate n) n))
  programs["length"] =
    #| (defn length[l]
    #|   (case l
    #|     [(Nil) 0]
    #|     [(Cons x xs) (add 1 (length xs))]))
  programs
}

typealias @list.T as List

enum RawExpr[T] {
  Var(T)
  Num(Int)
  Constructor(tag~ : Int, arity~ : Int) // tag, arity
  App(RawExpr[T], RawExpr[T])
  Let(Bool, List[(T, RawExpr[T])], RawExpr[T]) // isRec, Defs, Body
  Case(RawExpr[T], List[(Int, List[T], RawExpr[T])])
} derive(Show)

struct ScDef[T] {
  name : String
  args : List[T]
  body : RawExpr[T]
} derive(Show)

fn[T] is_atom(self : RawExpr[T]) -> Bool {
  match self {
    Var(_) => true
    Num(_) => true
    _ => false
  }
}

fn[T] ScDef::new(name : String, args : List[T], body : RawExpr[T]) -> ScDef[T] {
  { name, args, body }
}

let prelude_defs : List[ScDef[String]] = {
  let args : (FixedArray[String]) -> List[String] = @list.of
  let id = ScDef::new("I", args(["x"]), Var("x")) // id x = x
  let k = ScDef::new("K", args(["x", "y"]), Var("x")) // K x y = x
  let k1 = ScDef::new("K1", args(["x", "y"]), Var("y")) // K1 x y = y
  let s = ScDef::new(
    "S",
    args(["f", "g", "x"]),
    App(App(Var("f"), Var("x")), App(Var("g"), Var("x"))),
  ) // S f g x = f x (g x)
  let compose = ScDef::new(
    "compose",
    args(["f", "g", "x"]),
    App(Var("f"), App(Var("g"), Var("x"))),
  ) // compose f g x = f (g x)
  let twice = ScDef::new(
    "twice",
    args(["f"]),
    App(App(Var("compose"), Var("f")), Var("f")),
  ) // twice f = compose f f
  @list.of([id, k, k1, s, compose, twice])
}

// Use the 'type' keyword to encapsulate an address type.
type Addr Int derive(Eq, Show)

// Describe graph nodes with an enumeration type.
enum Node {
  NNum(Int)
  // The application node
  NApp(Addr, Addr)
  // To store the number of parameters and 
  // the corresponding sequence of instructions for a super combinator
  NGlobal(String, Int, List[Instruction])
  // The Indirection node. The key component of implementing lazy evaluation
  NInd(Addr)
} derive(Eq, Show)

struct GHeap {
  // The heap uses an array, 
  // and the space with None content in the array is available as free memory.
  mut object_count : Int
  memory : Array[Node?]
}

// Allocate heap space for nodes.
fn GHeap::alloc(self : GHeap, node : Node) -> Addr {
  let heap = self
  fn next(n : Int) -> Int {
    (n + 1) % heap.memory.length()
  }

  fn free(i : Int) -> Bool {
    match heap.memory[i] {
      None => true
      _ => false
    }
  }

  let mut i = heap.object_count
  while not(free(i)) {
    i = next(i)
  }
  heap.memory[i] = Some(node)
  heap.object_count = heap.object_count + 1
  return Addr(i)
}

fn GHeap::op_get(self : GHeap, key : Addr) -> Node {
  let Addr(i) = key
  match self.memory[i] {
    Some(node) => node
    None => abort("GHeap::get(): index \{i} was empty")
  }
}

fn GHeap::op_set(self : GHeap, key : Addr, val : Node) -> Unit {
  self.memory[key._] = Some(val)
}

struct GState {
  mut stack : List[Addr]
  heap : GHeap
  globals : @hashmap.T[String, Addr]
  mut dump : List[(List[Instruction], List[Addr])]
  mut code : List[Instruction]
  mut stats : GStats
}

type GStats Int

fn GState::stat_incr(self : GState) -> Unit {
  self.stats = self.stats._ + 1
}

fn GState::put_stack(self : GState, addr : Addr) -> Unit {
  self.stack = self.stack.prepend(addr)
}

fn GState::put_dump(
  self : GState,
  codes : List[Instruction],
  stack : List[Addr]
) -> Unit {
  self.dump = self.dump.prepend((codes, stack))
}

fn GState::put_code(self : GState, instrs : List[Instruction]) -> Unit {
  self.code = instrs + self.code
}

fn GState::pop1(self : GState) -> Addr {
  match self.stack {
    More(addr, tail=reststack) => {
      self.stack = reststack
      addr
    }
    Empty => abort("pop1(): stack size smaller than 1")
  }
}

// e1 e2 ..... -> (e1, e2) ......
fn GState::pop2(self : GState) -> (Addr, Addr) {
  match self.stack {
    More(addr1, tail=More(addr2, tail=reststack)) => {
      self.stack = reststack
      (addr1, addr2)
    }
    _ => abort("pop2(): stack size smaller than 2")
  }
}

fn GState::step(self : GState) -> Bool {
  match self.code {
    Empty => return false
    More(i, tail=rest) => {
      self.code = rest
      self.stat_incr()
      match i {
        PushGlobal(f) => self.push_global(f)
        PushInt(n) => self.push_int(n)
        Push(n) => self.push(n)
        MkApp => self.mk_apply()
        Unwind => self.unwind()
        Update(n) => self.update(n)
        Pop(n) => self.stack = self.stack.drop(n)
        Alloc(n) => self.alloc_nodes(n)
        Eval => self.eval()
        Slide(n) => self.slide(n)
        Add => self.lift_arith2(fn(x, y) { x + y })
        Sub => self.lift_arith2(fn(x, y) { x - y })
        Mul => self.lift_arith2(fn(x, y) { x * y })
        Div => self.lift_arith2(fn(x, y) { x / y })
        Neg => self.negate()
        Eq => self.lift_cmp2(fn(x, y) { x == y })
        Ne => self.lift_cmp2(fn(x, y) { x != y })
        Lt => self.lift_cmp2(fn(x, y) { x < y })
        Le => self.lift_cmp2(fn(x, y) { x <= y })
        Gt => self.lift_cmp2(fn(x, y) { x > y })
        Ge => self.lift_cmp2(fn(x, y) { x >= y })
        Cond(i1, i2) => self.condition(i1, i2)
      }
      return true
    }
  }
}

fn GState::reify(self : GState) -> Node {
  if self.step() {
    self.reify()
  } else {
    let stack = self.stack
    match stack {
      More(addr, tail=Empty) => {
        let res = self.heap[addr]
        return res
      }
      _ => abort("wrong stack \{stack}")
    }
  }
}

fn GState::push_int(self : GState, num : Int) -> Unit {
  let addr = self.heap.alloc(NNum(num))
  self.put_stack(addr)
}

fn GState::update(self : GState, n : Int) -> Unit {
  let addr = self.pop1()
  let dst = self.stack.unsafe_nth(n)
  self.heap[dst] = NInd(addr)
}

fn GState::push_global(self : GState, name : String) -> Unit {
  guard self.globals.get(name) is Some(addr) else {
    abort("push_global(): cant find supercombinator \{name}")
  }
  self.put_stack(addr)
}

fn GState::mk_apply(self : GState) -> Unit {
  let (a1, a2) = self.pop2()
  let appaddr = self.heap.alloc(NApp(a1, a2))
  self.put_stack(appaddr)
}

fn build_initial_heap(
  scdefs : List[(String, Int, List[Instruction])]
) -> (GHeap, @hashmap.T[String, Addr]) {
  let heap = { object_count: 0, memory: Array::make(10000, None) }
  let globals = @hashmap.new(capacity=50)
  loop scdefs {
    Empty => ()
    More((name, arity, instrs), tail=rest) => {
      let addr = heap.alloc(NGlobal(name, arity, instrs))
      globals[name] = addr
      continue rest
    }
  }
  return (heap, globals)
}

enum Instruction {
  Unwind
  PushGlobal(String)
  PushInt(Int)
  Push(Int)
  MkApp
  Slide(Int)
  Update(Int)
  Pop(Int)
  Alloc(Int)
  Eval
  Add
  Sub
  Mul
  Div
  Neg
  Eq // ==
  Ne // !=
  Lt // <
  Le // <=
  Gt // >
  Ge // >=
  Cond(List[Instruction], List[Instruction])
} derive(Eq, Show)

fn argOffset(n : Int, env : List[(String, Int)]) -> List[(String, Int)] {
  env.map(fn { (name, offset) => (name, offset + n) })
}

fn ScDef::compileSC(self : ScDef[String]) -> (String, Int, List[Instruction]) {
  let name = self.name
  let body = self.body
  let mut arity = 0
  fn gen_env(i : Int, args : List[String]) -> List[(String, Int)] {
    match args {
      Empty => {
        arity = i
        return @list.empty()
      }
      More(s, tail=ss) => gen_env(i + 1, ss).prepend((s, i))
    }
  }

  let env = gen_env(0, self.args)
  (name, arity, body.compileR(env, arity))
}

fn RawExpr::compileR(
  self : RawExpr[String],
  env : List[(String, Int)],
  arity : Int
) -> List[Instruction] {
  if arity == 0 {
    self.compileC(env) + @list.of([Update(arity), Unwind])
  } else {
    self.compileC(env) + @list.of([Update(arity), Pop(arity), Unwind])
  }
}

fn RawExpr::compileC(
  self : RawExpr[String],
  env : List[(String, Int)]
) -> List[Instruction] {
  match self {
    Var(s) =>
      match env.lookup(s) {
        None => @list.of([PushGlobal(s)])
        Some(n) => @list.of([Push(n)])
      }
    Num(n) => @list.of([PushInt(n)])
    App(e1, e2) =>
      e2.compileC(env) +
      e1.compileC(argOffset(1, env)) +
      @list.of([MkApp])
    Let(rec, defs, e) =>
      if rec {
        compileLetrec(RawExpr::compileC, defs, e, env)
      } else {
        compileLet(RawExpr::compileC, defs, e, env)
      }
    _ => abort("not support yet")
  }
}
```