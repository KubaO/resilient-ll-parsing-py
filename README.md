# Resilient LL Parsing

A Python port of the source code for the [_Resilient LL Parsing Tutorial_][1] article by Alex Kladov.

  [1]: https://matklad.github.io/2023/05/21/resilient-ll-parsing-tutorial.html
  

---
**Original text by Alex Kladov. Here, I only ported it from Rust to Python.**

# Resilient LL Parsing Tutorial

In this tutorial, I will explain a particular approach to parsing, which gracefully handles syntax errors and is thus suitable for language servers, which, by their nature, have to handle incomplete and invalid code.
Explaining the problem and the solution requires somewhat less than a trivial worked example, and I want to share a couple of tricks not directly related to resilience, so the tutorial builds a full, self-contained parser, instead of explaining abstractly _just_ the resilience.

The tutorial is descriptive, rather than prescriptive --- it tells you what you _can_ do, not what you _should_ do.

- If you are looking into building a production grade language server, treat it as a library of ideas, not as a blueprint.
- If you want to get something working quickly, I think today the best answer is "just use [Tree-sitter](https://tree-sitter.github.io)", so you'd better read its docs rather than this tutorial.
- If you are building an IDE-grade parser from scratch, then techniques presented here might be directly applicable.

## Why Resilience is Needed?

Let's look at one motivational example for resilient parsing:

```rust
fn fib_rec(f1: u32,

fn fib(n: u32) -> u32 {
  fib_rec(1, 1, n)
}
```

Here, a user is in the process of defining the `fib_rec` helper function.
For a language server, it's important that the incompleteness doesn't get in the way.
In particular:

- The following function, `fib`, should be parsed without any errors such that syntax and semantic highlighting is not disturbed, and all calls to `fib` elsewhere typecheck correctly.

- The `fib_rec` function itself should be recognized as a partially complete function, so that various language server assists can help complete it correctly.

- In particular, a smart language server can actually infer the expected type of `fib_rec` from a call we already have, and suggest completing the whole prototype.
  rust-analyzer doesn't do that today, but one day it should.

Generalizing this example, what we want from our parser is to recognize as much of the syntactic structure as feasible.
It should be able to localize errors --- a mistake in a function generally should not interfere with parsing unrelated functions.
As the code is read and written left-to-right, the parser should also recognize valid partial prefixes of various syntactic constructs.

Academic literature suggests another lens to use when looking at this problem: error recovery.
Rather than just recognizing incomplete constructs, the parser can attempt to guess a minimal edit which completes the construct and gets rid of the syntax error.
From this angle, the above example would look rather like `fn fib_rec(f1: u32, /* ) {} */`, where the stuff in a comment is automatically inserted by the parser.

Resilience is a more fruitful framing to use for a language server --- incomplete code is the ground truth, and only the user knows how to correctly complete it.
A language server can only offer guesses and suggestions, and they are more precise if they employ post-parsing semantic information.

Error recovery might work better when emitting understandable syntax errors, but, in a language server, the importance of clear error messages for _syntax_ errors is relatively lower, as highlighting such errors right in the editor synchronously with typing usually provides tighter, more useful tacit feedback.

## Approaches to Error Resilience

The classic approach for handling parser errors is to explicitly encode error productions and synchronization tokens into the language grammar.
This approach isn't a natural fit for resilience framing --- you don't want to anticipate every possible error, as there are just too many possibilities.
Rather, you want to recover as much of a valid syntax tree as possible, and more or less ignore arbitrary invalid parts.

Tree-sitter does something more interesting.
It is a **G**LR parser, meaning that it non-deterministically tries many possible LR (bottom-up) parses, and looks for the best one.
This allows Tree-sitter to recognize many complete valid small fragments of a tree, but it might have trouble assembling them into incomplete larger fragments.
In our example `fn fib_rec(f1: u32,` Tree-sitter correctly recognizes `f1: u32` as a formal parameter, but doesn't recognize `fib_rec` as a function.

Top-down (LL) parsing paradigm makes it harder to recognize valid small fragments, but naturally allows for incomplete large nodes.
Because code is written top-down and left-to-right, LL seems to have an advantage for typical patterns of incomplete code.
Moreover, there isn't really anything special you need to do to make LL parsing resilient.
You sort of... just not crash on the first error, and everything else more or less just works.

Details are fiddly though, so, in the rest of the post, we will write a complete implementation of a handwritten recursive descent + Pratt resilient parser.

## Introducing L

For the lack of imagination on my side, the toy language we will be parsing is called `L`.
It is a subset of Rust, which has just enough features to make some syntax mistakes.
Here's Fibonacci:

```rust
fn fib(n: u32) -> u32 {
    let f1 = fib(n - 1);
    let f2 = fib(n - 2);
    return f1 + f2;
}
```

Note that there's no base case, because L doesn't have syntax for `if`.
Here's the syntax it does have, as an [ungrammar](https://rust-analyzer.github.io/blog/2020/10/24/introducing-ungrammar.html):

```ungrammar
File = Fn*

Fn = 'fn' 'name' ParamList ('->' TypeExpr)? Block

ParamList = '(' Param* ')'
Param = 'name' ':' TypeExpr ','?

TypeExpr = 'name'

Block = '{' Stmt* '}'

Stmt =
  StmtExpr
| StmtLet
| StmtReturn

StmtExpr = Expr ';'
StmtLet = 'let' 'name' '=' Expr ';'
StmtReturn = 'return' Expr ';'

Expr =
  ExprLiteral
| ExprName
| ExprParen
| ExprBinary
| ExprCall

ExprLiteral = 'int' | 'true' | 'false'
ExprName = 'name'
ExprParen = '(' Expr ')'
ExprBinary = Expr ('+' | '-' | '*' | '/') Expr
ExprCall = Expr ArgList

ArgList = '(' Arg* ')'
Arg = Expr ','?
```

The meta syntax here is similar to BNF, with two important differences:

- the notation is better specified and more familiar (recursive regular expressions),
- it describes syntax _trees_, rather than strings (_sequences_ of tokens).

Single quotes signify terminals: `'fn'` and `'return'` are keywords, `'name'` stands for any identifier token, like `foo`, and `'('` is punctuation.
Unquoted names are non-terminals. For example, `x: i32,` would be an example of `Param`.
Unquoted punctuation are meta symbols of ungrammar itself, semantics identical to regular expressions. Zero or more repetition is `*`, zero or one is `?`, `|` is alternation and `()` are used for grouping.

The grammar doesn't nail the syntax precisely. For example, the rule for `Param`, `Param = 'name' ':' Type ','?`, says that `Param` syntax node has an optional comma, but there's nothing in the above `ungrammar` specifying whether the trailing commas are allowed.

Overall, `L` has very little to it --- a program is a series of function declarations, each function has a body which is a sequence of statements, the set of expressions is spartan, not even an `if`. Still, it'll take us some time to parse all that.

## Designing the Tree

A traditional AST for L might look roughly like this:

```python
class File:
  functions: list[Function]

class Function:
  name: str
  params: list[Param]
  return_type: Optional[TypeExpr]
  block: Block
```

Extending this structure to be resilient is non-trivial. There are two problems: trivia and errors.

For resilient parsing, we want the AST to contain every detail about the source text.
We actually don't want to use an _abstract_ syntax tree, and need a _concrete_ one.
In a traditional AST, the tree structure is rigidly defined --- any syntax node has a fixed number of children.
But there can be any number of comments and whitespace anywhere in the tree, and making space for them in the structure requires some fiddly data manipulation.
Similarly, errors (e.g., unexpected tokens), can appear anywhere in the tree.

One trick to handle these in the AST paradigm is to attach trivia and error tokens to other tokens.
That is, for something like
`fn /* name of the function -> */ f() {}`,
the `fn` and `f` tokens would be explicit parts of the AST, while the comment and surrounding whitespace would belong to the collection of trivia tokens hanging off the `fn` token.

One complication here is that it's not always just tokens that can appear anywhere, sometimes you can have full trees like that.
For example, comments might support Markdown syntax, and you might actually want to parse that properly (e.g., to resolve links to declarations).
Syntax errors can also span whole subtrees.
For example, when parsing `pub(crate) nope` in Rust, it would be smart to parse `pub(crate)` as a visibility modifier, and nest it into a bigger `Error` node.

SwiftSyntax meticulously adds error placeholders between any two fields of an AST node, giving rise to
`unexpectedBetweenModifiersAndDeinitKeyword`
and such ([source](https://github.com/apple/swift-syntax/blob/66450960b1ed88b842d63f7a38254aaba08bbd4d/Sources/SwiftSyntax/generated/syntaxNodes/SyntaxDeclNodes.swift#L1368), [docs](https://swiftpackageindex.com/apple/swift-syntax/508.0.1/documentation/swiftsyntax/classdeclsyntax#instance-properties)).

An alternative approach, used by IntelliJ, rust-analyzer, and the Lark parser generator, is to treat the syntax tree as a somewhat dynamically-typed data structure:

```python
class TokenKind(Enum):
  ErrorToken = 1
  LParen = 2
  RParen = 3
  Eq = 4
  ...

class Token(NamedTuple):
  kind: TokenKind
  text: str

class TreeKind(Enum):
  ErrorTree = 1
  File = 2
  Fn = 3
  Param = 4
  ...

@dataclass
class Tree:
  kind: TreeKind
  children: list[Child] = field(default_factory=list)

type Child = Token | Tree
```

This structure does not enforce any constraints on the shape of the syntax tree at all, and so it naturally accommodates errors anywhere.
It is possible to layer a well-typed API on top of this dynamic foundation.
An extra benefit of this representation is that you can use the same tree _type_ for different languages; this is a requirement for universal tools.

Discussing specifics of syntax tree representation goes beyond this article, as the topic is vast and lacks a clear winning solution.
To learn about it, take a look at Roslyn, SwiftSyntax, rowan, IntelliJ and Lark.

To simplify things, we'll ignore comments and whitespace, though you'll absolutely want those in a real implementation.
One approach would be to do the parsing without comments, like we do here, and then attach comments to the nodes in a separate pass.
Attaching comments needs some heuristics --- for example, non-doc comments generally want to be a part of the following syntax node.

Another design choice is handling of error messages.
One approach is to treat error messages as properties of the syntax tree itself, by either inferring them from the tree structure, or just storing them inline.
Alternatively, errors can be considered to be a side effect of the parsing process (that way, trees constructed manually during, eg, refactors, won't carry any error messages, even if they are invalid).

Here's the full set of token and tree kinds for our language L:

```python
TokenKind = Enum('TokenKind', """
  ErrorToken  Eof

  LParen  RParen  LCurly  RCurly
  Eq  Semi  Comma  Colon  Arrow
  Plus  Minus  Star  Slash

  FnKeyword  LetKeyword  ReturnKeyword
  TrueKeyword  FalseKeyword

  Name  Int
""")

TreeKind = Enum('TreeKind', """
  ErrorTree
  File  Fn  TypeExpr
  ParamList  Param
  Block
  StmtLet  StmtReturn  StmtExpr
  ExprLiteral  ExprName  ExprParen,
  ExprBinary  ExprCall
  ArgList  Arg
""")
```

Things to note:

- explicit `Error` kinds;
- no whitespace or comments, as an unrealistic simplification;
- `Eof` virtual token simplifies parsing, removing the need to handle `Optional[Token]`;
- punctuators are named after what they are, rather than after what they usually mean: `Star`, rather than `Mult`;
- a good set of name for various kinds of braces is `{L,R}{Paren,Curly,Brack,Angle}`.

## Lexer

Won't be covering lexer here, let's just say we have `def lex(text: str) -> Iterable[Token>]: ...` function. Two points worth mentioning:

- Lexer itself should be resilient, but that's easy --- produce an `Error` token for anything which isn't a valid token.
- Writing lexer by hand is somewhat tedious, but is very simple relative to everything else.
  If you are stuck in an analysis-paralysis picking a lexer generator, consider cutting the Gordian knot and hand-writing.

## Parser

With homogenous syntax trees, the task of parsing admits an elegant formalization --- we want to insert extra parenthesis into a stream of tokens.

```
+-Fun
|      +-Param
|      |
[fn f( [x: Int] ) {}]
     |            |
     |            +-Block
     +-ParamList

```

Note how the sequence of tokens with extra parenthesis is still a flat sequence.
The parsing will be two-phase:

- in the first phase, the parser emits a flat list of events,
- in the second phase, the list is converted to a tree.

Here's the basic setup for the parser:

```python
type Event = OpenEvent | CloseEvent | AdvanceEvent

@dataclass
class OpenEvent:
  kind: TreeKind #❷
  
class CloseEvent:
    pass

class AdvanceEvent:
    pass

class MarkOpened(NamedTuple):
  index: int

@dataclass
class Parser:
  tokens: list[Token]
  pos: int = 0
  fuel: int = 256  #❹
  events: list[Event] = field(default_factory=list)

  def open(self) -> MarkOpened: #❶
    mark = MarkOpened(index=len(self.events))
    self.events.append(EventOpen(kind=TreeKind.ErrorTree))
    return mark
  
  def close(self, m: MarkOpened, kind: TreeKind): #❶❷
    self.events[m.index] = EventOpen(kind=kind)
    self.events.append(EventClose())

  def advance(self): #❶
    assert not self.eof()
    self.fuel = 256  #❹
    self.events.append(EventAdvance())
    self.pos += 1

  def eof(self) -> bool:
    return self.pos == len(self.tokens)
  
  def nth(self, lookahead: int) -> TokenKind: #❸
    if self.fuel == 0: #❹
      panic("parser is stuck")
    self.fuel -= 1
    i = self.pos + lookahead
    return self.tokens[i].kind if i < len(self.tokens) else TokenKind.Eof

  def at(self, kind: TokenKind) -> bool: #❸
    return self.nth(0) == kind

  def eat(self, kind: TokenKind) -> bool: #❸
    if self.at(kind):
      self.advance()
      return True
    else:
      return False
      
  def expect(self, kind: TokenKind):
    if self.eat(kind):
      return
    # TODO: Error reporting.
    print(f"expected {kind}", file=sys.stderr)

  def advance_with_error(self, error: str):
    m = self.open()
    # TODO: Error reporting.
    print(error, file=sys.stderr)
    self.advance()
    self.close(m, TreeKind.ErrorTree)
```

1) `open`, `advance`, and `close` form the basis for constructing the stream of events.

2) Note how `kind` is stored in the `Open` event, but is supplied with the `close` method.
   This is required for flexibility --- sometimes it's possible to decide on the type of syntax node only after it is parsed.
   The way this works is that the `open` method returns a `Mark` which is subsequently passed to `close` to modify the corresponding `Open` event.

3) There's a set of short, convenient methods to navigate through the sequence of tokens:

    - `nth` is the lookahead method. Note how it doesn't return an `Option`, and uses `Eof` special value for "out of bounds" indexes.
      This simplifies the call-site, "no more tokens" and "token of a wrong kind" are always handled the same.
    - `at` is a convenient specialization to check for a specific next token.
    - `eat` is `at` combined with consuming the next token.
    - `expect` is `eat` combined with error reporting.

    These methods are not a very orthogonal basis, but they are a convenience basis for parsing.
    Finally, `advance_with_error` advanced over any token, but also wraps it into an error node.

4) When writing parsers by hand, it's very easy to accidentally write the code which loops or recurses forever.
   To simplify debugging, it's helpful to add an explicit notion of "fuel", which is replenished every time the parser makes progress,
   and is spent every time it does not.

The function to transform a flat list of events into a tree is a bit involved.
It juggles three things: an iterator of events, an iterator of tokens, and a stack of partially constructed nodes (we expect the stack to contain just one node at the end).

```python
class Parser:
  def build_tree(self) -> Tree:
    tokens = iter(self.tokens)
    events = self.events
    stack = []

    # Special case: pop the last `Close` event to ensure
    # that the stack is non-empty inside the loop.
    assert isinstance(events.pop(), EventClose)

    for event in events:
      match event:
        # Starting a new node; just push an empty tree to the stack.
        case EventOpen(kind):
          stack.append(Tree(kind, []))

        # A tree is done.
        # Pop it off the stack and append to a new current tree.
        case EventClose():
          tree = stack.pop()
          stack[-1].children.append(tree)

        # Consume a token and append it to the current tree
        case EventAdvance():
          token = next(tokens)
          stack[-1].children.append(token)

    # Our parser will guarantee that all the trees are closed
    # and cover the entirety of tokens.
    assert len(stack) == 1
    assert next(tokens, None) is None

    stack.pop()
```

## Grammar

We are finally getting to the actual topic of resilient parser.
Now we will write a full grammar for L as a sequence of functions.
Usually both atomic parser operations, like `fn advance`, and grammar productions, like `fn parse_fn` are implemented as methods on the `Parser` struct.
I prefer to separate the two and to use free functions for the latter category, as the code is a bit more readable that way.

Let's start with parsing the top level.

```python
# File = Fn*
def file(p: Parser):
  m = p.open()  #❶

  while not p.eof(): #❷
    if p.at(TokenKind.FnKeyword):
      func(p)
    else:
      p.advance_with_error("expected a function")  #❸

  p.close(m, TreeKind.File)  #❶
```

1) Wrap the whole thing into a `File` node.

2) Use the `while` loop to parse a file as a series of functions.
   Importantly, the entirety of the file is parsed; we break out of the loop only when the eof is reached.

3) To not get stuck in this loop, it's crucial that every iteration consumes at least one token.
   If the token is `fn`, we'll parse at least a part of a function.
   Otherwise, we consume the token and wrap it into an error node.

Lets parse functions now:


```python
# Fn = 'fn' 'name' ParamList ('->' TypeExpr)? Block
def func(p: Parser):
  assert p.at(TokenKind.FnKeyword) #❶ 
  m = p.open()  #❷

  p.expect(TokenKind.FnKeyword)
  p.expect(TokenKind.Name)
  if p.at(TokenKind.LParen):  #❸
    param_list(p)
  if p.eat(TokenKind.Arrow):
    type_expr(p)
  if p.at(TokenKind.LCurly):  #❸
    block(p)

  p.close(m, TreeKind.Fn)  #❷
```

1) When parsing a function, we assert that the current token is `fn`.
   There's some duplication with the `if p.at(FnKeyword)`, check at the call-site, but this duplication actually helps readability.

2) Again, we surround the body of the function with `open`/`close` pair.

3) Although parameter list and function body are mandatory, we precede them with an `at` check.
   We can still report the syntax error by analyzing the structure of the syntax tree (or we can report it as a side effect of parsing in the `else` branch if we want).
   It wouldn't be wrong to just remove the `if` altogether and try to parse `param_list` unconditionally, but the `if` helps with reducing cascading errors.

Now, the list of parameters:

```python
# ParamList = '(' Param* ')'
def param_list(p: Parser):
  assert p.at(TokenKind.LParen)
  m = p.open()

  p.expect(TokenKind.LParen)  #❶ 
  while not p.at(TokenKind.RParen) and not p.eof():  #❷
    if p.at(TokenKind.Name):  #❸
      param(p)
    else:
      break  #❸
  p.expect(TokenKind.RParen)  #❶

  p.close(m, TreeKind.ParamList)
```

1) Inside, we have a standard code shape for parsing a bracketed list.
   It can be extracted into a high-order function, but typing out the code manually is not a problem either.
   This bit of code starts and ends with consuming the corresponding parenthesis.
2) In the happy case, we loop until the closing parenthesis.
   However, it could also be the case that there's no closing parenthesis at all, so we add an `eof` condition as well.
   Generally, every loop we write would have `and not p.eof()` tackled on.
3) As with any loop, we need to ensure that each iteration consumes at least one token to not get stuck.
   If the current token is an identifier, everything is ok, as we'll parse at least some part of the parameter.

Parsing parameter is almost nothing new at this point:

```python
# Param = 'name' ':' TypeExpr ','?
def param(p: Parser):
  assert p.at(TokenKind.Name)
  m = p.open()

  p.expect(TokenKind.Name)
  p.expect(TokenKind.Colon)
  type_expr(p)
  if not p.at(TokenKind.RParen):  #❶
    p.expect(TokenKind.Comma)

  p.close(m, TreeKind.Param)
```

1) This is the only interesting bit.
   To parse a comma-separated list of parameters with a trailing comma, it's enough to check if the following token after parameter is `)`.
   This correctly handles all three cases:

   - if the next token is `)`, we are at the end of the list, and no comma is required;
   - if the next token is `,`, we correctly advance past it;
   - finally, if the next token is anything else, then it's not a `)`, so we are not at the last element of the list and correctly emit an error.

Parsing types is trivial:

```python
# TypeExpr = 'name'
def type_expr(p: Parser):
  m = p.open()
  p.expect(TokenKind.Name)
  p.close(m, TreeKind.TypeExpr)
```

The notable aspect here is naming.
The production is deliberately named `TypeExpr`, rather than `Type`, to avoid confusion down the line.
Consider `fib(92)`.
It is an _expression_, which evaluates to a _value_.
The same thing happens with types.
For example, `Foo<Int>` is not a type yet, it's an expression which can be "evaluated" (at compile time) to a type (if `Foo` is a type alias, the result might be something like `Pair<Int, Int>`).

Parsing a block gets a bit more involved:

```python
# Block = '{' Stmt* '}'
#
# Stmt =
#   StmtLet
# | StmtReturn
# | StmtExpr
def block(p: Parser):
  assert p.at(TokenKind.LCurly)
  m = p.open()

  p.expect(TokenKind.LCurly)
  while not p.at(TokenKind.RCurly) and not p.eof():
    match p.nth(0):
      case TokenKind.LetKeyword: stmt_let(p)
      case TokenKind.ReturnKeyword: stmt_return(p)
      case _: stmt_expr(p)
  p.expect(TokenKind.RCurly)

  p.close(m, TreeKind.Block)
```

Block can contain many different kinds of statements, so we branch on the first token in the loop's body.
As usual, we need to maintain an invariant that the body consumes at least one token.
For `let` and `return` statements that's easy, they consume the fixed first token.
For the expression statement (things like `1 + 1;`) it gets more interesting, as an expression can start with many different tokens.
For the time being, we'll just kick the can down the road and require `stmt_expr` to deal with it (that is, to guarantee that at least one token is consumed).

Statements themselves are straightforward:

```python
# StmtLet = 'let' 'name' '=' Expr ';'
def stmt_let(p: Parser):
  assert p.at(TokenKind.LetKeyword)
  m = p.open()

  p.expect(TokenKind.LetKeyword)
  p.expect(TokenKind.Name)
  p.expect(TokenKind.Eq)
  expr(p)
  p.expect(TokenKind.Semi)

  p.close(m, TreeKind.StmtLet)

# StmtReturn = 'return' Expr ';'
def stmt_return(p: Parser):
  assert p.at(TokenKind.ReturnKeyword)
  m = p.open()

  p.expect(TokenKind.ReturnKeyword)
  expr(p)
  p.expect(TokenKind.Semi)

  p.close(m, TreeKind.StmtReturn)

# StmtExpr = Expr ';'
def stmt_expr(p: Parser):
  m = p.open()

  expr(p)
  p.expect(TokenKind.Semi)

  p.close(m, TreeKind.StmtExpr)
```

Again, for `stmt_expr`, we push "must consume a token" invariant onto `expr`.

Expressions are tricky.
They always are.
For starters, let's handle just the clearly-delimited cases, like literals and parenthesis:


```python
def expr(p: Parser):
  expr_delimited(p)

def expr_delimited(p: Parser):
  m = p.open()
  match p.nth(0):
    # ExprLiteral = 'int' | 'true' | 'false'
    case TokenKind.Int | TokenKind.TrueKeyword | TokenKind.FalseKeyword:
      p.advance()
      p.close(m, TreeKind.ExprLiteral)

    # ExprName = 'name'
    case TokenKind.Name:
      p.advance()
      p.close(m, TreeKind.ExprName)

    # ExprParen   = '(' Expr ')'
    case TokenKind.LParen:
      p.expect(TokenKind.LParen)
      expr(p)
      p.expect(TokenKind.RParen)
      p.close(m, TreeKind.ExprParen)

    case _:
      if not p.eof():
        p.advance()
      p.close(m, TreeKind.ErrorTree)
```

In the catch-all arm, we take care to consume the token, to make sure that the statement loop in `block` can always make progress.

Next expression to handle would be `ExprCall`.
This requires some preparation.
Consider this example: `f(1)(2)`.

We want the following parenthesis structure here:

```
+-ExprCall
|
|   +-ExprName
|   |       +-ArgList
|   |       |
[ [ [f](1) ](2) ]
  |    |
  |    +-ArgList
  |
  +-ExprCall
```

The problem is, when the parser is at `f`, it doesn't yet know how many `Open` events it should emit.

We solve the problem by adding an API to go back and inject a new `Open` event into the middle of existing events.

```python
class MarkOpened(NamedTuple):
  index: int

class MarkClosed(NamedTuple):
  index: int

class Parser:
  # continued
  
  def open(self) -> MarkOpened:
    mark = MarkOpened(index=len(self.events))
    self.events.append(EventOpen(kind=TreeKind.ErrorTree))
    return mark

  def close(self, m: MarkOpened,
            kind: TreeKind) -> MarkClosed:  #❶
    self.events[m.index] = EventOpen(kind=kind)
    self.events.append(EventClose())
    return MarkClosed(index=m.index)

  def open_before(self, m: MarkClosed) -> MarkOpened:  #❷
    mark = MarkOpened(index=m.index)
    self.events.insert(
      m.index,
      EventOpen(kind=TreeKind.ErrorTree)
    )
    return mark
```

1) Here we adjust `close` to also return a `MarkClosed`, such that we can go back and add a new event before it.

2) The new API. It is like `open`, but also takes a `MarkClosed` which carries an index of an `Open` event in front of which we are to inject a new `Open`.
   In the current implementation, for simplicity, we just inject into the middle of the vector, which is an O(N) operation worst-case.
   A proper solution here would be to use an index-based linked list.
   That is, `open_before` can push the new open event to the end of the list, and also mark the old event with a pointer to the freshly inserted one.
   To store a pointer, an extra field is needed:

   ```python
   class EventOpen(NamedTuple):
       kind: TreeKind
       # Points forward into a list at the Open event
       # which logically happens before this one.
       open_before: Optional[int]
   ```

   The loop in `build_tree` needs to follow the `open_before` links.

With this new API, we can parse function calls:

```python
def expr_delimited(p: Parser) -> MarkClosed: #❶
  ...

def expr(p: Parser):
  lhs = expr_delimited(p)  #❶

  # ExprCall = Expr ArgList
  while p.at(TokenKind.LParen): #❷
    m = p.open_before(lhs)
    arg_list(p)
    lhs = p.close(m, TreeKind.ExprCall)

# ArgList = '(' Arg* ')'
def arg_list(p: Parser):
  assert p.at(TokenKind.LParen)
  m = p.open()

  p.expect(TokenKind.LParen)
  while not p.at(TokenKind.RParen) and not p.eof():  #❸
    arg(p)
  p.expect(TokenKind.RParen)

  p.close(m, TreeKind.ArgList)

# Arg = Expr ','?
def arg(p: Parser):
  m = p.open()

  expr(p)
  if not p.at(TokenKind.RParen):  #❹
    p.expect(TokenKind.Comma)

  p.close(m, TreeKind.Arg)
```

1) `expr_delimited` now returns a `MarkClosed` rather than `()`.
    No code changes are required for this, as `close` calls are already in the tail position.

2) To parse function calls, we check whether we are at `(` and use `open_before` API if that is the case.

3) Parsing argument list should be routine by now.
    Again, as an expression can start with many different tokens, we don't add an `if p.at` check to the loop's body, and require `arg` to consume at least one token.

4) Inside `arg`, we use an already familiar construct to parse an optionally trailing comma.

Now only binary expressions are left.
We will use a Pratt parser for those.
This is genuinely tricky code, so I have a dedicated article explaining how it all works:

[_Simple but Powerful Pratt Parsing_](https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html).

Here, I'll just dump a pageful of code without much explanation:

```python
def expr(p: Parser):
  expr_rec(p, Eof)  #❷

def expr_rec(p: Parser, left: TokenKind):  #❶
  lhs = expr_delimited(p)

  while p.at(TokenKind.LParen):
    m = p.open_before(lhs)
    arg_list(p)
    lhs = p.close(m, TreeKind.ExprCall)

  while True:
    right = p.nth(0)
    if right_binds_tighter(left, right):  #❶
      m = p.open_before(lhs)
      p.advance()
      expr_rec(p, right)
      lhs = p.close(m, TreeKind.ExprBinary)
    else:
      break

def right_binds_tighter( #<1>
  left: TokenKind, right: TokenKind,
) -> bool:
  def tightness(kind: TokenKind) -> Optional[int]:
    pt = (
      # Precedence table:
      (TokenKind.Plus, TokenKind.Minus),
      (TokenKind.Star, TokenKind.Slash),
    )
    for i, tokens in enumerate(pt):
      if kind in tokens:
        return i
    return None

  if not (right_tightness := tightness(right)):  #❸
    return False

  if not (left_tightness := tightness(left)):
    assert left == TokenKind.Eof
    return True

  return right_tightness > left_tightness
```

1) In this version of pratt, rather than passing numerical precedence, I pass the actual token (learned that from [jamii's post](https://www.scattered-thoughts.net/writing/better-operator-precedence/)).
    So, to determine whether to break or recur in the Pratt loop, we ask which of the two tokens binds tighter and act accordingly.

2) When we start parsing an expression, we don't have an operator to the left yet, so I just pass `Eof` as a dummy token.

3) The code naturally handles the case when the next token is not an operator (that is, when expression is complete, or when there's some syntax error).

And that's it! We have parsed the entirety of L!

## Basic Resilience

Let's see how resilient our basic parser is.
Let's check our motivational example:

```rust
fn fib_rec(f1: u32,

fn fib(n: u32) -> u32 {
  return fib_rec(1, 1, n);
}
```

Here, the syntax tree our parser produces is surprisingly exactly what we want:

```ungrammar
File
  Fn
    'fn'
    'fib_rec'
    ParamList
      '('
      Param
      'f1'
      ':'
      TypeExpr
        'u32'
      ','
    error: expected RParen

  Fn
    'fn'
    'fib'
    ...
```

For the first incomplete function, we get `Fn`, `Param` and `ParamList`, as we should.
The second function is parsed without any errors.

Curiously, we get this great result without much explicit effort to make parsing resilient, it's a natural outcome of just not failing in the presence of errors.
The following ingredients help us:

- homogeneous syntax tree supports arbitrary malformed code,
- any syntactic construct is parsed left-to-right, and valid prefixes are always recognized,
- our top-level loop in `file` is greedy: it either parses a function, or skips a single token and tries to parse a function again.
  That way, if there's a valid function somewhere, it will be recognized.

Thinking about the last case both reveals the limitations of our current code, and shows avenues for improvement.
In general, parsing works as a series of nested loops:

```python
while True: # parse a list of functions

  while True: # parse a list of statements inside a function

    while True: # parse a list of expressions
      ...
```

If something goes wrong inside a loop, our choices are:

- skip a token, and continue with the next iteration of the current loop,
- break out of the inner loop, and let the outer loop handle recovery.

The top-most loop must use the "skip a token" solution, because it needs to consume all the input tokens.

## Improving Resilience

Right now, each loop either always skips, or always breaks.
This is not optimal.
Consider this example:

```rust
fn f1(x: i32,

fn f2(x: i32,, z: i32) {}

fn f3() {}
```

Here, for `f1` we want to break out of `param_list` loop, and our code does just that.
For `f2` though, the error is a duplicated comma (the user will add a new parameter between `x` and `z` shortly), so we want to skip here.
We don't, and, as a result, the syntax tree for `f2` is a train wreck:

```ungrammar
Fn
  'fn'
  'f2'
  ParamList
    '('
    Param
      'x'
      ':'
      TypeExpr
        'i32'
      ','
ErrorTree
  ','
ErrorTree
  'z'
ErrorTree
  ':'
ErrorTree
  'i32'
ErrorTree
  ')'
ErrorTree
  '{'
ErrorTree
  '}'
```

For parameters, it is reasonable to skip tokens until we see something which implies the end of the parameter list.
For example, if we are parsing a list of parameters and see an `fn` token, then we'd better stop.
If we see some less salient token, it's better to gobble it up.
Let's implement the idea:

```python
PARAM_LIST_RECOVERY = (TokenKind.Arrow, TokenKind.LCurly, TokenKind.FnKeyword)

def param_list(p: Parser):
  assert p.at(LParen)
  m = p.open()

  p.expect(TokenKind.LParen)
  while not p.at(TokenKind.RParen) and not !p.eof():
    if p.at(TokenKind.Name):
      param(p)
    else:
      if p.at_any(PARAM_LIST_RECOVERY):
        break
      p.advance_with_error("expected parameter")

  p.expect(TokenKind.RParen)

  p.close(m, TreeKind.ParamList)
```

Here, we use `at_any` helper function, which is like `at`, but takes a list of tokens.
The real implementation would use bitsets for this purpose.

The example now parses correctly:

```ungrammar
File
  Fn
    'fn'
    'f1'
    ParamList
      '('
      (Param 'x' ':' (TypeExpr 'i32') ',')
      error: expected RParen
  Fn
    'fn'
    'f2'
    ParamList
      '('
      (Param 'x' ':' (TypeExpr 'i32') ',')
      ErrorTree
        error: expected parameter
        ','
      (Param 'z' ':' (TypeExpr 'i32'))
      ')'
    (Block '{' '}')
  Fn
    'fn'
    'f3'
    (ParamList '(' ')')
    (Block '{' '}')
```

What is a reasonable `RECOVERY` set in a general case?
I don't know the answer to this question, but follow{.dfn} sets from formal grammar theory give a good intuition.
We don't want _exactly_ the follow{.dfn} set: for `ParamList`, `{` is in follow{.dfn}, and we do want it to be a part of the recovery set, but `fn` is _not_ in follow{.dfn}, and yet it is important to recover on it.
`fn` is included because it's in the follow{.dfn} for `Fn`, and `ParamList` is a child of `Fn`: we also want to recursively include ancestor follow{.dfn} sets into the recovery set.

For expressions and statements, we have the opposite problem --- `block` and `arg_list` loops eagerly consume erroneous tokens, but sometimes it would be wise to break out of the loop instead.

Consider this example:

```rust
fn f() {
  g(1,
  let x =
}

fn g() {}
```

It gives another train wreck syntax tree, where the `g` function is completely missed:

```ungrammar
File
  Fn
    'fn'
    'f'
    (ParamList '(' ')')
    Block
      '{'
      StmtExpr
        ExprCall
          (ExprName 'g')
          ArgList
            '('
            (Arg (ExprLiteral '1') ',')
            (Arg (ErrorTree 'let'))
            (Arg (ExprName 'x'))
            (Arg (ErrorTree '='))
            (Arg (ErrorTree '}'))
            (Arg (ErrorTree 'fn'))
            Arg
              ExprCall
                (ExprName 'g')
                (ArgList '(' ')')
            (Arg (ErrorTree '{'))
            (Arg (ErrorTree '}'))
```

Recall that the root cause here is that we require `expr` to consume at least one token, because it's not immediately obvious which tokens can start an expression.
It's not immediately obvious, but easy to compute --- that's exactly first{.dfn} set from formal grammars.

Using it, we get:

```python
STMT_RECOVERY = (TokenKind.FnKeyword,)
EXPR_FIRST = (TokenKind.Int, TokenKind.TrueKeyword,
              TokenKind.FalseKeyword, TokenKind.Name,
              TokenKind.LParen)

def block(p: Parser):
  assert p.at(TokenKind.LCurly)
  m = p.open()

  p.expect(TokenKind.LCurly)
  while not p.at(TokenKind.RCurly) and not p.eof():
    match p.nth(0):
      case TokenKind.LetKeyword: stmt_let(p)
      case TokenKind.ReturnKeyword: stmt_return(p)
      case _:
        if p.at_any(EXPR_FIRST):
          stmt_expr(p)
        else:
          if p.at_any(STMT_RECOVERY):
            break
          p.advance_with_error("expected statement")
  p.expect(TokenKind.RCurly)

  p.close(m, TreeKind.Block)

def arg_list(p: Parser):
  assert p.at(TokenKind.LParen)
  m = p.open()

  p.expect(TokenKind.LParen)
  while not p.at(TokenKind.RParen) and not p.eof():
    if p.at_any(EXPR_FIRST):
      arg(p)
    else:
      break
  p.expect(TokenKind.RParen)

  p.close(m, TreeKind.ArgList)
```

This fixes the syntax tree:

```ungrammar
File
  Fn
    'fn'
    'f'
    (ParamList '(' ')')
    Block
      '{'
      StmtExpr
        ExprCall
          (ExprName 'g')
          ArgList
            '('
            (Arg (ExprLiteral '1' ','))
      StmtLet
        'let'
        'x'
        '='
        (ErrorTree '}')
  Fn
    'fn'
    'g'
    (ParamList '(' ')')
    (Block '{' '}')
```

There's only one issue left.
Our `expr` parsing is still greedy, so, in a case like this

```rust
fn f() {
  let x = 1 +
  let y = 2
}
```

the `let` will be consumed as a right-hand-side operand of `+`.
Now that the callers of `expr` contain a check for `EXPR_FIRST`, we no longer need this greediness and can return `None` if no expression can be parsed:

```python
def expr_delimited(p: Parser) -> Optional[MarkClosed]:
  match p.nth(0):
    # ExprLiteral = 'int' | 'true' | 'false'
    case TokenKind.Int | TokenKind.TrueKeyword | TokenKind.FalseKeyword:
      m = p.open()
      p.advance()
      return p.close(m, TreeKind.ExprLiteral)

    # ExprName = 'name'
    case TokenKind.Name:
      m = p.open()
      p.advance()
      return p.close(m, TreeKind.ExprName)

    # ExprParen   = '(' Expr ')'
    case TokenKind.LParen:
      m = p.open()
      p.expect(TokenKind.LParen)
      expr(p)
      p.expect(TokenKind.RParen)
      return p.close(m, TreeKind.ExprParen)

    case _:
      assert not p.at_any(EXPR_FIRST):
      return None

def expr_rec(p: Parser, left: TokenKind):
  if not (lhs := expr_delimited(p)):
    return
  ...
```

This gives the following syntax tree:

```ungrammar
File
  Fn
    'fn'
    'f'
    (ParamList '(' ')')
    Block
      '{'
      StmtLet
        'let'
        'x'
        '='
        (ExprBinary (ExprLiteral '1') '+')
      StmtLet
        'let'
        'y'
        '='
        (ExprLiteral '2')
      '}'
```

And this concludes the tutorial!
You are now capable of implementing an IDE-grade parser for a real programming language from scratch.

Summarizing:

- Resilient parsing means recovering as much syntactic structure from erroneous code as possible.

- Resilient parsing is important for IDEs and language servers, who's job mostly ends when the code does not have errors anymore.

- Resilient parsing is related, but distinct from error recovery and repair.
  Rather than guessing what the user meant to write, the parser tries to make sense of what is actually written.

- Academic literature tends to focus on error repair, and mostly ignores pure resilience.

- The biggest challenge of resilient parsing is the design of a syntax tree data structure.
  It should provide convenient and type-safe access to well-formed syntax trees, while allowing arbitrary malformed trees.

- One possible design here is to make the underlying tree a dynamically-typed data structure (like JSON), and layer typed accessors on top (not covered in this article).

- LL style parsers are a good fit for resilient parsing.
  Because code is written left-to-right, it's important that the parser recognizes well-formed prefixes of incomplete syntactic constructs, and LL does just that.

- Ultimately, parsing works as a stack of nested `for` loops.
  Inside a single `for` loop, on each iteration, we need to decide between:

  - trying to parse a sequence element,
  - skipping over an unexpected token,
  - breaking out of the nested loop and delegating recovery to the parent loop.

- first{.dfn}, follow{.dfn} and recovery sets help making a specific decision.

- In any case, if a loop tries to parse an item, item parsing _must_ consume at least one token (if only to report an error).


Source code for the article is here: <https://github.com/matklad/resilient-ll-parsing/blob/master/src/lib.rs#L44>