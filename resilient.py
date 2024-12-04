from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NamedTuple, Optional


def panic(msg: str):
    print(msg, file=sys.stderr)
    sys.stderr.flush()
    os.abort()


class TokenKind(Enum):
    # Token types recognized by the regexp-based scanner
    Arrow = '->'
    LParen = r'\('
    RParen = r'\)'
    LCurly = '{'
    RCurly = '}'
    Eq = '='
    Semi = ';'
    Comma = ','
    Colon = ':'
    Plus = r'\+'
    Minus = '-'
    Star = r'\*'
    Slash = '/'
    Int = '[0-9]+'
    Name = '[_a-zA-Z0-9]+'

    # Keyword tokens that Name gets specialized to
    FnKeyword = 'fn',
    LetKeyword = 'let',
    ReturnKeyword = 'return',
    TrueKeyword = 'true',
    FalseKeyword = 'false',

    # Special tokens
    Eof = ['$']
    ErrorToken = ['ERROR']


class TreeKind(Enum):
    ErrorTree = auto()
    File = auto()
    Fn = auto()
    TypeExpr = auto()
    ParamList = auto()
    Param = auto()
    Block = auto()
    StmtLet = auto()
    StmtReturn = auto()
    StmtExpr = auto()
    ExprLiteral = auto()
    ExprName = auto()
    ExprParen = auto()
    ExprBinary = auto()
    ExprCall = auto()
    ArgList = auto()
    Arg = auto()


type Child = Token | Tree


class Token(NamedTuple):
    kind: TokenKind
    text: str


class Tree(NamedTuple):
    kind: TreeKind
    children: list[Child]

    def print(self, buf: str = "", level: int = 0) -> str:
        indent = "  " * level
        buf += f"{indent}{self.kind.name}\n"
        for child in self.children:
            match child:
                case Token(_, text):
                    buf += f"{indent}  '{text}'\n"
                case Tree(_, __) as tree:
                    buf = tree.print(buf, level + 1)

        assert buf.endswith('\n')
        return buf

    def __repr__(self):
        return self.print()


def parse(text: str) -> Tree:
    tokens = lex(text)
    p = Parser(tokens)
    file(p)
    return p.build_tree()


KEYWORD_TYPES = tuple(t for t in TokenKind if isinstance(t.value, tuple))
KEYWORDS = tuple(t.value[0] for t in KEYWORD_TYPES)

TYPES = tuple(t for t in TokenKind if isinstance(t.value, str))
SCANNER = re.compile("|".join(f"(?P<{t.name}>{t.value})" for t in TYPES))


def scan(text: str) -> Iterable[Token]:
    for match in SCANNER.finditer(text):
        if match:
            i = match.lastindex - 1
            text = match[i + 1]
            type_ = TYPES[i]
            if type_ == TokenKind.Name and text in KEYWORDS:
                i = KEYWORDS.index(text)
                yield Token(KEYWORD_TYPES[i], text)
            else:
                yield Token(TYPES[i], text)
        else:
            yield Token(TokenKind.ErrorToken, '')


def lex(text: str) -> list[Token]:
    return list(scan(text))


type Event = EventOpen | EventClose | EventAdvance


class EventOpen(NamedTuple):
    kind: TreeKind


class EventClose:
    pass


class EventAdvance:
    pass


class MarkOpened(NamedTuple):
    index: int


class MarkClosed(NamedTuple):
    index: int


@dataclass
class Parser:
    tokens: list[Token]
    pos: int = 0
    fuel: int = 256
    events: list[Event] = field(default_factory=list)

    def build_tree(self) -> Tree:
        tokens = iter(self.tokens)
        events = self.events
        stack = []
        assert isinstance(events.pop(), EventClose)

        for event in events:
            match event:
                case EventOpen(kind):
                    stack.append(Tree(kind, []))
                case EventClose():
                    tree = stack.pop()
                    stack[-1].children.append(tree)
                case EventAdvance():
                    token = next(tokens)
                    stack[-1].children.append(token)

        tree = stack.pop()
        assert not stack
        assert next(tokens, None) is None
        return tree

    def open(self) -> MarkOpened:
        mark = MarkOpened(index=len(self.events))
        self.events.append(EventOpen(kind=TreeKind.ErrorTree))
        return mark

    def open_before(self, m: MarkClosed) -> MarkOpened:
        mark = MarkOpened(index=m.index)
        self.events.insert(m.index, EventOpen(kind=TreeKind.ErrorTree))
        return mark

    def close(self, m: MarkOpened, kind: TreeKind) -> MarkClosed:
        self.events[m.index] = EventOpen(kind=kind)
        self.events.append(EventClose())
        return MarkClosed(index=m.index)

    def advance(self):
        assert not self.eof()
        self.fuel = 256
        self.events.append(EventAdvance())
        self.pos += 1

    def advance_with_error(self, error: str):
        m = self.open()
        # TODO: Error reporting.
        print(error, file=sys.stderr)
        self.advance()
        self.close(m, TreeKind.ErrorTree)

    def eof(self) -> bool:
        return self.pos == len(self.tokens)

    def nth(self, lookahead: int) -> TokenKind:
        if self.fuel == 0:
            panic("parser is stuck")
        self.fuel -= 1
        i = self.pos + lookahead
        return self.tokens[i].kind if i < len(self.tokens) else TokenKind.Eof

    def at(self, kind: TokenKind) -> bool:
        return self.nth(0) == kind

    def at_any(self, kinds: Iterable[TokenKind]) -> bool:
        return self.nth(0) in kinds

    def eat(self, kind: TokenKind) -> bool:
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


def file(p: Parser):
    m = p.open()
    while not p.eof():
        if p.at(TokenKind.FnKeyword):
            func(p)
        else:
            p.advance_with_error("expected a function")
    p.close(m, TreeKind.File)


def func(p: Parser):
    assert p.at(TokenKind.FnKeyword)
    m = p.open()
    p.expect(TokenKind.FnKeyword)
    p.expect(TokenKind.Name)
    if p.at(TokenKind.LParen):
        param_list(p)
    if p.eat(TokenKind.Arrow):
        type_expr(p)
    if p.at(TokenKind.LCurly):
        block(p)
    p.close(m, TreeKind.Fn)


PARAM_LIST_RECOVERY = (TokenKind.FnKeyword, TokenKind.LCurly)


def param_list(p: Parser):
    assert p.at(TokenKind.LParen)
    m = p.open()
    p.expect(TokenKind.LParen)
    while not p.at(TokenKind.RParen) and not p.eof():
        if p.at(TokenKind.Name):
            param(p)
        else:
            if p.at_any(PARAM_LIST_RECOVERY):
                break
            p.advance_with_error("expected parameter")
    p.expect(TokenKind.RParen)
    p.close(m, TreeKind.ParamList)


def param(p: Parser):
    assert p.at(TokenKind.Name)
    m = p.open()
    p.expect(TokenKind.Name)
    p.expect(TokenKind.Colon)
    type_expr(p)
    if not p.at(TokenKind.RParen):
        p.expect(TokenKind.Comma)
    p.close(m, TreeKind.Param)


def type_expr(p: Parser):
    m = p.open()
    p.expect(TokenKind.Name)
    p.close(m, TreeKind.TypeExpr)


STMT_RECOVERY = (TokenKind.FnKeyword,)
EXPR_FIRST = (TokenKind.Int, TokenKind.TrueKeyword, TokenKind.FalseKeyword, TokenKind.Name, TokenKind.LParen)


def block(p: Parser):
    assert p.at(TokenKind.LCurly)
    m = p.open()
    p.expect(TokenKind.LCurly)
    while not p.at(TokenKind.RCurly) and not p.eof():
        match p.nth(0):
            case TokenKind.LetKeyword:
                stmt_let(p)
            case TokenKind.ReturnKeyword:
                stmt_return(p)
            case _:
                if p.at_any(EXPR_FIRST):
                    stmt_expr(p)
                else:
                    if p.at_any(STMT_RECOVERY):
                        break
                    p.advance_with_error("expected statement")
    p.expect(TokenKind.RCurly)
    p.close(m, TreeKind.Block)


def stmt_let(p: Parser):
    assert p.at(TokenKind.LetKeyword)
    m = p.open()
    p.expect(TokenKind.LetKeyword)
    p.expect(TokenKind.Name)
    p.expect(TokenKind.Eq)
    expr(p)
    p.expect(TokenKind.Semi)
    p.close(m, TreeKind.StmtLet)


def stmt_return(p: Parser):
    assert p.at(TokenKind.ReturnKeyword)
    m = p.open()
    p.expect(TokenKind.ReturnKeyword)
    expr(p)
    p.expect(TokenKind.Semi)
    p.close(m, TreeKind.StmtReturn)


def stmt_expr(p: Parser):
    m = p.open()
    expr(p)
    p.expect(TokenKind.Semi)
    p.close(m, TreeKind.StmtExpr)


def expr(p: Parser):
    expr_rec(p, TokenKind.Eof)


def expr_rec(p: Parser, left: TokenKind):
    if not (lhs := expr_delimited(p)):
        return

    while p.at(TokenKind.LParen):
        m = p.open_before(lhs)
        arg_list(p)
        lhs = p.close(m, TreeKind.ExprCall)

    while True:
        right = p.nth(0)
        if right_binds_tighter(left, right):
            m = p.open_before(lhs)
            p.advance()
            expr_rec(p, right)
            lhs = p.close(m, TreeKind.ExprBinary)
        else:
            break


def right_binds_tighter(left: TokenKind, right: TokenKind) -> bool:
    def tightness(kind: TokenKind) -> Optional[int]:
        pt = (
            # Precedence table:
            (TokenKind.Plus, TokenKind.Minus),
            (TokenKind.Star, TokenKind.Slash)
        )
        for i, tokens in enumerate(pt):
            if kind in tokens:
                return i
        return None

    if not (right_tightness := tightness(right)):
        return False

    if not (left_tightness := tightness(left)):
        assert left == TokenKind.Eof
        return True

    return right_tightness > left_tightness


def expr_delimited(p: Parser) -> Optional[MarkClosed]:
    match p.nth(0):
        case TokenKind.TrueKeyword | TokenKind.FalseKeyword | TokenKind.Int:
            m = p.open()
            p.advance()
            return p.close(m, TreeKind.ExprLiteral)

        case TokenKind.Name:
            m = p.open()
            p.advance()
            return p.close(m, TreeKind.ExprName)

        case TokenKind.LParen:
            m = p.open()
            p.expect(TokenKind.LParen)
            expr(p)
            p.expect(TokenKind.RParen)
            return p.close(m, TreeKind.ExprParen)

        case _:
            return None


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


def arg(p: Parser):
    m = p.open()
    expr(p)
    if not p.at(TokenKind.RParen):
        p.expect(TokenKind.Comma)
    p.close(m, TreeKind.Arg)


def smoke():
    text = """
    fn f() {
      let x = 1 +
      let y = 2
    }
    """
    cst = parse(text)
    print(cst)


if __name__ == '__main__':
    smoke()
