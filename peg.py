import ast
import itertools
import inspect
from enum import Enum, auto
from typing import Any, List
from functools import singledispatchmethod, cached_property
from dataclasses import dataclass

class GrammarError(Exception):
    pass

class OpCode(Enum):
    NOOP = auto()
    RET = auto()
    CHAR = auto()
    CHAR_SET = auto()
    CHAR_RANGE = auto()
    ANY = auto()
    JUMP = auto()
    CALL = auto()
    FAIL = auto()
    CHOICE = auto() # CHOICE, <offset>
    COMMIT = auto()
    LOOP = auto()
    BACK_COMMIT = auto()
    LEFT_CAPTURE = auto()
    RIGHT_CAPTURE = auto()

class PegError(Enum):
    OK = 0
    NO_MATCH = auto()
    NO_COMMIT_ENTRY = auto()

class Stack:
    def __init__(self):
        self.stack = []
    def push(self, item):
        self.stack.append(item)
    def pop(self):
        return self.stack.pop()
    def peek(self):
        return self.stack[-1]
    def isEmpty(self):
        return self.stack == []
    def size(self):
        return len(self.stack)
    def replaceTop(self, item):
        self.stack[-1] = item
    def __repr__(self):
        return str(self.stack)

class CallFrame:
    def __init__(self, type, pc, pos, caps_size):
        self.type = type
        self.pc = pc
        self.pos = pos
        self.caps_size = caps_size
    def __repr__(self):
        return str(vars(self))

class CapEntry:
    def __init__(self, type, pc, pos):
        self.type = type
        self.pc = pc
        self.pos = pos

    def __eq__(self, o):
        return self.type == o.type and self.pc == o.pc and self.pos == o.pos

    def __repr__(self):
        return str(vars(self))

class Node:
    def __init__(self, start, end, text, rule):
        self.start = start
        self.end = end
        self.text = text
        self.rule = rule

    def __repr__(self):
        return f'{self.__class__.__name__}({self.start}, {self.end}, {self.text}, {self.rule})'

class PegVMState:

    def __init__(self, text, pc, pos, err, matching, refs, calls, caps, precedences):
        self.text = text
        self.pc = pc
        self.pos = pos
        self.err = err
        self.matching = matching
        self.refs = refs
        self.calls = calls
        self.caps = caps
        self.precedences = precedences
        self.errmsg = ''

    def __repr__(self):
        return str(vars(self))

    def __str__(self):
        if self.matching:
            return '<Result success=True>'
        else:
            return f'<Result error={self.err}>'

    def __bool__(self):
        return self.err == PegError.OK and self.matching



class NodeVisitor:

    def __init__(self, grammar):
        self.grammar = grammar

    def generic_visit(self, _, __):
        return None

    def visit(self, state):
        """
        [[[[]][]][]]
        node_stack: keep track of the current node. L: push(L), R: (pop(), R)
        children_stack: keep track of the current children. L: push([]), R: pop() for node, peek() for current value.
        """
        rule_indices = {v: k for k, v in self.grammar.bc_indices.items()}
        node_stack = Stack()
        children_stack = Stack()
        children_stack.push([])
        for cap in state.caps.stack:
            if cap.type == 'L':
                node_stack.push(Node(cap.pos, 0, None, rule_indices.get(cap.pc, '')))
                children_stack.push([])
            elif cap.type == 'R':
                node = node_stack.pop()
                node.end = cap.pos
                node.text = state.text[node.start:node.end]
                visitor = getattr(self, 'visit_' + node.rule, self.generic_visit)
                child = visitor(node, children_stack.pop())
                children_stack.peek().append(child)

        root = children_stack.pop()
        if root:
            return root[0]

class PegVM:
    def __init__(self, bc, start=0, debug=False):
        self.bc = bc
        self.start = start
        self.debug = debug

    def debug_print(self, state):
        s = f'pos={state.pos}, pc={state.pc}, err={state.err}, matching={state.matching}, op={self.bc[state.pc]}'
        if not state.matching:
            s += f', calls={state.calls.stack}'
        print(s)

    def run(self, text):
        # Initialize the VM state
        state = PegVMState(text, self.start, 0, PegError.OK, True, {}, Stack(), Stack(), Stack())

        while True:
            # check whether the vm run loop should continue.
            if state.pos > len(text) or state.pc >= len(self.bc):
                return state

            if self.debug:
                self.debug_print(state)

            if not state.matching:
                if not self.handle_exit(state):
                    return state

            if state.err != PegError.OK:
                return state

            op = self.bc[state.pc]
            opcode = op[0]
            operand = op[1]

            if opcode == OpCode.RET and state.calls.isEmpty():
                self.handle_NOOP(state, operand)
                return state

            handler = getattr(self, 'handle_' + opcode.name)
            handler(state, operand)


    def handle_exit(self, state):
        # parse failed if no matching and nothing to backtrack.
        if state.calls.isEmpty():
            state.err = PegError.NO_MATCH
            return False

        # recover the state if no matching but something to backtrack.
        # state is recovered using the last call's state.
        peek = state.calls.pop()

        if self.debug:
            print(f'peek type is {peek.type}')

        if peek.type == "backtrack":
            state.pc = peek.pc
            state.pos = peek.pos

            for _ in range(
                state.caps.size() - peek.caps_size
            ):
                state.caps.pop()
            # state.precedences = peek.precedences # todo: support precedences in callframe

        # keep proceeding the vm loop.
        state.matching = True
        state.err = PegError.OK

        return True

    def handle_NOOP(self, state, _):
        state.pc += 1

    def handle_RET(self, state, _):
        state.pc = state.calls.pop().pc

    def handle_ANY(self, state, n):
        pos = state.pos + n
        if pos > len(state.text):
            state.err = PegError.NO_MATCH
            state.matching = False
        else:
            state.pos = pos
            state.pc += 1

    def handle_CHAR(self, state, c):
        if state.pos >= len(state.text):
            state.matching = False
            return

        ch = ord(state.text[state.pos])
        if ch == c:
            state.pos += 1
            state.pc += 1
        else:
            state.matching = False

    def handle_JUMP(self, state, offset):
        state.pc += offset

    def handle_CALL(self, state, offset):
        frame = CallFrame("nat", state.pc + 1, state.pos, state.caps.size())
        state.calls.push(frame)
        state.pc += offset

    def handle_FAIL(self, state, _):
        state.matching = False

    def handle_CHOICE(self, state, offset):
        frame = CallFrame("backtrack", state.pc + offset, state.pos, state.caps.size())
        state.calls.push(frame)
        state.pc += 1

    def handle_COMMIT(self, state, offset):
        if state.calls.isEmpty():
            state.err = PegError.NO_COMMIT_ENTRY
        else:
            state.pc += offset
            state.matching = True
            state.calls.pop()

    def handle_LOOP(self, state, offset):
        if state.pos < len(state.text): # loop before eot ??? is it right?
            peek = state.calls.pop()
            peek.pos = state.pos
            state.pc -= offset
        else:
            state.calls.pop()
            state.pc += 1

    def handle_BACK_COMMIT(self, state, offset):
        frame = state.calls.pop()
        state.pc += offset
        state.pos = frame.pos

    def handle_LEFT_CAPTURE(self, state, _):
        state.caps.push(CapEntry('L', state.pc, state.pos))
        state.pc += 1

    def handle_RIGHT_CAPTURE(self, state, _):
        state.caps.push(CapEntry('R', state.pc, state.pos))
        state.pc += 1

    def handle_CHAR_SET(self, state, offset):
        if state.pos >= len(state.text):
            state.matching = False
            return

        ch = ord(state.text[state.pos])
        for i in range(offset):
            op = self.bc[state.pc + i + 1]
            opcode = op[0]
            assert opcode == OpCode.CHAR
            operand = op[1]
            if ch == operand:
                state.pos += 1
                state.pc += offset + 1
                return

        state.matching = False

    def handle_CHAR_RANGE(self, state, offset):
        if state.pos >= len(state.text):
            state.matching = False
            return

        ch = ord(state.text[state.pos])
        for i in range(0, offset, 2):
            opleft = self.bc[state.pc + i + 1]
            opright = self.bc[state.pc + i + 2]

            assert opleft[0] == OpCode.CHAR
            assert opright[0] == OpCode.CHAR

            operand_left, operand_right = opleft[1], opright[1]

            if operand_left <= ch <= operand_right:
                state.pos += 1
                state.pc += offset + 1
                return

        state.matching = False

@dataclass
class BaseNode:

    def __mul__(self, other):
        return BinNode(left=self, right=other)

    def __or__(self, other):
        return EitherNode(left=self, right=other)

    def __pow__(self, other):
        return LoopNode(node=self, num=other)

    def zeroormore(self):
        return LoopNode(node=self, num=0)

    def onceormore(self):
        return LoopNode(node=self, num=1)

    def optional(self):
        return OptionalNode(node=self)

    def __pos__(self):
        return PositiveNode(node=self)

    def __neg__(self):
        return NegativeNode(node=self)

@dataclass
class RefNode(BaseNode):
    var: str

@dataclass
class StrNode(BaseNode):
    text: str

@dataclass
class CharsetNode(BaseNode):
    chars: str

@dataclass
class CharRangeNode(BaseNode):
    chars: List[str]

@dataclass
class AnyNode(BaseNode):
    num: int

@dataclass
class RetNode(BaseNode):
    pass

@dataclass
class BinNode(BaseNode):
    """This node represents a binary tree node for sequence ."""
    left: Any
    right: Any

@dataclass
class EitherNode(BaseNode):
    """This node represents a binary tree node for choice."""
    left: Any
    right: Any

@dataclass
class LoopNode(BaseNode):
    """This node represents a loop node."""
    node: Any
    num: int = 0

@dataclass
class OptionalNode(BaseNode):
    """This node represents a optional node."""
    node: Any

@dataclass
class PositiveNode(BaseNode):
    """This node represents a positive node."""
    node: Any

@dataclass
class NegativeNode(BaseNode):
    """This node represents a negative node."""
    node: Any

@dataclass
class CaptureNode(BaseNode):
    """This node represents a capture node."""
    node: Any

@dataclass
class CaptureLeftNode(BaseNode):
    pass

@dataclass
class CaptureRightNode(BaseNode):
    pass

class PegCompiler:

    def compile(self, ast):
        bc = []
        self.compile_expr(ast, bc)
        return bc

    @singledispatchmethod
    def compile_expr(self, expr, bc):
        raise GrammarError(f"invalid grammar expression: {expr}")

    @compile_expr.register
    def compile_str(self, expr: StrNode, bc):
        for ch in expr.text:
            bc.append((OpCode.CHAR, ord(ch)))

    @compile_expr.register
    def compile_any(self, expr: AnyNode, bc):
        bc.append((OpCode.ANY, expr.num))

    @compile_expr.register
    def compile_ret(self, expr: RetNode, bc):
        bc.append((OpCode.RET, 0))

    @compile_expr.register
    def compile_bin(self, expr: BinNode, bc):
        self.compile_expr(expr.left, bc)
        self.compile_expr(expr.right, bc)

    @compile_expr.register
    def compile_either(self, expr: EitherNode, bc):
        idx_start = len(bc)
        bc.append((OpCode.CHOICE, 0))

        self.compile_expr(expr.left, bc)

        idx_commit = len(bc)
        bc.append((OpCode.COMMIT, 0))

        bc[idx_start] = (OpCode.CHOICE, len(bc) - idx_start)

        self.compile_expr(expr.right, bc)

        bc[idx_commit] = (OpCode.COMMIT, len(bc) - idx_commit)

    @compile_expr.register
    def compile_loop(self, expr: LoopNode, bc):
        if expr.num > 0:
            for _ in range(expr.num):
                self.compile_expr(expr.node, bc)

        idx_start = len(bc)
        bc.append((OpCode.CHOICE, 0))

        self.compile_expr(expr.node, bc)

        bc.append((OpCode.LOOP, len(bc) - idx_start))
        bc[idx_start] = (OpCode.CHOICE, len(bc) - idx_start)

    @compile_expr.register
    def compile_optional(self, expr: OptionalNode, bc):
        idx_start = len(bc)
        bc.append((OpCode.CHOICE, 0))

        self.compile_expr(expr.node, bc)

        bc.append((OpCode.COMMIT, 1))
        bc[idx_start] = (OpCode.CHOICE, len(bc) - idx_start)

    @compile_expr.register
    def compile_positive(self, expr: PositiveNode, bc):
        idx_start = len(bc)
        bc.append((OpCode.CHOICE, 0))

        self.compile_expr(expr.node, bc)

        idx_back_commit = len(bc)
        bc.append((OpCode.BACK_COMMIT, 0))

        bc[idx_start] = (OpCode.CHOICE, len(bc) - idx_start)

        bc.append((OpCode.FAIL, 0))

        bc[idx_back_commit] = (OpCode.BACK_COMMIT, len(bc) - idx_back_commit)

    @compile_expr.register
    def compile_negative(self, expr: NegativeNode, bc):
        idx_start = len(bc)
        bc.append((OpCode.CHOICE, 0))

        self.compile_expr(expr.node, bc)

        bc.append((OpCode.COMMIT, 1))
        bc.append((OpCode.FAIL, 0))

        bc[idx_start] = (OpCode.CHOICE, len(bc) - idx_start)

    @compile_expr.register
    def compile_charset(self, expr: CharsetNode, bc):
        bc.append((OpCode.CHAR_SET, len(expr.chars)))
        for ch in expr.chars:
            bc.append((OpCode.CHAR, ord(ch)))

    @compile_expr.register
    def compile_charrange(self, expr: CharRangeNode, bc):
        bc.append((OpCode.CHAR_RANGE, len(expr.chars)))
        for ch in expr.chars:
            bc.append((OpCode.CHAR, ord(ch)))

    @compile_expr.register
    def compile_capture(self, expr: CaptureNode, bc):
        bc.append((OpCode.LEFT_CAPTURE, 0))
        self.compile_expr(expr.node, bc)
        bc.append((OpCode.RIGHT_CAPTURE, 0))

    @compile_expr.register
    def compile_ref(self, expr: RefNode, bc):
        bc.append((OpCode.CALL, expr.var)) # TODO: replace index.

    @compile_expr.register
    def compile_capture_left(self, expr: CaptureLeftNode, bc):
        bc.append((OpCode.LEFT_CAPTURE, 0))

    @compile_expr.register
    def compile_capture_right(self, expr: CaptureRightNode, bc):
        bc.append((OpCode.RIGHT_CAPTURE, 0))

def str_(n):
    return StrNode(n)

def any_(n=1):
    return AnyNode(n)

def set_(s):
    return CharsetNode(s)

def range_(start, end, *args):
    if len(args) > 0 and len(args) % 2 != 0:
        raise ValueError('range values not even')
    return CharRangeNode([start, end] + list(args))

def capture_(n):
    return CaptureNode(n)

def begin_():
    return CaptureLeftNode()

def end_():
    return CaptureRightNode()

def rule_(n):
    return RefNode(n)

class Bootstrap:

    Grammar = rule_('Spacing') * rule_('Definition').onceormore() * rule_('EndOfFile')
    Definition = rule_('Identifier') * rule_('LEFTARROW') * rule_('Expression')
    Expression = rule_('Sequence') * (rule_('SLASH') * rule_('Sequence')).zeroormore()
    Sequence = rule_('Prefix').zeroormore()
    Prefix = (rule_('AND') | rule_('NOT')).optional() * rule_('Suffix')
    Suffix = rule_('Primary') * (rule_('QUERY') | rule_('STAR') | rule_('PLUS')).optional()
    Primary = (
        (rule_('Identifier') * -rule_('LEFTARROW')) |
        ((rule_('OPEN') * rule_('Expression') * rule_('CLOSE')) |
        (rule_('Literal') |
        (rule_('Class') |
        (rule_('DOT') |
        (rule_('BEGIN') |
        rule_('END'))))))
    )
    Identifier = rule_('IdentStart') * rule_('IdentCont').zeroormore() * rule_('Spacing')
    IdentStart = range_('a', 'z', 'A', 'Z') | str_('_')
    IdentCont = rule_('IdentStart') | range_('0', '9')
    Literal = ( # TODO: support escaping inside the literal.
        (
            str_('\'') *
            (-str_('\'') * rule_('Char')).zeroormore() *
            str_('\'') *
            rule_('Spacing')
        ) |
        (
            str_('"') *
            (-str_('"') * rule_('Char')).zeroormore() *
            str_('"') *
            rule_('Spacing')
        )
    )
    Class = (
        str_('[') *
        (-str_(']') * rule_('Range')).zeroormore() *
        str_(']') *
        rule_('Spacing')
    )
    Range = rule_('Char') * str_('-') * rule_('Char') | rule_('Char')
    Char = (
        str_('\\') * set_('abefnrtv\'"[]\\') |
        (str_('\\') * range_('0', '3') * range_('0', '7') * range_('0', '7') |
        (str_('\\') * range_('0', '7') * range_('0', '7').optional() |
        (str_('\\') * str_('-') |
        -str_('\\') * any_())))
    )
    LEFTARROW = str_('<-') * rule_('Spacing')
    SLASH = str_('/') * rule_('Spacing')
    AND = str_('&') * rule_('Spacing')
    NOT = str_('!') * rule_('Spacing')
    QUERY = str_('?') * rule_('Spacing')
    STAR = str_('*') * rule_('Spacing')
    PLUS = str_('+') * rule_('Spacing')
    OPEN = str_('(') * rule_('Spacing')
    CLOSE = str_(')') * rule_('Spacing')
    DOT = str_('.') * rule_('Spacing')
    Spacing = (rule_('Space') | rule_('Comment')).zeroormore()
    Comment = (
        str_('#') * 
        (-rule_('EndOfLine') * any_()).zeroormore() *
        rule_('EndOfLine')
    )
    Space = (str_(' ') | (str_('\t') | rule_('EndOfLine')))
    EndOfLine = (str_('\r\n') | (str_('\r') | str_('\n')))
    EndOfFile = -any_()
    BEGIN = str_('<') * rule_('Spacing') 
    END = str_('>') * rule_('Spacing')

class BootstrapVisitor(NodeVisitor):

    def visit_Identifier(self, node, children):
        return ''.join(c for c in children if c is not None)

    def visit_IdentStart(self, node, children):
        return node.text

    def visit_IdentCont(self, node, children):
        return node.text 

    def visit_Char(self, node, children):
        if node.text.startswith('\\'):
            if node.text[1] == 'a':
                return '\a'
            elif node.text[1] == 'b':
                return '\b'
            elif node.text[1] == 'e':
                return '\x1b'
            elif node.text[1] == 'f':
                return '\f'
            elif node.text[1] == 'n':
                return '\n'
            elif node.text[1] == 'r':
                return '\r'
            elif node.text[1] == 't':
                return '\t'
            elif node.text[1] == 'v':
                return '\v'
            elif node.text[1] == '\'':
                return '\''
            elif node.text[1] == '"':
                return '"'
            elif node.text[1] == '[':
                return '['
            elif node.text[1] == ']':
                return ']'
            elif node.text[1] == '\\':
                return '\\'
            else:
                return chr(int(node.text[1:], 8))
        return node.text

    def visit_Range(self, node, children):
        return children

    def visit_Class(self, node, children):
        charset = set()
        charrange = []
        for child in children:
            if not child:
                continue
            if len(child) == 1:
                charset.add(child[0])
            elif len(child) == 2:
                charrange.extend(child)
        if charset and not charrange:
            return set_(''.join(charset))
        elif not charset and charrange:
            return range_(*charrange)
        elif charset and charrange:
            return set_(''.join(charset)) | range_(*charrange)
        else:
            raise ValueError('Empty character class')

    def visit_Literal(self, node, children):
        return str_(''.join(c for c in children if c is not None))

    def visit_DOT(self, node, children):
        return any_()

    def visit_Primary(self, node, children):
        if isinstance(children[0], str): # Identifier
            return rule_(children[0])
        else:
            try:
                return next(c for c in children if c is not None)
            except StopIteration:
                raise ValueError(f'Empty primary: {node.text}')

    def visit_QUERY(self, node, children):
        return '?'

    def visit_STAR(self, node, children):
        return '*'

    def visit_PLUS(self, node, children):
        return '+'

    def visit_NOT(self, node, children):
        return '!'

    def visit_LEFTARROW(self, node, children):
        return '<-'

    def visit_BEGIN(self, node, children):
        return begin_()

    def visit_END(self, node, children):
        return end_()

    def visit_Suffix(self, node, children):
        if len(children) == 1:
            return children[0]
        else:
            if children[1] == '?':
                return children[0].optional()
            elif children[1] == '*':
                return children[0].zeroormore()
            elif children[1] == '+':
                return children[0].onceormore()

    def visit_Prefix(self, node, children):
        if len(children) == 1:
            return children[0]
        else:
            if children[0] == '!':
                return -children[1]
            elif children[0] == '&':
                return +children[1]

    def visit_Sequence(self, node, children):
        if not children:
            return None
        result = children[0]
        for child in children[1:]:
            result = result * child
        return result

    def visit_Expression(self, node, children):
        children = [c for c in children if c is not None]
        result = children[-1]
        for child in reversed(children[:-1]):
            result = child | result
        return result

    def visit_Definition(self, node, children):
        identifier, _, expression = children
        return (identifier, expression)

    def visit_Grammar(self, node, children):
        children = children[1:-1]
        grammar = {identifier: expression for identifier, expression in children}
        return grammar

compiler = PegCompiler()

class Grammar:

    def __init__(self, grammar, auto_capture=True):
        self.rules = {}
        self.bc_indices = {}
        self.bc = []

        if isinstance(grammar, str):
            tree = bootstrap.parse(grammar, rule='Grammar', strict_eot=True)
            if not tree or tree.errmsg:
                raise ValueError(f'invalid grammar: {tree.errmsg}')
            _rules = BootstrapVisitor(bootstrap).visit(tree)
        elif isinstance(grammar, dict):
            _rules = grammar
        elif inspect.isclass(grammar):
            _rules = grammar.__dict__
        else:
            raise ValueError('invalid grammar')

        for key, value in _rules.items():
            if key.startswith('__'): # ignore magic methods
                continue

            self.rules[key] = value
            self.bc_indices[key] = len(self.bc)

            try:
                rule = self.rules[key]
                if auto_capture:
                    rule = capture_(rule)
                self.bc += compiler.compile(
                    rule * RetNode()
                )
            except GrammarError as e:
                raise GrammarError(f'invalid grammar rule `{key}`: {e}')

        # fix call label.
        for idx in range(len(self.bc)):
            op = self.bc[idx]
            if op[0] == OpCode.CALL:
                if op[1] not in self.bc_indices:
                    raise GrammarError(f'invalid grammar reference `{op[1]}`')
                self.bc[idx] = (op[0], self.bc_indices[op[1]]-idx)


    def parse(self, text, rule=None, strict_eot=False, debug=False):
        if not rule:
            start = 0
        elif rule in self.bc_indices:
            start = self.bc_indices[rule]
        else:
            raise ValueError('rule not found: ' + rule)

        vm = PegVM(self.bc, start, debug=debug)
        state = vm.run(text)

        if not state or (strict_eot and state.pos != len(text)):
            expect_rule = list(itertools.dropwhile(
                lambda x: x[1] > state.pc,
                self.bc_indices.items()
            ))[-1][0]
            state.errmsg = f"expect {expect_rule} but '{text[state.pos:state.pos+1]}' found at pos {state.pos}"
        return state
        
bootstrap = Grammar(Bootstrap)
