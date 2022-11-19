import string
import json
from pprint import pprint
from dataclasses import dataclass
from hypothesis import example, given, strategies as st
import pytest
from peg import *

def test_noop():
    bc = [
        (OpCode.NOOP, 0),
        (OpCode.NOOP, 0),
        (OpCode.NOOP, 0),
    ]
    vm = PegVM(bc)
    state = vm.run("abc")
    assert state.pos == 0
    assert state.pc == 3

def test_any():
    bc = [
        (OpCode.ANY, 1),
        (OpCode.ANY, 1),
        (OpCode.ANY, 1),
    ]
    vm = PegVM(bc)
    state = vm.run("abc")
    assert state.pos == 3
    assert state.pc == 3

def test_any3():
    bc = [
        (OpCode.ANY, 3),
    ]
    vm = PegVM(bc)
    state = vm.run("abc")
    assert state.pos == 3
    assert state.pc == 1

def test_any4():
    bc = [
        (OpCode.ANY, 4),
    ]
    vm = PegVM(bc)
    state = vm.run("abc")
    assert state.err == PegError.NO_MATCH
    assert state.pos == 0
    assert state.pc == 0

def test_char_abc():
    bc = [
        (OpCode.CHAR, ord('a')),
        (OpCode.CHAR, ord('b')),
        (OpCode.CHAR, ord('c')),
    ]
    vm = PegVM(bc)
    state = vm.run("abc")
    assert state.pos == 3
    assert state.pc == 3


def test_char_abc_ne_abd():
    bc = [
        (OpCode.CHAR, ord('a')),
        (OpCode.CHAR, ord('b')),
        (OpCode.CHAR, ord('d')),
    ]
    vm = PegVM(bc)
    state = vm.run("abc")
    assert state.pos == 2
    assert state.pc == 2
    assert state.err == PegError.NO_MATCH

def test_char_set():
    bc = [
        (OpCode.CHAR_SET, 2),
        (OpCode.CHAR, ord('a')),
        (OpCode.CHAR, ord('A')),
    ]
    vm = PegVM(bc)

    state = vm.run("a")
    assert state.pos == 1
    assert state.pc == 3
    assert state.err == PegError.OK

    state = vm.run("A")
    assert state.pos == 1
    assert state.pc == 3
    assert state.err == PegError.OK

    state = vm.run("b")
    assert state.pos == 0
    assert state.pc == 0
    assert state.err == PegError.NO_MATCH

def test_char_set():
    bc = [
        (OpCode.CHAR_RANGE, 2),
        (OpCode.CHAR, ord('a')),
        (OpCode.CHAR, ord('z')),
    ]
    vm = PegVM(bc)

    state = vm.run("a")
    assert state.pos == 1
    assert state.pc == 3
    assert state.err == PegError.OK

    state = vm.run("z")
    assert state.pos == 1
    assert state.pc == 3
    assert state.err == PegError.OK

    state = vm.run("A")
    assert state.pos == 0
    assert state.pc == 0
    assert state.err == PegError.NO_MATCH

def test_ret():
    bc = [
        (OpCode.CHAR, ord('a')),
        (OpCode.CHAR, ord('b')),
        (OpCode.CHAR, ord('c')),
        (OpCode.RET, 0),
        (OpCode.ANY, 1),
    ]
    vm = PegVM(bc)
    state = vm.run("abc")
    assert state.pos == 3
    assert state.pc == 4
    assert state.err == PegError.OK

def test_jump():
    bc = [
        (OpCode.CHAR, ord('a')),
        (OpCode.JUMP, 2),
        (OpCode.ANY, 1),
        (OpCode.CHAR, ord('c')),
    ]
    vm = PegVM(bc)
    state = vm.run("ac")
    assert state.pos == 2
    assert state.pc == 4
    assert state.err == PegError.OK

def test_call():
    # 'abc'
    bc = [
        (OpCode.CALL, 2),
        (OpCode.RET, 0),

        (OpCode.CHAR, ord('a')),
        (OpCode.CHAR, ord('b')),
        (OpCode.CHAR, ord('c')),
        (OpCode.RET, 0),
    ]
    vm = PegVM(bc)
    state = vm.run("abc")
    assert state.pos == 3
    assert state.pc == 2
    assert state.err == PegError.OK

def test_choice():
    # either 'a' or 'A'
    bc = [
        (OpCode.CHOICE, 3),
        (OpCode.CHAR, ord('a')),
        (OpCode.COMMIT, 2),
        (OpCode.CHAR, ord('A')),
        (OpCode.RET, 0),
    ]
    vm = PegVM(bc)
    state = vm.run("a")
    assert state.pos == 1
    assert state.pc == 5
    assert state.err == PegError.OK

    state = vm.run("A")
    assert state.pos == 1
    assert state.pc == 5
    assert state.err == PegError.OK

    state = vm.run("b")
    assert state.pos == 0
    assert state.pc == 3 # checked 'a', and failed when checking 'b'.
    assert state.err == PegError.NO_MATCH

def test_commit_nothing():
    # either 'a' or 'A'
    bc = [
        (OpCode.COMMIT, 2),
        (OpCode.CHAR, ord('A')),
        (OpCode.RET, 0),
    ]
    vm = PegVM(bc)
    state = vm.run("a")
    assert state.pos == 0
    assert state.pc == 0
    assert state.err == PegError.NO_COMMIT_ENTRY

def test_loop():
    # 'a'*
    bc = [
        (OpCode.CHOICE, 3),
        (OpCode.CHAR, ord('a')),
        (OpCode.LOOP, 2),
        (OpCode.RET, 0),
    ]
    vm = PegVM(bc)

    state = vm.run("a")
    assert state.pos == 1
    assert state.pc == 4
    assert state.err == PegError.OK

    state = vm.run("aaa")
    assert state.pos == 3
    assert state.pc == 4
    assert state.err == PegError.OK

def test_positive_predicate():
    # &"a" . "b"
    bc = [
        (OpCode.CHOICE, 3),
        (OpCode.CHAR, ord('a')),
        (OpCode.BACK_COMMIT, 2),
        (OpCode.FAIL, 0),
        (OpCode.ANY, 1),
        (OpCode.CHAR, ord('b')),
        (OpCode.RET, 0),
    ]
    vm = PegVM(bc)

    state = vm.run("ab")
    assert state.pos == 2
    assert state.pc == 7
    assert state.err == PegError.OK

    state = vm.run("bb")
    assert state.pos == 0
    assert state.pc == 4
    assert state.err == PegError.NO_MATCH

def test_capture():
    # <"a">
    bc = [
        (OpCode.LEFT_CAPTURE, 0),
        (OpCode.CHAR, ord('a')),
        (OpCode.RIGHT_CAPTURE, 0),
        (OpCode.RET, 0),
    ]
    vm = PegVM(bc)

    state = vm.run("a")
    assert state.pos == 1
    assert state.pc == 4
    assert state.caps.stack == [
        CapEntry('L', pc=0, pos=0),
        CapEntry('R', pc=2, pos=1),
    ]
    assert state.err == PegError.OK

def test_choice_capture():
    # <"a"> | <"A">
    bc = [
        (OpCode.CHOICE, 5),
        (OpCode.LEFT_CAPTURE, 0),
        (OpCode.CHAR, ord('a')),
        (OpCode.RIGHT_CAPTURE, 0),
        (OpCode.COMMIT, 4),
        (OpCode.LEFT_CAPTURE, 0),
        (OpCode.CHAR, ord('A')),
        (OpCode.RIGHT_CAPTURE, 0),
        (OpCode.RET, 0),
    ]
    vm = PegVM(bc)

    state = vm.run("a")
    assert state.pos == 1
    assert state.pc == 9
    assert state.caps.stack == [
        CapEntry('L', pc=1, pos=0),
        CapEntry('R', pc=3, pos=1),
    ]
    assert state.err == PegError.OK

    state = vm.run("A")
    assert state.pos == 1
    assert state.pc == 9
    assert state.caps.stack == [
        CapEntry('L', pc=5, pos=0),
        CapEntry('R', pc=7, pos=1),
    ]
    assert state.err == PegError.OK

    # this case's behavior is a little odd.
    state = vm.run("b")
    assert state.pos == 0
    assert state.pc == 6
    assert state.err == PegError.NO_MATCH
    # particularly, the capture stack is not cleared.
    assert state.caps.stack == [
        CapEntry('L', pc=5, pos=0),
    ]
