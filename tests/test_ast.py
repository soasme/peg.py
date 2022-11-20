import string
import json
from pprint import pprint
from dataclasses import dataclass
from hypothesis import example, given, strategies as st
import pytest
from peg import *

def test_compile_bin():
    ast = str_("abc") * RetNode()
    compiler = PegCompiler()
    bc = compiler.compile(ast)
    vm = PegVM(bc)
    state = vm.run("abc")
    assert state.pos == 3
    assert state.pc == 4

def test_compile_bin2():
    ast = str_("a") * str_("b") * str_("c") * RetNode()
    compiler = PegCompiler()
    bc = compiler.compile(ast)
    vm = PegVM(bc)
    state = vm.run("abc")
    assert state.pos == 3
    assert state.pc == 4

def test_compile_any():
    ast = any_(3) * RetNode()
    compiler = PegCompiler()
    bc = compiler.compile(ast)
    vm = PegVM(bc)
    state = vm.run("abc")
    assert state.pos == 3
    assert state.pc == 2

def test_compile_either():
    ast = str_("a") | str_("A") * RetNode()
    compiler = PegCompiler()
    bc = compiler.compile(ast)
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

def test_compile_loop():
    ast = str_("a") ** 0 * RetNode()

    compiler = PegCompiler()
    bc = compiler.compile(ast)
    vm = PegVM(bc)

    state = vm.run("a")
    assert state.pos == 1
    assert state.pc == 4
    assert state.err == PegError.OK

    state = vm.run("aaa")
    assert state.pos == 3
    assert state.pc == 4
    assert state.err == PegError.OK

def test_compile_loop2():
    ast = str_("a") ** 2 * RetNode()

    compiler = PegCompiler()
    bc = compiler.compile(ast)
    vm = PegVM(bc)

    state = vm.run("aaa")
    assert state.pos == 3
    assert state.pc == 6
    assert state.err == PegError.OK

    state = vm.run("aa")
    assert state.pos == 2
    assert state.pc == 6
    assert state.err == PegError.OK

    state = vm.run("a")
    assert state.pos == 1
    assert state.pc == 1
    assert state.err == PegError.NO_MATCH

def test_compile_optional():
    ast = str_("a").optional() * RetNode()

    compiler = PegCompiler()
    bc = compiler.compile(ast)
    vm = PegVM(bc)

    state = vm.run("a")
    assert state.pos == 1
    assert state.pc == 4
    assert state.err == PegError.OK

    state = vm.run("")
    assert state.pos == 0
    assert state.pc == 4
    assert state.err == PegError.OK

def test_compile_zeroormore():
    ast = str_("a").zeroormore() * RetNode()

    compiler = PegCompiler()
    bc = compiler.compile(ast)
    vm = PegVM(bc)

    state = vm.run("")
    assert state.pos == 0
    assert state.pc == 4
    assert state.err == PegError.OK

    state = vm.run("a")
    assert state.pos == 1
    assert state.pc == 4
    assert state.err == PegError.OK

    state = vm.run("aa")
    assert state.pos == 2
    assert state.pc == 4
    assert state.err == PegError.OK

def test_compile_onceormore():
    ast = str_("a").onceormore() * RetNode()

    compiler = PegCompiler()
    bc = compiler.compile(ast)
    vm = PegVM(bc)

    state = vm.run("")
    assert state.pos == 0
    assert state.pc == 0
    assert state.err == PegError.NO_MATCH

    state = vm.run("a")
    assert state.pos == 1
    assert state.pc == 5
    assert state.err == PegError.OK

    state = vm.run("aa")
    assert state.pos == 2
    assert state.pc == 5
    assert state.err == PegError.OK

def test_compile_positive():
    ast = +str_("a") * any_() * str_("b") * RetNode()

    compiler = PegCompiler()
    bc = compiler.compile(ast)
    vm = PegVM(bc)

    state = vm.run("ab")
    assert state.pos == 2
    assert state.pc == 7
    assert state.err == PegError.OK

    state = vm.run("bb")
    assert state.pos == 0
    assert state.pc == 3
    assert state.err == PegError.NO_MATCH

def test_compile_negative():
    ast = -str_("a") * any_() * str_("b") * RetNode()

    compiler = PegCompiler()
    bc = compiler.compile(ast)
    vm = PegVM(bc)

    state = vm.run("bb")
    assert state.pos == 2
    assert state.pc == 7
    assert state.err == PegError.OK

    state = vm.run("ab")
    assert state.pos == 1
    assert state.pc == 3
    assert state.err == PegError.NO_MATCH

def test_compile_set():
    ast = set_("aA") * RetNode()

    compiler = PegCompiler()
    bc = compiler.compile(ast)
    vm = PegVM(bc)

    state = vm.run("a")
    assert state.pos == 1
    assert state.pc == 4
    assert state.err == PegError.OK

    state = vm.run("A")
    assert state.pos == 1
    assert state.pc == 4
    assert state.err == PegError.OK

    state = vm.run("b")
    assert state.pos == 0
    assert state.pc == 0
    assert state.err == PegError.NO_MATCH

def test_compile_char_set():
    ast = range_("a", "f", "A", "F") * RetNode()

    compiler = PegCompiler()
    bc = compiler.compile(ast)
    vm = PegVM(bc)

    state = vm.run("a")
    assert state.pos == 1
    assert state.pc == 6
    assert state.err == PegError.OK

    state = vm.run("z")
    assert state.pos == 0
    assert state.pc == 0
    assert state.err == PegError.NO_MATCH

    state = vm.run("A")
    assert state.pos == 1
    assert state.pc == 6
    assert state.err == PegError.OK

    state = vm.run("Z")
    assert state.pos == 0
    assert state.pc == 0
    assert state.err == PegError.NO_MATCH

def test_compile_capture():
    ast = capture_(str_("a")) * RetNode()

    compiler = PegCompiler()
    bc = compiler.compile(ast)
    vm = PegVM(bc)

    state = vm.run("a")
    assert state.pos == 1
    assert state.pc == 4
    assert state.caps.stack == [
        CapEntry('L', pc=0, pos=0),
        CapEntry('R', pc=2, pos=1),
    ]
    assert state.err == PegError.OK

def test_compile_choice_capture():
    ast = capture_(str_("a")) | capture_(str_("A")) * RetNode()

    compiler = PegCompiler()
    bc = compiler.compile(ast)
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

def test_grammar_str():
    class Str:
        a = str_("a")

    gram = Grammar(Str)
    assert gram.parse('a')
    assert not gram.parse('b')
    
def test_grammar_strcjk():
    class Str:
        a = str_("中文")

    gram = Grammar(Str)
    assert gram.parse('中文')
    assert not gram.parse('hah 中文')

def test_grammar_strrange_outofindex():
    class Str:
        a = str_("a") * set_("aA")

    gram = Grammar(Str)
    assert gram.parse('aa')
    assert gram.parse('aA')
    assert not gram.parse('a')

def test_grammar_seq():
    class Str:
        a = str_("a") * str_("b") * str_("c")

    gram = Grammar(Str)
    assert not gram.parse('a')
    assert not gram.parse('ab')
    assert gram.parse('abc')
    assert gram.parse('abcd')

def test_grammar_choice():
    class Str:
        a = str_("a") * str_("b") * str_("c") | str_("ABC")

    gram = Grammar(Str)
    assert gram.parse('abc')
    assert gram.parse('ABC')

def test_grammar_choice2():
    class Str:
        a = str_("ABC") | str_("a") * str_("b") * str_("c")

    gram = Grammar(Str)
    assert gram.parse('abc')
    assert gram.parse('ABC')


def test_grammar_var():
    class Str:
        lower = str_("abc") 
        upper = str_("ABC")
        start = rule_('lower') | rule_('upper')

    gram = Grammar(Str)
    assert gram.parse('abc', rule='start')
    assert gram.parse('ABC', rule='start')
    assert not gram.parse('aBc', rule='start')

def test_grammar_bad_range():
    try:
        class Str:
            a = range_("a", "b", "c")
    except ValueError:
        pass
