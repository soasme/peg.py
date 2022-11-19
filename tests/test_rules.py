import string
import json
from pprint import pprint
from dataclasses import dataclass
from hypothesis import example, given, strategies as st
import pytest
from peg import *

def test_bootstraping_peg():
    PEG = r"""
           Grammar <- Spacing Definition+ EndOfFile
           Definition      <- Identifier LEFTARROW Expression
           Expression      <- Sequence ( SLASH Sequence )*
           Sequence        <- Prefix*
           Prefix          <- AND Action
                            / ( AND / NOT )? Suffix
           Suffix          <- Primary ( QUERY / STAR / PLUS )?
           Primary         <- Identifier !LEFTARROW
                            / OPEN Expression CLOSE
                            / Literal
                            / Class
                            / DOT
                            / Action
                            / BEGIN
                            / END

           Identifier      <- < IdentStart IdentCont* > Spacing
           IdentStart      <- [a-zA-Z_]
           IdentCont       <- IdentStart / [0-9]
           Literal         <- ['] < ( !['] Char  )* > ['] Spacing
                            / ["] < ( !["] Char  )* > ["] Spacing
           Class           <- '[' < ( !']' Range )* > ']' Spacing
           Range           <- Char '-' Char / Char
           Char            <- '\\' [abefnrtv'"\[\]\\]
                            / '\\' [0-3][0-7][0-7]
                            / '\\' [0-7][0-7]?
                            / '\\' '-'
                            / !'\\' .
           LEFTARROW       <- '<-' Spacing
           SLASH           <- '/' Spacing
           AND             <- '&' Spacing
           NOT             <- '!' Spacing
           QUERY           <- '?' Spacing
           STAR            <- '*' Spacing
           PLUS            <- '+' Spacing
           OPEN            <- '(' Spacing
           CLOSE           <- ')' Spacing
           DOT             <- '.' Spacing
           Spacing         <- ( Space / Comment )*
           Comment         <- '#' ( !EndOfLine . )* EndOfLine
           Space           <- ' ' / '\t' / EndOfLine
           EndOfLine       <- '\r\n' / '\n' / '\r'
           EndOfFile       <- !.
           Action          <- '{' < [^}]* > '}' Spacing
           BEGIN           <- '<' Spacing
           END             <- '>' Spacing
    """
    gram = Grammar(PEG)
    assert gram.parse(PEG, rule='Grammar')

def test_bootstrap_str():
    gram = Grammar("""
    rule <- "hello world"
    """)
    x = gram.parse("hello world", rule='rule')
    assert bool(x)

def test_bootstrap_any():
    gram = Grammar("""
    rule <- . . .
    """)
    x = gram.parse("abc", rule='rule')
    assert bool(x)

def test_bootstrap_eot():
    gram = Grammar("""
    rule <- . . . !.
    """)
    assert gram.parse("abc", rule='rule')
    assert not gram.parse("abcd", rule='rule')

def test_bootstrap_empty_chars():
    try:
        gram = Grammar("""
    rule <- []
    """)
    except ValueError:
        pass
    else:
        pytest.fail("expected ValueError")

def test_bootstrap_either():
    gram = Grammar("""
    rule <- "a" / "b"
    """)
    assert gram.parse("a", rule='rule')
    assert gram.parse("b", rule='rule')
    assert not gram.parse("c", rule='rule')
    assert gram.parse("c", rule='rule').errmsg == "expect rule but 'c' found at pos 0"

def test_bootstrap_either123():
    gram = Grammar("""
    rule <- "a" / "b" / "c"
    """)
    assert gram.parse("a", rule='rule')
    assert gram.parse("b", rule='rule')
    assert gram.parse("c", rule='rule')
    assert not gram.parse("d", rule='rule')

def test_bootstrap_either_preemptive():
    gram = Grammar("""
    rule <- "ab" / "abc"
    """)
    assert gram.parse("abc", rule='rule').pos == 2
    assert gram.parse("ab", rule='rule').pos == 2

def test_bootstrap_enclose():
    gram = Grammar("""
    rule <- <"a"> b
    b <- "b"
    """)

    class MyVisitor(NodeVisitor):
        def visit_rule(self, node, children):
            return ('rule', children[0], children[1])

        def visit_b(self, node, _):
            return node.text

        def generic_visit(self, node, children):
            return node.text

    tree = gram.parse("ab", rule='rule')
    assert tree
    assert MyVisitor(gram).visit(tree) == ('rule', 'a', 'b')
    

@given(s=st.text(alphabet=string.ascii_letters + string.digits))
def test_bootstrap_bold(s):
    BoldGrammar = """
        Bold <- BOLD "(" Text ")"
        BOLD <- "Bold"
        Text <- [a-zA-Z0-9 ]*
    """

    @dataclass
    class Bold:
        text: str

    class BoldVisitor(NodeVisitor):

        def visit_Bold(self, node, children):
            _, text = children
            return Bold(text)
        def visit_Text(self, node, _):
            return node.text

    input = f'Bold({s})'

    gram = Grammar(BoldGrammar)
    result = gram.parse(input)
    assert result

    visitor = BoldVisitor(gram, input)
    assert visitor.visit(result) == Bold(s)

@given(s=st.text(alphabet=string.ascii_letters + string.digits))
def test_bold(s):
    class BoldGrammar:
        Bold = rule_("BOLD") * str_("(") * rule_("Text") * str_(")")
        BOLD = str_("Bold")
        Text = (range_("a", "z", "A", "Z", "0", "9") | str_(" ")).zeroormore()

    @dataclass
    class Bold:
        text: str

    class BoldVisitor(NodeVisitor):

        def visit_Bold(self, node, children):
            _, text = children
            return Bold(text)
        def visit_Text(self, node, _):
            return node.text

    input = f'Bold({s})'

    gram = Grammar(BoldGrammar)
    result = gram.parse(input)
    assert result

    visitor = BoldVisitor(gram, input)
    assert visitor.visit(result) == Bold(s)

class MathGrammar:
    Expr = rule_("Sum")
    Sum = rule_("Product") * (rule_('SumOp') * rule_("Product")).zeroormore()
    SumOp = str_("+") | str_("-")
    Product = rule_("Power") * (rule_('ProductOp') * rule_("Power")).zeroormore()
    ProductOp = str_("*") | str_("/")
    Power = rule_("Value") * (str_("**") * rule_("Power")).zeroormore()
    Value = str_("(") * rule_("Expr") * str_(")") | range_("0", "9").onceormore()

class MathVisitor(NodeVisitor):
    def generic_visit(self, node, _):
        return node.text
    def visit_Value(self, node, children):
        if not children:
            return int(node.text)
        _, expr, _ = children
        return expr
    def visit_Power(self, node, children):
        if len(children) == 1:
            return children[0]
        pow = sum(children[1:])
        return children[0] ** pow
    def visit_Product(self, node, children):
        if len(children) == 1:
            return children[0]
        prod = children[0]
        for i in range(1, len(children), 2):
            if children[i] == '*':
                prod *= children[i+1]
            else:
                prod /= children[i+1]
        return prod
    def visit_Sum(self, node, children):
        if len(children) == 1:
            return children[0]
        sum = children[0]
        for i in range(1, len(children), 2):
            if children[i] == '+':
                sum += children[i+1]
            else:
                sum -= children[i+1]
        return sum
    def visit_Expr(self, node, children):
        return children[0]

@given(
    a=st.integers(min_value=0, max_value=3),
    a_op=st.sampled_from(['+', '-', '*', '/', '**']),
    b=st.integers(min_value=0, max_value=3),
    b_op=st.sampled_from(['+', '-', '*', '/', '**']),
    c=st.integers(min_value=0, max_value=3),
    c_op=st.sampled_from(['+', '-', '*', '/', '**']),
    d=st.integers(min_value=0, max_value=3),
)
def test_math_formula(a, a_op, b, b_op, c, c_op, d):
    input = f'{a}{a_op}{b}{b_op}{c}{c_op}{d}'

    gram = Grammar(MathGrammar)
    result = gram.parse(input)
    assert result

    visitor = MathVisitor(gram, input)
    try:
        expected = eval(input)

    except ZeroDivisionError:
        try:
            visitor.visit(result)
            pytest.fail('ZeroDivisionError not raised')
        except ZeroDivisionError:
            pass

    else:
        value = visitor.visit(result)
        assert value == expected

JSON = r"""
value <- object / array / string / number / true / false / null
object <- "{" whitespace* (item whitespace* ("," whitespace* item)*)? whitespace* "}"
item <- string whitespace* ":" whitespace* value
array <- "[" whitespace* (value whitespace* ("," whitespace* value)*)? whitespace* "]"
string <- "\"" ([\040-\041] / [\043-\133] / [\135-\377] / escape )* "\""
true <- "true"
false <- "false"
null <- "null"
number <- minus? integral fractional? exponent?
escape <- "\\" ("\"" / "/" / "\\" / "b" / "f" / "n" / "r" / "t" / unicode)
unicode <- "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
minus <- "-"
plus <- "+"
integral <- "0" / [1-9] [0-9]*
fractional <- "." [0-9]+
exponent <- ("e" / "E") (plus / minus)? [0-9]+
whitespace <- " " / "\r" / "\n" / "\t"
"""
json_gram = Grammar(JSON)

@given(
    jsondata=st.recursive(
        st.none() | st.booleans() | st.floats(allow_nan=False, allow_infinity=False, width=32) | st.text(string.printable),
        lambda children: st.lists(children) | st.dictionaries(st.text(string.printable), children),
        max_leaves=5,
    )
)
def test_json(jsondata):
    input = json.dumps(jsondata, allow_nan=False)
    assert json_gram.parse(input, rule='value')

