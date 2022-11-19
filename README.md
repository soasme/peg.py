# peg.py

Peg.py is a Parsing Expression Grammar (PEG) parser written in pure Python.

## Install

You can install peg.py by running:

```bash
$ pip install peg.py
```

## Example Usage

You can build a `Grammar` and call its `parse` method to get the parse tree.

```python
from peg import Grammar

greeter = Grammar("""
    greet   <- "Hello, " someone
    someone <- [a-zA-Z0-9]
""")

input = "Hello, world"
tree = greeter.parse(input)
```

Peg.py provides a tree visitor `NodeVisitor`. You an inherit this class and define your own `visit_` methods to handle your parse tree. Each `visit_` method handles a rule, such as `visit_greet`, `visit_someone`.

```python
class GreeterVisitor(NodeVisitor):

    def visit_greet(self, node, children):
    	return ('Hello', children[0])

    def visit_someone(self, node, _):
        return node.text

visitor = GreeterVisitor(greeter)
result = visitor.visit(tree)

print(result)
```

The program should produce such a result:

```
('Hello', 'world')
```

## Syntax

A **grammar** consists of a set of rules.

A **rule** consists of a name, a left arrow, and a pattern.

```
name <- pattern
```

A **name** starts with alphabets or underscore, followed by alphabets or digits or underscore.

```
valid
Valid
_valid
0nay 	// invalid
```

A **pattern** contains one or more of the elements mentioned below.


A **literal** is a string enclosed in double quotes or single quotes. For example, `"hello"`, `'hello'`. Peg.py matches the input as-is.

A **set** is a set of characters enclosed in square brackets. Any pairs of characters having dash (`-`) in-between represents all characters from the first to the second (inclusive). For example, `[a-z]`, `[a-zA-Z0-9]`, `[a-zA-Z$_]`.

A **dot** (`.`) is for any character, except end-of-text.

`( pattern )` groups pattern with parentheses.

`< pattern >` captures input text associating with an unnamed rule.

`pattern?` means `pattern` is optional.

`pattern+` means `pattern` occuring one or more times.

`pattern*` means `pattern` occuring zero or more times.

`&pattern` checks if pattern matches the input. If so, consume no input. Otherwise, the parse is failed.

`!pattern` checks if pattern does not match the input. If so, consume no input. Otherwise, the parse is failed.

Several patterns can be written one after another, like `pattern1 pattern2 pattern3`. The sequence matches only when each underlying pattern matches.

Several patterns joined by slash (`/`), like `pattern1 / pattern2 / pattern3`, is ordered choices of patterns. The choices matches when any one of the underlying pattern matches.

`# ignored. ` is a comment.

### PEG Grammar For PEG Grammars

```
Grammar         <- Spacing Definition+ EndOfFile

Definition      <- Identifier LEFTARROW Expression
Expression      <- Sequence ( SLASH Sequence )*
Sequence        <- Prefix*
Prefix          <- ( AND / NOT )? Suffix
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
BEGIN           <- '<' Spacing
END             <- '>' Spacing
```

## References

* Peg.py provides identical PEG syntax as described in [Bryan Ford's PEG Paper](https://bford.info/pub/lang/peg.pdf).
* Peg.py implements a simplied VM similar to [lpeg](http://www.inf.puc-rio.br/~roberto/lpeg/).
* Peg.py provides a similar API (`Grammar`, `Grammar.parse()`, `NodeVisitor`, etc) with [parsimonious](https://github.com/erikrose/parsimonious).

