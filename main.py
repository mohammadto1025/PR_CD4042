from lex import Lexer
from pars import Parser
from highlighter import SyntaxHighlighter  

code = """
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
"""

lexer = Lexer(code)
tokens = lexer.tokenize()

for tok in tokens:
        print(tok)

parser = Parser(tokens)
ast = parser.parse_program()

print("=" * 50)
print("AST Result:")
print(ast)
print("\nSyntax Errors (if any):")
for err in parser.errors:
    print(err)

highlighter = SyntaxHighlighter(tokens, ast)

print("=" * 50)
print("ANSI Output:")
print(highlighter.to_ansi())

html_file = highlighter.to_html("highlighted.html")
print(f"\nHTML output saved to: {html_file}")