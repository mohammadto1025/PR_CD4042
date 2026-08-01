from lex import Lexer
from pars import Parser  

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