from lex import Lexer  

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