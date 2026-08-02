from lex import Lexer
from pars import Parser
from highlighter import SyntaxHighlighter
from phase2 import PhaseTwoPipeline
from phase3 import PhaseThreePipeline
from cli import CompilerCLI
from metrics import CodeMetricsAnalyzer
from cfgvisual import CFGHTMLRenderer
from callgraphvisual import CallGraphHTMLRenderer


code = """
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main() {
    int result;
    result = factorial(5);
    return result;
}
"""

lexer = Lexer(code, filename="main.c")
tokens = lexer.tokenize()

for tok in tokens:
    print(tok)

parser = Parser(tokens)
ast = parser.parse_program()

print("=" * 50)
print("AST Result:")
print(ast)

print("\nLexical Diagnostics (if any):")
if len(lexer.diagnostics) == 0:
    print("No lexical errors.")
else:
    print(lexer.diagnostics)

print("\nSyntax Diagnostics (if any):")
if len(parser.diagnostics) == 0:
    print("No syntax errors.")
else:
    print(parser.diagnostics)

highlighter = SyntaxHighlighter(tokens, ast)

print("=" * 50)
print("ANSI Output:")
print(highlighter.to_ansi())

html_file = highlighter.to_html("highlighted.html")
print(f"\nHTML output saved to: {html_file}")

phase2_result = PhaseTwoPipeline(
    source=code,
    filename="main.c",
).run(ast)

print("=" * 50)
print("Phase Two Result:")
print(phase2_result.summary())

print("\nScope Tree:")
print(phase2_result.format_scope_tree())

print("\nSemantic and Type Diagnostics:")
if len(phase2_result.diagnostics) == 0:
    print("No diagnostics.")
else:
    print(phase2_result.format_diagnostics())

print("\nCompletion at line 9, column 5:")
completion_items = phase2_result.complete(
    line=9,
    column=5,
)
if len(completion_items) == 0:
    print("No completion items.")
else:
    for item in completion_items:
        print(item)

print("\nHover at line 9, column 14:")
hover_info = phase2_result.hover(
    line=9,
    column=14,
)
if hover_info is None:
    print("No hover information.")
else:
    print(hover_info)

phase3_result = PhaseThreePipeline(
    source=code,
    filename="main.c",
    entry_function="main",
).run(
    program=ast,
    phase2_result=phase2_result,
)

metrics_result = CodeMetricsAnalyzer(
    source=code,
    filename="main.c",
    complexity_threshold=10,
).analyze(
    program=ast,
    graphs=phase3_result.cfg_results,
    call_graph=phase3_result.call_graph,
)

cfg_report = CFGHTMLRenderer().write(
    graphs=phase3_result.cfg_results,
    output_path="cfg_report.html",
    title="Compoiler Control Flow Graphs",
    open_browser=False,
)

callgraph_report = CallGraphHTMLRenderer().write(
    graph=phase3_result.call_graph,
    output_path="callgraph_report.html",
    title="Compoiler Call Graph",
    metrics=metrics_result,
    program=ast,
    source=code,
    open_browser=False,
)

print(f"CFG graphical report saved to: {cfg_report}")
print(f"Call Graph graphical report saved to: {callgraph_report}")

print("=" * 50)
print("Phase Three Result:")
print()
print(phase3_result.format_all())

print("=" * 50)
print("Bonus - Code Metrics:")
print(metrics_result.format())

print("=" * 50)
print("Phase Three - Go-to-Definition:")

definition = phase3_result.goto_definition(
    line=9,
    column=14,
)

if definition is None:
    print("No symbol found.")
else:
    print(definition)

print("=" * 50)
print("Phase Three - Find All References:")

references = phase3_result.find_references(
    line=9,
    column=14,
    include_definition=True,
)

print(
    phase3_result.format_references(
        references
    )
)

print("=" * 50)
print("Phase Three - Safe Rename:")

rename_result = phase3_result.rename(
    line=9,
    column=14,
    new_name="process",
)

print(rename_result.format())

if rename_result.applied:
    print("\nUpdated Source:")
    print(rename_result.updated_source)

print("=" * 50)
print("Starting Interactive CLI...")

cli = CompilerCLI.from_phase3_result(
    result=phase3_result,
    tokens=tokens,
    parser_errors=parser.errors,
    lexer_diagnostics=list(lexer.diagnostics),
    parser_diagnostics=list(parser.diagnostics),
)

cli.run()
