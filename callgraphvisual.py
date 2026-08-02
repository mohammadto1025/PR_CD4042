from __future__ import annotations

from collections import defaultdict, deque
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, TYPE_CHECKING
import re
import webbrowser

from callgraph import CallGraph, CallSite
from pars import FuncDecl, Program

if TYPE_CHECKING:
    from metrics import CodeMetricsResult, FunctionMetrics


Point = Tuple[float, float]


class CallGraphHTMLRenderer:
    """Render a CallGraph as a standalone interactive HTML/SVG report.

    The renderer does not rebuild or change the call graph. It visualizes the
    exact nodes, edges, recursion, dead functions, call sites, and source
    locations already produced by ``callgraphkon.py``.
    """

    def __init__(
        self,
        node_width: int = 230,
        node_height: int = 92,
        horizontal_gap: int = 120,
        vertical_gap: int = 72,
        page_padding: int = 80,
    ) -> None:
        if node_width < 120 or node_height < 60:
            raise ValueError("node dimensions are too small")
        if horizontal_gap < 30 or vertical_gap < 20:
            raise ValueError("graph gaps must be positive")
        if page_padding < 20:
            raise ValueError("page_padding must be at least 20")

        self.node_width = node_width
        self.node_height = node_height
        self.horizontal_gap = horizontal_gap
        self.vertical_gap = vertical_gap
        self.page_padding = page_padding

    def render(
        self,
        graph: CallGraph,
        title: str = "Call Graph Report",
        *,
        metrics: Optional["CodeMetricsResult"] = None,
        program: Optional[Program] = None,
        source: Optional[str] = None,
    ) -> str:
        if not isinstance(graph, CallGraph):
            raise TypeError("graph must be a CallGraph")
        if program is not None and not isinstance(program, Program):
            raise TypeError("program must be a Program")
        if source is not None and not isinstance(source, str):
            raise TypeError("source must be a string")

        functions = self._function_map(program)
        source_lines = source.splitlines() if source is not None else []
        svg = self._render_svg(graph, metrics, functions, bool(source_lines))
        details = self._render_details(graph, metrics, functions)
        source_panel = self._render_source(source_lines) if source_lines else ""

        safe_title = escape(title)
        entry_exists = graph.entry_function in graph.nodes
        entry_note = (
            f"Entry function: {escape(graph.entry_function)}"
            if entry_exists
            else (
                "Entry function "
                f"<code>{escape(graph.entry_function)}</code> is not defined."
            )
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{safe_title}</title>
<style>
:root {{
    color-scheme: light;
    --page-bg: #f5f7fb;
    --panel-bg: #ffffff;
    --text: #172033;
    --muted: #66728b;
    --border: #d7deeb;
    --node: #eef4ff;
    --entry: #dff7e8;
    --recursive: #f2eaff;
    --dead: #f2f2f2;
    --external: #fff0f0;
    --edge: #34425f;
    --recursive-edge: #7451b8;
    --dead-edge: #7a7f89;
    --unresolved-edge: #b83b3b;
    --accent: #315fa8;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
    margin: 0;
    background: var(--page-bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
}}
header {{ padding: 30px 34px 18px; }}
header h1 {{ margin: 0 0 8px; font-size: 28px; }}
header p {{ margin: 4px 0; color: var(--muted); }}
code {{ font-family: "SFMono-Regular", Consolas, monospace; }}
.legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px 18px;
    margin-top: 18px;
    font-size: 13px;
}}
.legend-item {{ display: inline-flex; align-items: center; gap: 7px; }}
.legend-shape {{
    width: 20px;
    height: 13px;
    border: 2px solid #53617a;
    border-radius: 5px;
    background: var(--node);
}}
.legend-shape.entry {{ background: var(--entry); border-radius: 50%; }}
.legend-shape.recursive {{ background: var(--recursive); border-color: #7451b8; }}
.legend-shape.dead {{ background: var(--dead); border-style: dashed; }}
.legend-shape.external {{ background: var(--external); border-style: dashed; border-color: #b83b3b; }}
main {{ padding: 0 24px 42px; }}
.panel {{
    margin: 18px auto;
    max-width: 1700px;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: 0 8px 28px rgba(31, 45, 75, 0.08);
    overflow: hidden;
}}
.panel-title {{
    margin: 0;
    padding: 18px 22px;
    font-size: 20px;
    border-bottom: 1px solid var(--border);
}}
.canvas {{ overflow: auto; padding: 12px; }}
svg {{ display: block; margin: 0 auto; min-width: 760px; }}
.group-label {{
    font-size: 14px;
    font-weight: 700;
    fill: var(--muted);
    letter-spacing: 0.03em;
}}
.cg-node {{ cursor: pointer; outline: none; }}
.cg-node .node-shape {{
    fill: var(--node);
    stroke: #53617a;
    stroke-width: 2;
    transition: filter 120ms ease, stroke-width 120ms ease;
}}
.cg-node:hover .node-shape,
.cg-node:focus .node-shape {{
    filter: drop-shadow(0 5px 7px rgba(49, 95, 168, 0.22));
    stroke-width: 3;
}}
.cg-node.entry .node-shape {{ fill: var(--entry); }}
.cg-node.recursive .node-shape {{ fill: var(--recursive); stroke: #7451b8; stroke-width: 3; }}
.cg-node.dead .node-shape {{ fill: var(--dead); stroke: #7a7f89; stroke-dasharray: 8 6; }}
.cg-node.external .node-shape {{ fill: var(--external); stroke: #b83b3b; stroke-dasharray: 8 6; }}
.node-name {{
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 16px;
    font-weight: 750;
    fill: var(--text);
    text-anchor: middle;
}}
.node-subtitle {{
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 11px;
    fill: var(--muted);
    text-anchor: middle;
}}
.node-badge {{
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 10px;
    font-weight: 700;
    fill: var(--accent);
    text-anchor: middle;
}}
.edge {{ fill: none; stroke: var(--edge); stroke-width: 2; }}
.edge.recursive {{ stroke: var(--recursive-edge); stroke-width: 2.5; }}
.edge.dead {{ stroke: var(--dead-edge); stroke-dasharray: 7 5; }}
.edge.unresolved {{ stroke: var(--unresolved-edge); stroke-dasharray: 7 5; }}
.edge-label {{
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 11px;
    font-weight: 700;
    fill: var(--edge);
    paint-order: stroke;
    stroke: white;
    stroke-width: 5px;
    stroke-linejoin: round;
    text-anchor: middle;
}}
.edge-label.recursive {{ fill: var(--recursive-edge); }}
.edge-label.dead {{ fill: var(--dead-edge); }}
.edge-label.unresolved {{ fill: var(--unresolved-edge); }}
.summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    padding: 20px 22px;
}}
.summary-item {{
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    background: #fbfcff;
}}
.summary-item strong {{ display: block; font-size: 24px; }}
.summary-item span {{ color: var(--muted); font-size: 12px; }}
.details-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 14px;
    padding: 0 22px 22px;
}}
.function-card {{
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    scroll-margin-top: 18px;
}}
.function-card.dead {{ border-style: dashed; background: #fafafa; }}
.function-card.recursive {{ border-color: #b8a4dd; background: #fcfaff; }}
.function-card h3 {{ margin: 0 0 7px; font-size: 17px; }}
.signature {{
    overflow-wrap: anywhere;
    color: #274f8c;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 13px;
}}
.meta {{ margin-top: 10px; color: var(--muted); font-size: 13px; line-height: 1.55; }}
.badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
.badge {{
    padding: 3px 8px;
    border-radius: 999px;
    background: #eaf0fa;
    font-size: 11px;
    font-weight: 700;
}}
.badge.recursive {{ background: #eee5ff; color: #61409e; }}
.badge.dead {{ background: #ececec; color: #555; }}
.badge.entry {{ background: #dcf5e5; color: #18683f; }}
.source-wrap {{ overflow: auto; max-height: 620px; }}
.source-code {{
    margin: 0;
    min-width: max-content;
    padding: 16px 0;
    background: #111827;
    color: #e5e7eb;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 13px;
    line-height: 1.55;
}}
.source-line {{ display: block; padding: 0 18px; scroll-margin-top: 18px; }}
.source-line:target {{ background: #334155; }}
.line-number {{
    display: inline-block;
    width: 48px;
    margin-right: 16px;
    color: #8290a8;
    text-align: right;
    user-select: none;
}}
.empty {{ padding: 32px; color: var(--muted); text-align: center; }}
footer {{ color: var(--muted); font-size: 12px; text-align: center; padding: 0 20px 28px; }}
</style>
</head>
<body>
<header>
    <h1>{safe_title}</h1>
    <p>Graphical static call analysis generated from the program AST and symbol bindings.</p>
    <p>{entry_note}</p>
    <div class="legend">
        <span class="legend-item"><span class="legend-shape entry"></span>Entry</span>
        <span class="legend-item"><span class="legend-shape"></span>Function</span>
        <span class="legend-item"><span class="legend-shape recursive"></span>Recursive</span>
        <span class="legend-item"><span class="legend-shape dead"></span>Dead / unreachable</span>
        <span class="legend-item"><span class="legend-shape external"></span>Unresolved external call</span>
    </div>
</header>
<main>
<section class="panel">
    <h2 class="panel-title">Program Call Graph</h2>
    <div class="canvas">{svg}</div>
</section>
<section class="panel">
    <h2 class="panel-title">Function Details</h2>
    {details}
</section>
{source_panel}
</main>
<footer>Generated by CallGraphHTMLRenderer</footer>
</body>
</html>
"""

    def write(
        self,
        graph: CallGraph,
        output_path: str | Path = "callgraph_report.html",
        title: str = "Call Graph Report",
        *,
        metrics: Optional["CodeMetricsResult"] = None,
        program: Optional[Program] = None,
        source: Optional[str] = None,
        open_browser: bool = False,
    ) -> Path:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self.render(
                graph,
                title=title,
                metrics=metrics,
                program=program,
                source=source,
            ),
            encoding="utf-8",
        )

        if open_browser:
            webbrowser.open(path.as_uri())

        return path

    render_to_file = write

    def _render_svg(
        self,
        graph: CallGraph,
        metrics: Optional["CodeMetricsResult"],
        functions: Mapping[str, FuncDecl],
        has_source: bool,
    ) -> str:
        external_names = sorted(
            {
                site.callee
                for site in graph.unresolved_call_sites
                if site.callee not in graph.nodes
            }
        )
        positions, width, height, labels = self._layout(graph, external_names)
        recursive = set(graph.recursive_functions())
        dead = set(graph.dead_functions())
        recursive_edges = self._recursive_edges(graph)
        edge_counts = self._edge_call_counts(graph.call_sites)

        marker_defs = """
<defs>
  <marker id="cg-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#34425f"></path>
  </marker>
  <marker id="cg-arrow-recursive" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#7451b8"></path>
  </marker>
  <marker id="cg-arrow-dead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#7a7f89"></path>
  </marker>
  <marker id="cg-arrow-unresolved" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#b83b3b"></path>
  </marker>
</defs>
"""

        label_parts = [
            f'<text class="group-label" x="{x:.1f}" y="{y:.1f}">{escape(text)}</text>'
            for text, x, y in labels
        ]

        edge_parts: List[str] = []
        for caller in sorted(graph.nodes):
            for callee in graph.direct_callees(caller):
                path, label = self._edge_geometry(caller, callee, positions)
                is_recursive = (caller, callee) in recursive_edges
                is_dead = caller in dead and callee in dead
                css = "recursive" if is_recursive else ("dead" if is_dead else "")
                marker = (
                    "cg-arrow-recursive"
                    if is_recursive
                    else ("cg-arrow-dead" if is_dead else "cg-arrow")
                )
                count = edge_counts.get((caller, callee), 1)
                count_text = "1 call" if count == 1 else f"{count} calls"
                edge_parts.append(
                    f'<path class="edge {css}" d="{path}" marker-end="url(#{marker})"></path>'
                    f'<text class="edge-label {css}" x="{label[0]:.1f}" y="{label[1]:.1f}">{count_text}</text>'
                )

        for site in sorted(graph.unresolved_call_sites, key=self._site_key):
            if site.callee not in positions:
                continue
            path, label = self._edge_geometry(site.caller, site.callee, positions)
            edge_parts.append(
                f'<path class="edge unresolved" d="{path}" marker-end="url(#cg-arrow-unresolved)"></path>'
                f'<text class="edge-label unresolved" x="{label[0]:.1f}" y="{label[1]:.1f}">unresolved</text>'
            )

        node_parts: List[str] = []
        for name in sorted(graph.nodes):
            x, y = positions[name]
            function = functions.get(name)
            signature = self._signature(name, function)
            node_metrics = self._metrics_for(metrics, name)
            callers = graph.direct_callers(name)
            callees = graph.direct_callees(name)
            location = graph.nodes[name].definition_loc

            classes = ["cg-node"]
            badges: List[str] = []
            if name == graph.entry_function:
                classes.append("entry")
                badges.append("ENTRY")
            if name in recursive:
                classes.append("recursive")
                badges.append("RECURSIVE")
            if name in dead:
                classes.append("dead")
                badges.append("DEAD")

            subtitle = self._node_subtitle(node_metrics, callers, callees)
            tooltip = self._tooltip(
                signature=signature,
                location=str(location),
                callers=callers,
                callees=callees,
                metrics=node_metrics,
                statuses=badges,
            )
            target = (
                f"#source-line-{location.line}"
                if has_source
                else f"#function-detail-{self._slug(name)}"
            )

            node_parts.append(
                self._node_svg(
                    name=name,
                    x=x,
                    y=y,
                    classes=" ".join(classes),
                    signature=signature,
                    subtitle=subtitle,
                    badge_text=" • ".join(badges),
                    tooltip=tooltip,
                    target=target,
                )
            )

        for name in external_names:
            x, y = positions[name]
            node_parts.append(
                self._node_svg(
                    name=name,
                    x=x,
                    y=y,
                    classes="cg-node external",
                    signature=f"{name}(...) [unresolved]",
                    subtitle="external / undefined",
                    badge_text="UNRESOLVED",
                    tooltip=f"Unresolved call target: {name}",
                    target="#function-details",
                )
            )

        empty = ""
        if not graph.nodes and not external_names:
            empty = '<text class="group-label" x="80" y="100">No functions found.</text>'

        return (
            f'<svg viewBox="0 0 {width:.0f} {height:.0f}" '
            f'width="{width:.0f}" height="{height:.0f}" '
            'role="img" aria-label="Program call graph">'
            f"{marker_defs}{''.join(label_parts)}{''.join(edge_parts)}"
            f"{''.join(node_parts)}{empty}</svg>"
        )

    def _render_details(
        self,
        graph: CallGraph,
        metrics: Optional["CodeMetricsResult"],
        functions: Mapping[str, FuncDecl],
    ) -> str:
        recursive = set(graph.recursive_functions())
        dead = set(graph.dead_functions())
        unresolved = graph.unresolved_call_sites

        summary = (
            '<div class="summary-grid">'
            f'<div class="summary-item"><strong>{len(graph.nodes)}</strong><span>Functions</span></div>'
            f'<div class="summary-item"><strong>{graph.edge_count()}</strong><span>Resolved edges</span></div>'
            f'<div class="summary-item"><strong>{len(graph.call_sites)}</strong><span>Call sites</span></div>'
            f'<div class="summary-item"><strong>{len(recursive)}</strong><span>Recursive functions</span></div>'
            f'<div class="summary-item"><strong>{len(dead)}</strong><span>Dead functions</span></div>'
            f'<div class="summary-item"><strong>{len(unresolved)}</strong><span>Unresolved calls</span></div>'
            '</div>'
        )

        cards: List[str] = []
        for name in sorted(graph.nodes):
            node = graph.nodes[name]
            callers = graph.direct_callers(name)
            callees = graph.direct_callees(name)
            function = functions.get(name)
            signature = self._signature(name, function)
            item = self._metrics_for(metrics, name)

            classes = ["function-card"]
            badges: List[str] = []
            if name == graph.entry_function:
                badges.append('<span class="badge entry">ENTRY</span>')
            if name in recursive:
                classes.append("recursive")
                badges.append('<span class="badge recursive">RECURSIVE</span>')
            if name in dead:
                classes.append("dead")
                badges.append('<span class="badge dead">DEAD</span>')

            callers_text = ", ".join(callers) if callers else "None"
            callees_text = ", ".join(callees) if callees else "None"
            metrics_text = self._metrics_detail(item)
            source_link = (
                f'<a href="#source-line-{node.definition_loc.line}">Go to source definition</a>'
            )

            cards.append(
                f'<article id="function-detail-{self._slug(name)}" class="{" ".join(classes)}">'
                f'<h3>{escape(name)}</h3>'
                f'<div class="signature">{escape(signature)}</div>'
                f'<div class="badges">{"".join(badges)}</div>'
                '<div class="meta">'
                f'Defined at: {escape(str(node.definition_loc))}<br>'
                f'Direct callers ({len(callers)}): {escape(callers_text)}<br>'
                f'Direct callees ({len(callees)}): {escape(callees_text)}<br>'
                f'Call sites from function: {len(graph.call_sites_from(name))}<br>'
                f'{metrics_text}{source_link}'
                '</div>'
                '</article>'
            )

        if unresolved:
            unresolved_items = "".join(
                f"<li>{escape(str(site))}</li>"
                for site in sorted(unresolved, key=self._site_key)
            )
            cards.append(
                '<article class="function-card dead">'
                '<h3>Unresolved call sites</h3>'
                f'<ul class="meta">{unresolved_items}</ul>'
                '</article>'
            )

        if not cards:
            cards.append('<div class="empty">No functions found.</div>')

        return summary + f'<div id="function-details" class="details-grid">{"".join(cards)}</div>'

    def _render_source(self, lines: Sequence[str]) -> str:
        rendered = []
        for number, line in enumerate(lines, start=1):
            rendered.append(
                f'<span id="source-line-{number}" class="source-line">'
                f'<span class="line-number">{number}</span>'
                f'{escape(line) if line else " "}'
                '</span>'
            )

        return (
            '<section class="panel">'
            '<h2 class="panel-title">Source Definitions</h2>'
            '<div class="source-wrap">'
            f'<pre class="source-code">{"".join(rendered)}</pre>'
            '</div>'
            '</section>'
        )

    def _layout(
        self,
        graph: CallGraph,
        external_names: Sequence[str],
    ) -> Tuple[Dict[str, Point], float, float, List[Tuple[str, float, float]]]:
        positions: Dict[str, Point] = {}
        labels: List[Tuple[str, float, float]] = []
        dead = set(graph.dead_functions())
        reachable = [name for name in sorted(graph.nodes) if name not in dead]
        dead_names = [name for name in sorted(graph.nodes) if name in dead]

        current_y = float(self.page_padding + 45)
        max_x = float(self.page_padding + self.node_width)

        if reachable:
            labels.append(("REACHABLE FROM ENTRY", float(self.page_padding), current_y - 24))
            levels = self._reachable_levels(graph, reachable)
            layer_map: Dict[int, List[str]] = defaultdict(list)
            for name in reachable:
                layer_map[levels.get(name, 0)].append(name)

            max_layer_size = max(len(names) for names in layer_map.values())
            reachable_height = max(
                self.node_height,
                max_layer_size * self.node_height
                + max(0, max_layer_size - 1) * self.vertical_gap,
            )

            for layer in sorted(layer_map):
                names = sorted(layer_map[layer])
                column_height = (
                    len(names) * self.node_height
                    + max(0, len(names) - 1) * self.vertical_gap
                )
                start_y = current_y + (reachable_height - column_height) / 2
                x = float(self.page_padding + layer * (self.node_width + self.horizontal_gap))
                for index, name in enumerate(names):
                    y = start_y + index * (self.node_height + self.vertical_gap)
                    positions[name] = (x, y)
                    max_x = max(max_x, x + self.node_width)

            current_y += reachable_height + self.vertical_gap + 80

        if dead_names:
            labels.append(("DEAD / UNREACHABLE FROM ENTRY", float(self.page_padding), current_y - 24))
            columns = min(4, max(1, len(dead_names)))
            for index, name in enumerate(dead_names):
                column = index % columns
                row = index // columns
                x = float(self.page_padding + column * (self.node_width + self.horizontal_gap))
                y = current_y + row * (self.node_height + self.vertical_gap)
                positions[name] = (x, y)
                max_x = max(max_x, x + self.node_width)
            rows = (len(dead_names) + columns - 1) // columns
            current_y += rows * self.node_height + max(0, rows - 1) * self.vertical_gap + 80

        if external_names:
            labels.append(("UNRESOLVED CALL TARGETS", float(self.page_padding), current_y - 24))
            columns = min(4, max(1, len(external_names)))
            for index, name in enumerate(external_names):
                column = index % columns
                row = index // columns
                x = float(self.page_padding + column * (self.node_width + self.horizontal_gap))
                y = current_y + row * (self.node_height + self.vertical_gap)
                positions[name] = (x, y)
                max_x = max(max_x, x + self.node_width)
            rows = (len(external_names) + columns - 1) // columns
            current_y += rows * self.node_height + max(0, rows - 1) * self.vertical_gap + 50

        if not positions:
            return {}, 900.0, 220.0, labels

        width = max(900.0, max_x + self.page_padding)
        height = max(260.0, current_y + self.page_padding / 2)
        return positions, width, height, labels

    def _reachable_levels(
        self,
        graph: CallGraph,
        reachable: Sequence[str],
    ) -> Dict[str, int]:
        reachable_set = set(reachable)
        entry = graph.entry_function
        if entry not in reachable_set:
            return {name: 0 for name in reachable}

        level: Dict[str, int] = {entry: 0}
        queue = deque([entry])
        while queue:
            caller = queue.popleft()
            next_level = level[caller] + 1
            for callee in graph.direct_callees(caller):
                if callee not in reachable_set or callee == caller:
                    continue
                if callee not in level:
                    level[callee] = next_level
                    queue.append(callee)

        # Any reachable node not assigned because of a non-entry cycle is put
        # after the deepest assigned layer, without changing graph semantics.
        fallback = max(level.values(), default=-1) + 1
        for name in reachable:
            level.setdefault(name, fallback)
        return level

    def _edge_geometry(
        self,
        source: str,
        target: str,
        positions: Mapping[str, Point],
    ) -> Tuple[str, Point]:
        sx, sy = positions[source]
        tx, ty = positions[target]
        w = float(self.node_width)
        h = float(self.node_height)

        if source == target:
            start_x = sx + w
            start_y = sy + h * 0.38
            end_x = sx + w * 0.72
            end_y = sy
            loop_x = sx + w + 72
            loop_y = sy - 58
            path = (
                f"M {start_x:.1f},{start_y:.1f} "
                f"C {loop_x:.1f},{start_y:.1f} {loop_x:.1f},{loop_y:.1f} {sx + w * 0.72:.1f},{loop_y:.1f} "
                f"C {sx + w * 0.50:.1f},{loop_y:.1f} {sx + w * 0.58:.1f},{end_y:.1f} {end_x:.1f},{end_y:.1f}"
            )
            return path, (loop_x - 18, loop_y - 8)

        scx, scy = sx + w / 2, sy + h / 2
        tcx, tcy = tx + w / 2, ty + h / 2

        if tx > sx + 10:
            start = (sx + w, scy)
            end = (tx, tcy)
            delta = max(55.0, (end[0] - start[0]) * 0.48)
            c1 = (start[0] + delta, start[1])
            c2 = (end[0] - delta, end[1])
        elif tx < sx - 10:
            start = (scx, sy)
            end = (tcx, ty)
            lift = max(70.0, abs(tcx - scx) * 0.28)
            c1 = (scx, min(sy, ty) - lift)
            c2 = (tcx, min(sy, ty) - lift)
        else:
            side = 1 if ty >= sy else -1
            start = (sx + w, scy)
            end = (tx + w, tcy)
            bend = max(sx, tx) + w + 70
            c1 = (bend, scy + side * 15)
            c2 = (bend, tcy - side * 15)

        path = (
            f"M {start[0]:.1f},{start[1]:.1f} "
            f"C {c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {end[0]:.1f},{end[1]:.1f}"
        )
        label = self._cubic_point(start, c1, c2, end, 0.5)
        return path, (label[0], label[1] - 8)

    def _node_svg(
        self,
        *,
        name: str,
        x: float,
        y: float,
        classes: str,
        signature: str,
        subtitle: str,
        badge_text: str,
        tooltip: str,
        target: str,
    ) -> str:
        safe_name = escape(name)
        safe_signature = escape(signature)
        safe_subtitle = escape(subtitle)
        safe_badge = escape(badge_text)
        safe_tooltip = escape(tooltip)
        safe_target = escape(target, quote=True)

        center_x = x + self.node_width / 2
        name_y = y + 29
        subtitle_y = y + 52
        badge_y = y + 73

        badge = (
            f'<text class="node-badge" x="{center_x:.1f}" y="{badge_y:.1f}">{safe_badge}</text>'
            if badge_text
            else ""
        )

        return (
            f'<a href="{safe_target}">'
            f'<g class="{classes}" tabindex="0" data-function="{safe_name}">'
            f'<title>{safe_tooltip}</title>'
            f'<rect class="node-shape" x="{x:.1f}" y="{y:.1f}" '
            f'width="{self.node_width}" height="{self.node_height}" rx="16" ry="16"></rect>'
            f'<text class="node-name" x="{center_x:.1f}" y="{name_y:.1f}">{safe_name}</text>'
            f'<text class="node-subtitle" x="{center_x:.1f}" y="{subtitle_y:.1f}">{safe_subtitle}</text>'
            f'{badge}'
            f'<desc>{safe_signature}</desc>'
            '</g></a>'
        )

    def _function_map(
        self,
        program: Optional[Program],
    ) -> Dict[str, FuncDecl]:
        if program is None:
            return {}
        return {
            declaration.name: declaration
            for declaration in program.declarations
            if isinstance(declaration, FuncDecl)
        }

    def _signature(
        self,
        name: str,
        function: Optional[FuncDecl],
    ) -> str:
        if function is None:
            return f"{name}(...)"
        params = ", ".join(
            f"{param.type_spec} {param.name}"
            for param in function.params
        )
        return f"{function.return_type} {function.name}({params})"

    def _node_subtitle(
        self,
        metrics: Optional["FunctionMetrics"],
        callers: Sequence[str],
        callees: Sequence[str],
    ) -> str:
        if metrics is not None:
            return (
                f"M={metrics.cyclomatic_complexity}  "
                f"LOC={metrics.lines_of_code}  "
                f"in={len(callers)} out={len(callees)}"
            )
        return f"in={len(callers)}  out={len(callees)}"

    def _tooltip(
        self,
        *,
        signature: str,
        location: str,
        callers: Sequence[str],
        callees: Sequence[str],
        metrics: Optional["FunctionMetrics"],
        statuses: Sequence[str],
    ) -> str:
        lines = [signature, f"Defined at: {location}"]
        lines.append("Callers: " + (", ".join(callers) if callers else "None"))
        lines.append("Callees: " + (", ".join(callees) if callees else "None"))
        if metrics is not None:
            lines.append(
                f"Complexity: {metrics.cyclomatic_complexity}; "
                f"LOC: {metrics.lines_of_code}; "
                f"Statements: {metrics.statement_count}"
            )
        if statuses:
            lines.append("Status: " + ", ".join(statuses))
        return "\n".join(lines)

    def _metrics_for(
        self,
        metrics: Optional["CodeMetricsResult"],
        function_name: str,
    ) -> Optional["FunctionMetrics"]:
        if metrics is None:
            return None
        return metrics.functions.get(function_name)

    def _metrics_detail(
        self,
        item: Optional["FunctionMetrics"],
    ) -> str:
        if item is None:
            return ""
        return (
            f'Cyclomatic complexity: {item.cyclomatic_complexity}<br>'
            f'Lines of code: {item.lines_of_code}<br>'
            f'Statements: {item.statement_count}<br>'
            f'Nesting depth: {item.nesting_depth}<br>'
        )

    def _recursive_edges(self, graph: CallGraph) -> Set[Tuple[str, str]]:
        result: Set[Tuple[str, str]] = set()
        for component in graph.strongly_connected_components():
            members = set(component)
            if len(members) == 1:
                name = component[0]
                if name in graph.direct_callees(name):
                    result.add((name, name))
                continue
            for caller in members:
                for callee in graph.direct_callees(caller):
                    if callee in members:
                        result.add((caller, callee))
        return result

    def _edge_call_counts(
        self,
        sites: Iterable[CallSite],
    ) -> Dict[Tuple[str, str], int]:
        counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for site in sites:
            if site.resolved:
                counts[(site.caller, site.callee)] += 1
        return counts

    @staticmethod
    def _cubic_point(
        start: Point,
        control1: Point,
        control2: Point,
        end: Point,
        t: float,
    ) -> Point:
        mt = 1.0 - t
        x = (
            mt ** 3 * start[0]
            + 3 * mt ** 2 * t * control1[0]
            + 3 * mt * t ** 2 * control2[0]
            + t ** 3 * end[0]
        )
        y = (
            mt ** 3 * start[1]
            + 3 * mt ** 2 * t * control1[1]
            + 3 * mt * t ** 2 * control2[1]
            + t ** 3 * end[1]
        )
        return x, y

    @staticmethod
    def _site_key(site: CallSite) -> Tuple[str, int, int, str]:
        return (
            site.caller,
            site.location.line,
            site.location.column,
            site.callee,
        )

    @staticmethod
    def _slug(value: str) -> str:
        result = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
        return result or "function"
