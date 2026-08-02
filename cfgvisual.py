from __future__ import annotations

from collections import defaultdict, deque
from html import escape
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import webbrowser

from cfg import BlockKind, ControlFlowGraph


Point = Tuple[float, float]
NodeBox = Tuple[float, float, float, float]


class CFGHTMLRenderer:
    """Render one or more CFGs as a standalone HTML/SVG report."""

    def __init__(
        self,
        horizontal_gap: int = 90,
        vertical_gap: int = 105,
        page_padding: int = 70,
    ) -> None:
        self.horizontal_gap = horizontal_gap
        self.vertical_gap = vertical_gap
        self.page_padding = page_padding

    def render(
        self,
        graphs: Mapping[str, ControlFlowGraph],
        title: str = "Control Flow Graph Report",
    ) -> str:
        sections: List[str] = []

        if graphs:
            for function_name, graph in graphs.items():
                sections.append(
                    self._render_section(function_name, graph)
                )
        else:
            sections.append(
                '<section class="empty">'
                "<h2>No functions found</h2>"
                "<p>The parsed program does not contain a function.</p>"
                "</section>"
            )

        safe_title = escape(title)

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
    --muted: #65708a;
    --border: #d8deea;
    --entry: #dff7e8;
    --exit: #fde7e7;
    --basic: #eef4ff;
    --condition: #fff4cf;
    --loop: #f2eaff;
    --increment: #e8f7f7;
    --unreachable: #f3f3f3;
    --edge: #34425f;
    --true-edge: #147a4b;
    --false-edge: #b83b3b;
    --back-edge: #7451b8;
}}
* {{
    box-sizing: border-box;
}}
body {{
    margin: 0;
    background: var(--page-bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Arial, sans-serif;
}}
header {{
    padding: 30px 34px 18px;
}}
header h1 {{
    margin: 0 0 8px;
    font-size: 28px;
}}
header p {{
    margin: 0;
    color: var(--muted);
}}
.legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px 18px;
    margin-top: 18px;
    font-size: 13px;
}}
.legend-item {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
}}
.legend-shape {{
    width: 18px;
    height: 12px;
    border: 2px solid #53617a;
    border-radius: 4px;
}}
.legend-shape.entry {{ background: var(--entry); }}
.legend-shape.exit {{ background: var(--exit); }}
.legend-shape.basic {{ background: var(--basic); }}
.legend-shape.condition {{ background: var(--condition); }}
.legend-shape.loop {{ background: var(--loop); }}
.legend-shape.unreachable {{
    background: var(--unreachable);
    border-style: dashed;
}}
main {{
    padding: 0 24px 40px;
}}
.cfg-panel {{
    margin: 18px auto;
    max-width: 1500px;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: 0 8px 28px rgba(31, 45, 75, 0.08);
    overflow: hidden;
}}
.cfg-panel h2 {{
    margin: 0;
    padding: 18px 22px;
    font-size: 20px;
    border-bottom: 1px solid var(--border);
}}
.canvas {{
    overflow-x: auto;
    padding: 10px;
}}
svg {{
    display: block;
    margin: 0 auto;
    min-width: 760px;
}}
.node-text {{
    font-family: "SFMono-Regular", Consolas, "Liberation Mono",
                 monospace;
    font-size: 14px;
    fill: var(--text);
    text-anchor: middle;
}}
.node-title {{
    font-weight: 700;
}}
.edge {{
    fill: none;
    stroke: var(--edge);
    stroke-width: 2;
}}
.edge.true {{ stroke: var(--true-edge); }}
.edge.false {{ stroke: var(--false-edge); }}
.edge.back {{ stroke: var(--back-edge); }}
.edge-label {{
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 12px;
    font-weight: 700;
    paint-order: stroke;
    stroke: white;
    stroke-width: 5px;
    stroke-linejoin: round;
    text-anchor: middle;
}}
.edge-label.true {{ fill: var(--true-edge); }}
.edge-label.false {{ fill: var(--false-edge); }}
.edge-label.back {{ fill: var(--back-edge); }}
.edge-label.next,
.edge-label.return {{ fill: var(--edge); }}
.node-shape {{
    stroke: #53617a;
    stroke-width: 2;
}}
.node-entry {{ fill: var(--entry); }}
.node-exit {{ fill: var(--exit); }}
.node-basic {{ fill: var(--basic); }}
.node-condition {{ fill: var(--condition); }}
.node-loop-condition {{ fill: var(--loop); }}
.node-for-increment {{ fill: var(--increment); }}
.node-unreachable {{
    fill: var(--unreachable);
    stroke-dasharray: 8 6;
}}
.empty {{
    max-width: 900px;
    margin: 20px auto;
    padding: 28px;
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
}}
footer {{
    color: var(--muted);
    font-size: 12px;
    text-align: center;
    padding: 0 20px 28px;
}}
</style>
</head>
<body>
<header>
    <h1>{safe_title}</h1>
    <p>Graphical control-flow analysis generated from the program AST.</p>
    <div class="legend">
        <span class="legend-item">
            <span class="legend-shape entry"></span>ENTRY
        </span>
        <span class="legend-item">
            <span class="legend-shape exit"></span>EXIT
        </span>
        <span class="legend-item">
            <span class="legend-shape basic"></span>Basic block
        </span>
        <span class="legend-item">
            <span class="legend-shape condition"></span>Condition
        </span>
        <span class="legend-item">
            <span class="legend-shape loop"></span>Loop
        </span>
        <span class="legend-item">
            <span class="legend-shape unreachable"></span>Unreachable
        </span>
    </div>
</header>
<main>
{''.join(sections)}
</main>
<footer>Generated by CFGHTMLRenderer</footer>
</body>
</html>
"""

    def write(
        self,
        graphs: Mapping[str, ControlFlowGraph],
        output_path: str | Path = "cfg_report.html",
        title: str = "Control Flow Graph Report",
        open_browser: bool = False,
    ) -> Path:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self.render(graphs, title=title),
            encoding="utf-8",
        )

        if open_browser:
            webbrowser.open(path.as_uri())

        return path

    render_to_file = write

    def _render_section(
        self,
        function_name: str,
        graph: ControlFlowGraph,
    ) -> str:
        svg = self._render_svg(graph)
        safe_name = escape(function_name)
        return (
            '<section class="cfg-panel">'
            f"<h2>Function: {safe_name}</h2>"
            f'<div class="canvas">{svg}</div>'
            "</section>"
        )

    def _render_svg(self, graph: ControlFlowGraph) -> str:
        ordered_ids = self._ordered_block_ids(graph)
        levels = self._calculate_levels(graph, ordered_ids)
        node_sizes = {
            block_id: self._node_size(graph, block_id)
            for block_id in ordered_ids
        }

        positions, width, height = self._place_nodes(
            ordered_ids,
            levels,
            node_sizes,
        )

        marker = """
<defs>
  <marker id="arrow"
          markerWidth="10"
          markerHeight="10"
          refX="8"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#34425f"></path>
  </marker>
  <marker id="arrow-true"
          markerWidth="10"
          markerHeight="10"
          refX="8"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#147a4b"></path>
  </marker>
  <marker id="arrow-false"
          markerWidth="10"
          markerHeight="10"
          refX="8"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#b83b3b"></path>
  </marker>
  <marker id="arrow-back"
          markerWidth="10"
          markerHeight="10"
          refX="8"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#7451b8"></path>
  </marker>
</defs>
"""

        edge_parts: List[str] = []
        for index, edge in enumerate(graph.edges):
            if (
                edge.source not in positions
                or edge.target not in positions
            ):
                continue

            path, label_point = self._edge_geometry(
                edge.source,
                edge.target,
                edge.label,
                positions,
                index,
            )
            css_label = self._edge_css_label(edge.label)
            marker_id = self._edge_marker_id(edge.label)
            safe_edge_label = escape(edge.label)

            edge_parts.append(
                f'<path class="edge {css_label}" '
                f'd="{path}" marker-end="url(#{marker_id})"></path>'
            )
            edge_parts.append(
                f'<text class="edge-label {css_label}" '
                f'x="{label_point[0]:.1f}" '
                f'y="{label_point[1]:.1f}">'
                f"{safe_edge_label}</text>"
            )

        unreachable = set(graph.unreachable_block_ids())
        node_parts = [
            self._render_node(
                graph,
                block_id,
                positions[block_id],
                block_id in unreachable,
            )
            for block_id in ordered_ids
        ]

        safe_function = escape(graph.function_name)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" '
            f'role="img" '
            f'aria-label="CFG for {safe_function}">'
            f"{marker}"
            f"{''.join(edge_parts)}"
            f"{''.join(node_parts)}"
            "</svg>"
        )

    def _ordered_block_ids(
        self,
        graph: ControlFlowGraph,
    ) -> List[str]:
        middle = [
            block_id
            for block_id in graph.blocks
            if block_id not in {
                graph.entry_id,
                graph.exit_id,
            }
        ]

        def key(block_id: str) -> Tuple[int, str]:
            if block_id.startswith("B"):
                suffix = block_id[1:]
                if suffix.isdigit():
                    return int(suffix), block_id
            return 10**9, block_id

        middle.sort(key=key)

        result: List[str] = []
        if graph.entry_id in graph.blocks:
            result.append(graph.entry_id)
        result.extend(middle)
        if graph.exit_id in graph.blocks:
            result.append(graph.exit_id)
        return result

    def _calculate_levels(
        self,
        graph: ControlFlowGraph,
        ordered_ids: Sequence[str],
    ) -> Dict[str, int]:
        levels: Dict[str, int] = {graph.entry_id: 0}
        queue = deque([graph.entry_id])

        while queue:
            source = queue.popleft()
            source_level = levels[source]

            for edge in graph.successors(source):
                if edge.label == "back":
                    continue
                if edge.target == source:
                    continue

                proposed = source_level + 1
                current = levels.get(edge.target)

                if current is None or proposed > current:
                    levels[edge.target] = proposed
                    queue.append(edge.target)

        unresolved_reachable = [
            block_id
            for block_id in graph.reachable_block_ids()
            if block_id not in levels
        ]
        next_level = max(levels.values(), default=0) + 1

        for block_id in unresolved_reachable:
            levels[block_id] = next_level
            next_level += 1

        unreachable = set(graph.unreachable_block_ids())
        if unreachable:
            next_level = max(levels.values(), default=0) + 1
            for block_id in ordered_ids:
                if block_id in unreachable:
                    levels[block_id] = next_level
                    next_level += 1

        for block_id in ordered_ids:
            levels.setdefault(block_id, next_level)

        return levels

    def _node_size(
        self,
        graph: ControlFlowGraph,
        block_id: str,
    ) -> Tuple[float, float]:
        block = graph.blocks[block_id]
        lines = [block.block_id] + list(block.statements)

        longest = max((len(line) for line in lines), default=5)
        width = min(max(170.0, longest * 8.4 + 42), 380.0)
        height = max(68.0, 34.0 + len(lines) * 20.0)

        if block.kind in {
            BlockKind.CONDITION,
            BlockKind.LOOP_CONDITION,
        }:
            width = max(width, 230.0)
            height = max(height, 100.0)

        if block.kind in {BlockKind.ENTRY, BlockKind.EXIT}:
            width = max(width, 150.0)
            height = 68.0

        return width, height

    def _place_nodes(
        self,
        ordered_ids: Sequence[str],
        levels: Mapping[str, int],
        node_sizes: Mapping[str, Tuple[float, float]],
    ) -> Tuple[Dict[str, NodeBox], int, int]:
        by_level: Dict[int, List[str]] = defaultdict(list)
        for block_id in ordered_ids:
            by_level[levels[block_id]].append(block_id)

        widest_row = 0.0
        for block_ids in by_level.values():
            row_width = sum(node_sizes[item][0] for item in block_ids)
            row_width += self.horizontal_gap * max(
                0,
                len(block_ids) - 1,
            )
            widest_row = max(widest_row, row_width)

        canvas_width = int(
            max(820.0, widest_row + self.page_padding * 2)
        )

        level_heights = {
            level: max(
                node_sizes[block_id][1]
                for block_id in block_ids
            )
            for level, block_ids in by_level.items()
        }

        sorted_levels = sorted(by_level)
        y_by_level: Dict[int, float] = {}
        current_y = float(self.page_padding)

        for level in sorted_levels:
            y_by_level[level] = current_y
            current_y += (
                level_heights[level] + self.vertical_gap
            )

        positions: Dict[str, NodeBox] = {}

        for level in sorted_levels:
            block_ids = by_level[level]
            row_width = sum(
                node_sizes[block_id][0]
                for block_id in block_ids
            )
            row_width += self.horizontal_gap * max(
                0,
                len(block_ids) - 1,
            )

            current_x = (canvas_width - row_width) / 2.0

            for block_id in block_ids:
                width, height = node_sizes[block_id]
                positions[block_id] = (
                    current_x,
                    y_by_level[level],
                    width,
                    height,
                )
                current_x += width + self.horizontal_gap

        canvas_height = int(
            current_y - self.vertical_gap + self.page_padding
        )
        return positions, canvas_width, canvas_height

    def _render_node(
        self,
        graph: ControlFlowGraph,
        block_id: str,
        box: NodeBox,
        unreachable: bool,
    ) -> str:
        block = graph.blocks[block_id]
        x, y, width, height = box

        class_name = (
            "node-unreachable"
            if unreachable
            else f"node-{block.kind.value}"
        )

        if block.kind in {BlockKind.ENTRY, BlockKind.EXIT}:
            shape = (
                f'<ellipse class="node-shape {class_name}" '
                f'cx="{x + width / 2:.1f}" '
                f'cy="{y + height / 2:.1f}" '
                f'rx="{width / 2:.1f}" '
                f'ry="{height / 2:.1f}"></ellipse>'
            )
        elif block.kind in {
            BlockKind.CONDITION,
            BlockKind.LOOP_CONDITION,
        }:
            points = [
                (x + width / 2, y),
                (x + width, y + height / 2),
                (x + width / 2, y + height),
                (x, y + height / 2),
            ]
            point_text = " ".join(
                f"{px:.1f},{py:.1f}" for px, py in points
            )
            shape = (
                f'<polygon class="node-shape {class_name}" '
                f'points="{point_text}"></polygon>'
            )
        else:
            shape = (
                f'<rect class="node-shape {class_name}" '
                f'x="{x:.1f}" y="{y:.1f}" '
                f'width="{width:.1f}" height="{height:.1f}" '
                f'rx="12" ry="12"></rect>'
            )

        lines = [block.block_id] + list(block.statements)
        if unreachable:
            lines.append("[unreachable]")

        line_height = 19.0
        start_y = y + height / 2 - (
            (len(lines) - 1) * line_height / 2
        )

        text_parts: List[str] = []
        for index, line in enumerate(lines):
            text_class = (
                "node-text node-title"
                if index == 0
                else "node-text"
            )
            safe_line = escape(line)
            text_parts.append(
                f'<text class="{text_class}" '
                f'x="{x + width / 2:.1f}" '
                f'y="{start_y + index * line_height:.1f}">'
                f"{safe_line}</text>"
            )

        return shape + "".join(text_parts)

    def _edge_geometry(
        self,
        source_id: str,
        target_id: str,
        label: str,
        positions: Mapping[str, NodeBox],
        edge_index: int,
    ) -> Tuple[str, Point]:
        sx, sy, sw, sh = positions[source_id]
        tx, ty, tw, th = positions[target_id]

        source_center_x = sx + sw / 2
        source_center_y = sy + sh / 2
        target_center_x = tx + tw / 2
        target_center_y = ty + th / 2

        if label == "back" or target_center_y <= source_center_y:
            lane_x = min(sx, tx) - 45 - (edge_index % 3) * 24
            start = (sx, source_center_y)
            end = (tx, target_center_y)
            path = (
                f"M {start[0]:.1f} {start[1]:.1f} "
                f"C {lane_x:.1f} {start[1]:.1f}, "
                f"{lane_x:.1f} {end[1]:.1f}, "
                f"{end[0]:.1f} {end[1]:.1f}"
            )
            label_point = (
                lane_x - 4,
                (start[1] + end[1]) / 2 - 6,
            )
            return path, label_point

        start = (source_center_x, sy + sh)
        end = (target_center_x, ty)
        middle_y = (start[1] + end[1]) / 2

        path = (
            f"M {start[0]:.1f} {start[1]:.1f} "
            f"C {start[0]:.1f} {middle_y:.1f}, "
            f"{end[0]:.1f} {middle_y:.1f}, "
            f"{end[0]:.1f} {end[1]:.1f}"
        )

        label_point = (
            (start[0] + end[0]) / 2,
            middle_y - 7,
        )
        return path, label_point

    def _edge_css_label(self, label: str) -> str:
        if label in {"true", "false", "back", "return"}:
            return label
        return "next"

    def _edge_marker_id(self, label: str) -> str:
        if label == "true":
            return "arrow-true"
        if label == "false":
            return "arrow-false"
        if label == "back":
            return "arrow-back"
        return "arrow"
