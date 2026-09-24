"""Draw the four-panel back-reference frontier figure from analysis output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np


COLORS = {
    "navy": "#293948",
    "blue": "#245B7A",
    "teal": "#147C7A",
    "gray": "#85919A",
    "light_gray": "#F4F6F7",
    "grid": "#E8EDF0",
    "orange": "#B45F2A",
    "light_orange": "#F9E9DC",
}

def configure_style() -> None:
    """Set publication-friendly typography and editable vector text."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def draw_edge(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    rad: float = 0.0,
    linewidth: float = 1.15,
) -> None:
    """Draw one directed edge in axes coordinates."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=linewidth,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
    )


def draw_schematic(ax: plt.Axes, step: int) -> None:
    """Draw the back-reference graph for latent step 3 or 4."""
    ax.axis("off")

    if step == 3:
        positions = {
            "R": (0.00, 0.45),
            "A": (1.00, 0.72),
            "B": (1.00, -0.25),
            "C": (2.05, 0.72),
            "D": (3.10, 0.72),
        }
        plain_edges = [("R", "A"), ("R", "B"), ("A", "C"), ("C", "D")]
        back_edge = ("C", "B")
        old_node, current_node = "B", "D"
        frontier_labels = {"R": "F0", "A": "F1", "B": "F1", "C": "F2", "D": "F3"}
    elif step == 4:
        positions = {
            "R": (0.00, 0.40),
            "A": (0.75, 0.40),
            "B": (1.55, 0.72),
            "C": (1.55, -0.30),
            "D": (2.35, 0.72),
            "E": (3.15, 0.72),
        }
        plain_edges = [("R", "A"), ("A", "B"), ("A", "C"), ("B", "D"), ("D", "E")]
        back_edge = ("D", "C")
        old_node, current_node = "C", "E"
        frontier_labels = {"R": "F0", "A": "F1", "B": "F2", "C": "F2", "D": "F3", "E": "F4"}
    else:
        raise ValueError("step must be 3 or 4")

    for source, target in plain_edges:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        start = (x1 + 0.13 if x2 > x1 else x1, y1 + 0.03 if y2 > y1 else y1 - 0.03)
        end = (x2 - 0.13 if x2 > x1 else x2, y2 - 0.03 if y2 > y1 else y2 + 0.03)
        draw_edge(ax, start, end, COLORS["gray"])

    source, target = back_edge
    x1, y1 = positions[source]
    x2, y2 = positions[target]
    draw_edge(
        ax,
        (x1 - 0.10, y1 - 0.07),
        (x2 + 0.11, y2 + 0.07),
        COLORS["orange"],
        rad=-0.06,
        linewidth=1.6,
    )

    current_color = COLORS["blue"] if step == 3 else COLORS["teal"]
    for node, (x, y) in positions.items():
        if node == old_node:
            fill, rim = COLORS["light_orange"], COLORS["orange"]
        elif node == current_node:
            fill = rim = current_color
        else:
            fill, rim = COLORS["light_gray"], COLORS["gray"]

        ax.scatter(x, y, s=350, color=fill, edgecolor=rim, linewidth=1, zorder=3)
        ax.text(
            x,
            y,
            node,
            ha="center",
            va="center",
            weight="bold",
            color="white" if node == current_node else COLORS["navy"],
            fontsize=7,
            zorder=4,
        )
        ax.text(x, y - 0.29, frontier_labels[node], ha="center", color=COLORS["gray"], fontsize=6.8)

    ax.set_xlim(-0.25, 3.55)
    ax.set_ylim(-0.75, 1.28)


def display_values_from_summary(
    summary_path: Path,
) -> tuple[dict[int, list[float]], dict[int, float]]:
    """Load the natural-back-reference means used by panels c and d."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metric_keys = ["auc_unreachable", "auc_other_reachable", "auc_back_reference"]
    frontier_auroc = {
        step: [float(summary[str(step)][key]["mean"]) for key in metric_keys]
        for step in (3, 4)
    }
    old_node_decline = {
        step: -float(summary[str(step)]["temporal_delta"]["mean"])
        for step in (3, 4)
    }
    return frontier_auroc, old_node_decline


def build_figure(
    frontier_auroc: dict[int, list[float]],
    old_node_decline: dict[int, float],
) -> plt.Figure:
    """Build the complete 2 x 2 figure."""
    configure_style()
    fig = plt.figure(figsize=(7.5, 5.15), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.16, wspace=0.18)

    graph3 = fig.add_subplot(grid[0, 0])
    graph4 = fig.add_subplot(grid[0, 1])
    bars = fig.add_subplot(grid[1, 0])
    decline = fig.add_subplot(grid[1, 1])

    draw_schematic(graph3, 3)
    draw_schematic(graph4, 4)
    graph3.set_title("a  F1 revisited at step 3", loc="left", weight="bold", fontsize=8.5, pad=5)
    graph4.set_title("b  F2 revisited at step 4", loc="left", weight="bold", fontsize=8.5, pad=5)

    metric_labels = ["Unreachable", "Other reachable\ndepths", "Referenced\nold node"]
    x = np.arange(len(metric_labels))
    case_colors = {3: COLORS["blue"], 4: COLORS["teal"]}

    for step, offset in ((3, -0.18), (4, 0.18)):
        values = frontier_auroc[step]
        rectangles = bars.bar(
            x + offset,
            values,
            width=0.34,
            color=case_colors[step],
            label=f"F{step - 2} → step {step}",
            edgecolor="white",
            linewidth=0.5,
        )
        for rectangle, value in zip(rectangles, values):
            bars.text(
                rectangle.get_x() + rectangle.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=case_colors[step],
            )

    bars.set_xticks(x, metric_labels)
    bars.set_ylim(0, 1.09)
    bars.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    bars.set_ylabel("Current-frontier AUROC")
    bars.set_title("c  Frontier discrimination", loc="left", weight="bold", fontsize=8.5, pad=5)
    bars.legend(
        loc="lower left",
        bbox_to_anchor=(0.43, 1.02),
        ncol=2,
        fontsize=6.8,
        handlelength=1.2,
        columnspacing=0.9,
    )
    bars.grid(axis="y", color=COLORS["grid"], linewidth=0.65)
    bars.set_axisbelow(True)

    for index, step in enumerate((3, 4)):
        value = old_node_decline[step]
        decline.bar(index, value, width=0.52, color=case_colors[step], edgecolor="white", linewidth=0.5)
        decline.text(
            index,
            value + 0.006,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
            color=case_colors[step],
        )

    decline.set_xticks([0, 1], ["F1 → step 3", "F2 → step 4"])
    decline.set_xlim(-0.55, 1.55)
    decline.set_ylim(0, 0.16)
    decline.set_yticks([0, 0.04, 0.08, 0.12, 0.16])
    decline.set_ylabel("Decrease in old-node cosine")
    decline.set_title("d  Old-node score decline", loc="left", weight="bold", fontsize=8.5, pad=5)
    decline.grid(axis="y", color=COLORS["grid"], linewidth=0.65)
    decline.set_axisbelow(True)

    return fig


def save_figure(fig: plt.Figure, output_dir: Path, prefix: str, dpi: int) -> list[Path]:
    """Export editable vector files and a raster preview."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{prefix}.{extension}" for extension in ("svg", "pdf", "png")]
    for path in outputs:
        fig.savefig(path, dpi=dpi if path.suffix == ".png" else None, bbox_inches="tight")
    return outputs


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parents[2] / "manuscript" / "figures"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--prefix", default="back_reference_combined_reproduced")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="summary.json generated by analyze_back_reference_frontier.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = display_values_from_summary(args.summary)
    figure = build_figure(*values)
    outputs = save_figure(figure, args.output_dir, args.prefix, args.dpi)
    plt.close(figure)
    for path in outputs:
        print(path.resolve())


if __name__ == "__main__":
    main()
