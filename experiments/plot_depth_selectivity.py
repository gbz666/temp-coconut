"""Redraw the checkpoint350 depth figure from saved per-query results.

Panel A uses native full-vocabulary argmax. The baseline and each single-KV
condition share the same four-step trajectory, prompt, and answer position.
Panel B scores each original input vector against nodes at other reachable
BFS depths, excluding the root. Both panels average within graph first.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/depth_selectivity"
DEFAULT_OUTPUT = ROOT / "results/figures/simplified_prosqa_frontier"
DEFAULT_SOURCE_DATA = ROOT / "results/source_data/depth_selectivity.csv"
DEPTHS = (1, 2, 3, 4)
CONDITIONS = ("C4", "only_P1", "only_P2", "only_P3", "only_P4")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def graph_mean_accuracy(rows: list[dict], depth: int, condition: str) -> tuple[float, int, int]:
    by_graph: dict[int, list[float]] = defaultdict(list)
    count = 0
    for row in rows:
        if row["target_depth"] == depth:
            by_graph[row["graph_id"]].append(float(row["results"][condition]["correct"]))
            count += 1
    return float(np.mean([np.mean(values) for values in by_graph.values()])), len(by_graph), count


def auroc(scores: np.ndarray, positive: list[int], negative: list[int]) -> float:
    if not positive or not negative:
        return float("nan")
    delta = scores[np.asarray(positive)[:, None]] - scores[np.asarray(negative)[None, :]]
    return float(np.mean((delta > 0) + 0.5 * (delta == 0)))


def representation_matrix(originals: list[dict], graphs: list[dict], selected: set[int]) -> tuple[np.ndarray, np.ndarray]:
    by_graph: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in originals:
        graph_id = row["graph_id"]
        if graph_id not in selected:
            continue
        graph = graphs[graph_id]
        distances = {int(node): depth for node, depth in graph["distances"].items()}
        reachable = [node for node, depth in distances.items() if depth > 0]
        cells = np.full((4, 4), np.nan)
        for time in range(4):
            scores = np.asarray(row["node_cosine"][time])
            for depth in DEPTHS:
                positive = [node for node in reachable if distances[node] == depth]
                negative = [node for node in reachable if distances[node] != depth]
                cells[time, depth - 1] = auroc(scores, positive, negative)
        by_graph[graph_id].append(cells)
    assert set(by_graph) == selected
    graph_cells = np.asarray([np.mean(by_graph[graph_id], axis=0) for graph_id in sorted(by_graph)])
    return np.mean(graph_cells, axis=0), np.full((4, 4), len(graph_cells), dtype=int)


def write_source_data(path: Path, accuracy: np.ndarray, auc: np.ndarray, n_queries: list[int], n_graphs: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["panel", "target_bfs_depth", "condition", "value", "unit", "n_graphs", "n_queries"])
        for row, depth in enumerate(DEPTHS):
            for col, condition in enumerate(CONDITIONS):
                writer.writerow(["A", depth, condition, f"{accuracy[row, col]:.12g}", "percent", n_graphs, n_queries[row]])
        for time in DEPTHS:
            for depth in DEPTHS:
                writer.writerow(["B", depth, f"z{time}", f"{auc[time - 1, depth - 1]:.12g}", "AUROC", n_graphs, ""])


def add_labels(ax: plt.Axes, matrix: np.ndarray, image, decimals: int) -> None:
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            r, g, b, _ = image.cmap(image.norm(value))
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(col, row, f"{value:.{decimals}f}", ha="center", va="center",
                    color="white" if luminance < 0.48 else "#152536", fontsize=10,
                    fontweight="semibold" if row == col - 1 and col > 0 else "normal")


def make_figure(accuracy: np.ndarray, auc: np.ndarray, output: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": 0.8, "axes.edgecolor": "#667085",
    })
    fig = plt.figure(figsize=(11.8, 4.0), facecolor="white")
    axes = [fig.add_axes([.065, .22, .405, .65]), fig.add_axes([.575, .22, .325, .65])]
    colorbars = [fig.add_axes([.485, .22, .013, .65]), fig.add_axes([.915, .22, .013, .65])]

    left = axes[0]
    image_a = left.imshow(accuracy, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    add_labels(left, accuracy, image_a, 1)
    left.add_patch(Rectangle((-.49, -.49), .98, 3.98, fill=False, edgecolor="#25324A", linewidth=1.8))
    left.axvline(.5, color="white", linewidth=3.0)
    left.set_xticks(range(5), ["4-step baseline\n(all KV)", "P1 only", "P2 only", "P3 only", "P4 only"])
    left.set_yticks(range(4), [str(depth) for depth in DEPTHS])
    left.set_ylabel("Target BFS depth")
    left.set_xlabel("Latent KV visible to the answer position", labelpad=8)
    left.set_title("A   Full-vocabulary accuracy", loc="left", weight="semibold", pad=12)
    fig.colorbar(image_a, cax=colorbars[0], ticks=[0, 50, 100])

    right = axes[1]
    image_c = right.imshow(auc, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    add_labels(right, auc, image_c, 2)
    right.set_xticks(range(4), [str(depth) for depth in DEPTHS])
    right.set_yticks(range(4), [str(time) for time in DEPTHS])
    right.set_ylabel(r"Input vector $z_t$")
    right.set_xlabel("BFS depth of scored nodes", labelpad=8)
    right.set_title("B   Frontier separability", loc="left", weight="semibold", pad=12)
    cb_c = fig.colorbar(image_c, cax=colorbars[1], ticks=[0, .5, 1])
    cb_c.set_label("Cosine AUROC")

    for ax in axes:
        ax.tick_params(length=0, pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{extension}"), dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args()
    metadata = json.loads((args.input / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["complete"]
    queries = read_jsonl(args.input / "queries.jsonl")
    originals = read_jsonl(args.input / "originals.jsonl")
    graphs = json.loads((args.input / "graphs.json").read_text(encoding="utf-8"))
    selected = set(metadata["selected_graphs"])
    accuracy = np.empty((4, 5))
    counts = []
    for row, depth in enumerate(DEPTHS):
        for col, condition in enumerate(CONDITIONS):
            value, n_graphs, n_queries = graph_mean_accuracy(queries, depth, condition)
            assert n_graphs == len(selected)
            accuracy[row, col] = value * 100
        counts.append(n_queries)
    auc, auc_counts = representation_matrix(originals, graphs, selected)
    assert np.all(auc_counts == len(selected))
    assert sum(counts) == len(queries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.source_data.parent.mkdir(parents=True, exist_ok=True)
    write_source_data(args.source_data, accuracy, auc, counts, len(selected))
    make_figure(accuracy, auc, args.output)
    print("Baseline (full-vocabulary %):", np.round(accuracy[:, 0], 2).tolist())
    print("Output:", args.output)


if __name__ == "__main__":
    main()
