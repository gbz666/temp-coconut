"""Compare depth-matched and four-step readout on original-target test queries."""

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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/kv_frontier"
DEFAULT_OUTPUT = ROOT / "results/figures/native_vs_four_steps"
DEFAULT_SOURCE_DATA = ROOT / "results/source_data/native_baseline.csv"
MODELS = (
    ("2-layer", "2layer", "#4C72B0"),
    ("6-layer", "6layer", "#DD8452"),
)
DEPTHS = (3, 4, "All")
BUDGETS = ("depth-matched", "four-steps")


def read_rows(folder: Path) -> list[dict]:
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["complete"] and metadata["steps"] == 4
    with (folder / "originals.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 838
    assert len({row["graph_id"] for row in rows}) == 419
    assert {row["original_depth"] for row in rows} == {3, 4}
    return rows


def estimate(rows: list[dict], budget: str) -> dict:
    by_graph: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        steps = row["original_depth"] if budget == "depth-matched" else 4
        by_graph[row["graph_id"]].append(int(row["results"][f"clean_C{steps}_all"]["correct"]))
    # Preserve first-seen graph order to reproduce the existing report's bootstrap.
    graph_values = np.asarray([np.mean(items) for items in by_graph.values()])
    assert all(len(items) == 2 for items in by_graph.values())
    rng = np.random.default_rng(20260920)
    n = len(graph_values)
    weights = rng.multinomial(n, np.ones(n) / n, size=2000) / n
    bootstrap = weights @ graph_values
    return {
        "accuracy": float(graph_values.mean() * 100),
        "ci_low": float(np.quantile(bootstrap, .025) * 100),
        "ci_high": float(np.quantile(bootstrap, .975) * 100),
        "n_graphs": n,
        "n_queries": len(rows),
        "n_correct": int(sum(sum(items) for items in by_graph.values())),
    }


def write_data(path: Path, results: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["budget", "model", "target_depth", "accuracy_percent", "ci95_low", "ci95_high", "n_graphs", "n_queries", "n_correct"])
        for budget in BUDGETS:
            for model, _, _ in MODELS:
                for depth in DEPTHS:
                    value = results[(budget, model, depth)]
                    writer.writerow([budget, model, depth, *[value[key] for key in ("accuracy", "ci_low", "ci_high", "n_graphs", "n_queries", "n_correct")]])


def draw(results: dict, output: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.0), sharey=True)
    fig.subplots_adjust(left=.075, right=.985, bottom=.17, top=.79, wspace=.15)
    groups = np.arange(3)
    width = .31
    for ax, budget, title in zip(axes, BUDGETS,
                                 ("A   Depth-matched steps", "B   Four steps for every question")):
        for index, (model, _, color) in enumerate(MODELS):
            values = [results[(budget, model, depth)] for depth in DEPTHS]
            means = np.asarray([value["accuracy"] for value in values])
            lower = np.asarray([value["ci_low"] for value in values])
            upper = np.asarray([value["ci_high"] for value in values])
            xpos = groups + (index - .5) * width
            ax.bar(xpos, means, width, color=color, label=model, zorder=2)
            ax.errorbar(xpos, means, yerr=[means - lower, upper - means], fmt="none",
                        color="#30343B", elinewidth=.8, capsize=2.5, zorder=3)
            for x, mean, high in zip(xpos, means, upper):
                ax.text(x, max(mean, high) + 2.1, f"{mean:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, loc="left", weight="semibold", pad=12)
        ax.set_xticks(groups, ["3 hops", "4 hops", "Overall"])
        ax.set_ylim(0, 111)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis="y", color="#E5E7EB", linewidth=.7, zorder=0)
        ax.tick_params(length=0, pad=5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#98A2B3")
    axes[0].set_ylabel("Full-vocabulary accuracy (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               bbox_to_anchor=(.53, .99), frameon=False)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{extension}"), dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.source_data.parent.mkdir(parents=True, exist_ok=True)
    rows_by_model = {label: read_rows(args.input / dirname) for label, dirname, _ in MODELS}
    assert [[(row["graph_id"], row["target_slot"], row["original_depth"]) for row in rows]
            for rows in rows_by_model.values()][0] == [
                (row["graph_id"], row["target_slot"], row["original_depth"])
                for row in rows_by_model[MODELS[1][0]]]
    results = {}
    for budget in BUDGETS:
        for model, _, _ in MODELS:
            rows = rows_by_model[model]
            for depth in DEPTHS:
                selected = rows if depth == "All" else [row for row in rows if row["original_depth"] == depth]
                results[(budget, model, depth)] = estimate(selected, budget)
    assert np.isclose(results[("depth-matched", "2-layer", "All")]["accuracy"], 97.97136038186158)
    assert np.isclose(results[("depth-matched", "6-layer", "All")]["accuracy"], 61.097852028639615)
    write_data(args.source_data, results)
    draw(results, args.output)
    for budget in BUDGETS:
        print(budget, {model: [round(results[(budget, model, depth)]["accuracy"], 2) for depth in DEPTHS]
                       for model, _, _ in MODELS})


if __name__ == "__main__":
    main()
