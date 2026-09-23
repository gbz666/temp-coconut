"""从已保存的 source data 重绘论文中的两层/六层原生任务基线图。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/source_data/native_baseline.csv"
DEFAULT_OUTPUT = ROOT / "results/figures/fig_native_baseline.png"
MODELS = ("2-layer", "6-layer")
COLORS = ("#4C72B0", "#DD8452")


def load_values(path: Path) -> list[dict[str, float | str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = {
        row["model"]: row
        for row in rows
        if row["budget"] == "depth-matched" and row["target_depth"] == "All"
    }
    if set(selected) != set(MODELS):
        raise ValueError("source data 缺少 depth-matched / All 的两层或六层结果")
    return [
        {
            "model": model,
            "accuracy": float(selected[model]["accuracy_percent"]),
            "ci_low": float(selected[model]["ci95_low"]),
            "ci_high": float(selected[model]["ci95_high"]),
        }
        for model in MODELS
    ]


def draw(rows: list[dict[str, float | str]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    means = np.asarray([float(row["accuracy"]) for row in rows])
    lows = np.asarray([float(row["ci_low"]) for row in rows])
    highs = np.asarray([float(row["ci_high"]) for row in rows])
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(11.52, 7.68))
    bars = ax.bar(x, means, width=0.46, color=COLORS)
    ax.errorbar(
        x,
        means,
        yerr=np.vstack((means - lows, highs - means)),
        fmt="none",
        color="#30343B",
        elinewidth=1.2,
        capsize=5,
        zorder=3,
    )
    for bar, mean, high in zip(bars, means, highs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            high + 1.6,
            f"{mean:.1f}%",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    ax.set_title("Native-task baseline accuracy", fontsize=18)
    ax.set_ylabel("Accuracy (%)", fontsize=15)
    ax.set_xticks(x, ["2-layer\ncheckpoint_350", "6-layer\ncheckpoint_350"])
    ax.set_ylim(0, 108)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(bars, ["2-layer checkpoint_350", "6-layer checkpoint_350"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=100, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = load_values(args.input)
    assert np.isclose(float(rows[0]["accuracy"]), 97.97136038186157)
    assert np.isclose(float(rows[1]["accuracy"]), 61.09785202863962)
    draw(rows, args.output)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
