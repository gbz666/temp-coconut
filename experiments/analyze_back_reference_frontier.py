"""Analyze natural cross-BFS back references and draw the four-panel figure.

The input is a completed depth-selectivity artifact containing all 419 official
test graphs and two original-query candidate orders per graph.  No graph is
constructed for the main analysis: eligible cases are selected from edges that
already occur in the official test graphs.

For step t in {3, 4}, a natural back reference is an existing edge
F_(t-1) -> F_(t-2).  The positive set is F_t.  Negative sets are unreachable
node IDs, other reachable non-root depths, or the selected old endpoint.  Node
scores are cosine similarities between z_t and the tied node embeddings.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from plot_back_reference_reproduction import build_figure, save_figure


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_INPUT = ROOT / "artifacts" / "depth_selectivity"
DEFAULT_OUTPUT = ROOT / "artifacts" / "back_reference_frontier"
NODE_IDS = set(range(31))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def contains_directed_cycle(edges: list[list[int]]) -> bool:
    adjacent: dict[int, list[int]] = defaultdict(list)
    for source, target in edges:
        adjacent[source].append(target)
    state: dict[int, int] = {}

    def visit(node: int) -> bool:
        if state.get(node) == 1:
            return True
        if state.get(node) == 2:
            return False
        state[node] = 1
        if any(visit(nxt) for nxt in adjacent[node]):
            return True
        state[node] = 2
        return False

    return any(visit(node) for node in list(adjacent) if node not in state)


def cosine_array(row: dict[str, Any]) -> np.ndarray:
    """Support both the cleaned artifact schema and the archived legacy schema."""
    if "node_cosine" in row:
        return np.asarray(row["node_cosine"], dtype=float)
    try:
        return np.asarray(row["representations"]["clean"]["cosine"], dtype=float)
    except KeyError as error:
        raise KeyError("originals.jsonl lacks node cosine representations") from error


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Pairwise AUROC with half credit for ties."""
    difference = np.asarray(positive)[:, None] - np.asarray(negative)[None, :]
    return float(((difference > 0) + 0.5 * (difference == 0)).mean())


def bootstrap_summary(values: list[float], seed: int = 20260922) -> dict[str, Any]:
    """Equal-graph mean and 5,000 whole-graph bootstrap interval."""
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(len(array), size=(5000, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "n_graphs": len(array),
    }


def select_cases(
    graphs: list[dict[str, Any]], originals: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Select deterministic natural back-reference cases from official graphs."""
    rows_by_graph: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in originals:
        rows_by_graph[int(row["graph_id"])].append(row)

    if any(contains_directed_cycle(graph["edges"]) for graph in graphs):
        raise ValueError("Expected the official graph set to be acyclic")

    cases: list[dict[str, Any]] = []
    for graph in graphs:
        graph_id = int(graph["graph_id"])
        distances = {int(node): int(depth) for node, depth in graph["distances"].items()}
        edges = {tuple(edge) for edge in graph["edges"]}
        base_rows = sorted(rows_by_graph[graph_id], key=lambda row: row["target_slot"])
        if len(base_rows) != 2 or {row["target_slot"] for row in base_rows} != {0, 1}:
            raise ValueError(f"Graph {graph_id} does not have both candidate orders")

        for step in (3, 4):
            frontier = sorted(node for node, depth in distances.items() if depth == step)
            candidates = sorted(
                (source, old_node)
                for source, old_node in edges
                if distances.get(source) == step - 1 and distances.get(old_node) == step - 2
            )
            if not frontier or not candidates:
                continue

            # Fixed deterministic rule: choose the lexicographically first edge.
            source, old_node = candidates[0]
            if (old_node, source) in edges:
                raise ValueError(f"Graph {graph_id} unexpectedly contains a directed 2-cycle")
            unreachable = sorted(NODE_IDS - set(distances))
            other_reachable = sorted(
                node for node, depth in distances.items() if depth > 0 and depth != step
            )
            if not unreachable or not other_reachable:
                continue

            cases.append(
                {
                    "graph_id": graph_id,
                    "step": step,
                    "back_source": source,
                    "back_target": old_node,
                    "frontier": frontier,
                    "unreachable": unreachable,
                    "other_reachable": other_reachable,
                    "base_rows": base_rows,
                }
            )
    return cases


def analyze(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        step = case["step"]
        old_node = case["back_target"]
        # Candidate order is a nuisance factor; average the two trajectories
        # within graph before any graph-level statistic is computed.
        cosine = np.mean([cosine_array(row) for row in case["base_rows"]], axis=0)
        current = cosine[step - 1]
        record = {
            "graph_id": case["graph_id"],
            "step": step,
            "back_source": case["back_source"],
            "back_target": old_node,
            "n_frontier": len(case["frontier"]),
            "n_unreachable": len(case["unreachable"]),
            "n_other_reachable": len(case["other_reachable"]),
            "auc_unreachable": auc(current[case["frontier"]], current[case["unreachable"]]),
            "auc_other_reachable": auc(
                current[case["frontier"]], current[case["other_reachable"]]
            ),
            "auc_back_reference": auc(current[case["frontier"]], current[[old_node]]),
            "old_cos_when_frontier": float(cosine[step - 3, old_node]),
            "old_cos_when_revisited": float(cosine[step - 1, old_node]),
        }
        record["temporal_delta"] = (
            record["old_cos_when_revisited"] - record["old_cos_when_frontier"]
        )
        rows.append(record)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "auc_unreachable",
        "auc_other_reachable",
        "auc_back_reference",
        "old_cos_when_frontier",
        "old_cos_when_revisited",
        "temporal_delta",
    ]
    output: dict[str, Any] = {
        str(step): {
            key: bootstrap_summary([row[key] for row in rows if row["step"] == step])
            for key in metric_keys
        }
        for step in (3, 4)
    }

    common = sorted(
        {row["graph_id"] for row in rows if row["step"] == 3}
        & {row["graph_id"] for row in rows if row["step"] == 4}
    )
    by_step = {
        step: {row["graph_id"]: row for row in rows if row["step"] == step}
        for step in (3, 4)
    }
    output["n_common_graphs"] = len(common)
    output["common_step4_minus_step3"] = {
        key: bootstrap_summary(
            [by_step[4][graph_id][key] - by_step[3][graph_id][key] for graph_id in common]
        )
        for key in ("auc_unreachable", "auc_other_reachable", "auc_back_reference", "temporal_delta")
    }
    output["definition"] = (
        "Natural official-test-graph F(t-1)->F(t-2) back references for t=3 or 4; "
        "one lexicographically first qualifying edge per graph. Positive nodes are F_t. "
        "Negatives are unreachable node IDs, all other reachable non-root depths, or the "
        "selected old endpoint. z_t cosine scores are averaged over both candidate orders "
        "within graph; graphs are equally weighted. No artificial edge is used."
    )
    return output


def write_outputs(
    rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "source_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    frontier_auroc = {
        step: [
            summary[str(step)][key]["mean"]
            for key in ("auc_unreachable", "auc_other_reachable", "auc_back_reference")
        ]
        for step in (3, 4)
    }
    old_node_decline = {
        step: -summary[str(step)]["temporal_delta"]["mean"] for step in (3, 4)
    }
    figure = build_figure(frontier_auroc, old_node_decline)
    figure_paths = save_figure(figure, output_dir, "back_reference_combined", 300)
    import matplotlib.pyplot as plt

    plt.close(figure)
    return [csv_path, summary_path, *figure_paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graphs = json.loads((args.input / "graphs.json").read_text(encoding="utf-8"))
    originals = read_jsonl(args.input / "originals.jsonl")
    cases = select_cases(graphs, originals)
    rows = analyze(cases)
    summary = summarize(rows)
    outputs = write_outputs(rows, summary, args.output)

    counts = {step: sum(row["step"] == step for row in rows) for step in (3, 4)}
    print(f"Official graphs: {len(graphs)}; natural cases: {counts}")
    for step in (3, 4):
        values = summary[str(step)]
        print(
            f"step {step}: "
            f"AUROC={values['auc_unreachable']['mean']:.6f}, "
            f"{values['auc_other_reachable']['mean']:.6f}, "
            f"{values['auc_back_reference']['mean']:.6f}; "
            f"old-node decline={-values['temporal_delta']['mean']:.6f}"
        )
    for path in outputs:
        print(path.resolve())


if __name__ == "__main__":
    main()
