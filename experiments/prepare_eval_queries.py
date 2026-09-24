"""Create deterministic evaluation queries from the upstream ProsQA test set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import random

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/prosqa_test_graph_4_coconut_shuffled_with_bfs.json"
DEFAULT_OUTPUT = ROOT / "artifacts/eval_queries"


def graph_info(sample: dict) -> dict[int, int]:
    adjacent: dict[int, list[int]] = defaultdict(list)
    for source, target in sample["edges"]:
        adjacent[source].append(target)
    distances = {sample["root"]: 0}
    queue = deque([sample["root"]])
    while queue:
        source = queue.popleft()
        for target in adjacent[source]:
            if target not in distances:
                distances[target] = distances[source] + 1
                queue.append(target)
    assert sample["neg_target"] not in distances
    assert distances[sample["target"]] == len(sample["steps"])
    return distances


def make_queries(data: list[dict], max_graphs: int, seed: int):
    distances = [graph_info(sample) for sample in data]
    eligible = [
        index for index, graph_distances in enumerate(distances)
        if all(depth in graph_distances.values() for depth in range(1, 5))
    ]
    selected = (
        sorted(np.random.default_rng(seed).choice(
            eligible, min(max_graphs, len(eligible)), replace=False
        ).tolist())
        if max_graphs else eligible
    )
    queries, originals = [], []
    seen = set()
    for index, (sample, graph_distances) in enumerate(zip(data, distances)):
        key = (sample["root"], tuple(sorted(map(tuple, sample["edges"]))))
        assert key not in seen
        seen.add(key)
        edges = [edge[:] for edge in sample["edges"]]
        random.Random(seed + index).shuffle(edges)
        edge_text = "<eos> " + " | ".join(f"{source} {target}" for source, target in edges)
        outdegree = Counter(source for source, _ in edges)
        for target in sorted(graph_distances):
            if not 1 <= graph_distances[target] <= 4:
                continue
            for order in (0, 1):
                candidates = ([target, sample["neg_target"]] if order == 0
                              else [sample["neg_target"], target])
                prompt = (
                    f'{edge_text} [Q] {candidates[0]} {candidates[1]} '
                    f'[R] {sample["root"]}'
                )
                row = dict(
                    graph_id=index,
                    query_id=f"g{index}_v{target}_o{order}",
                    target=target,
                    neg_target=sample["neg_target"],
                    root=sample["root"],
                    target_depth=graph_distances[target],
                    original_target=sample["target"],
                    original_depth=len(sample["steps"]),
                    target_slot=order,
                    leaf_status_matched=(outdegree[target] == 0)
                    == (outdegree[sample["neg_target"]] == 0),
                    prompt=prompt,
                    prompt_length=len(prompt.split()),
                )
                if target == sample["target"]:
                    originals.append(row)
                if index in selected:
                    queries.append(row)
    return queries, originals, distances, selected, eligible


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--graphs", type=int, default=0,
                        help="0 selects every graph with BFS depths 1 through 4")
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new output directory; prepared queries are not overwritten.")
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    queries, originals, distances, selected, eligible = make_queries(
        data, args.graphs, args.seed
    )
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "originals.jsonl", originals)
    write_jsonl(args.output / "queries.jsonl", queries)
    graphs = [
        dict(graph_id=index, edges=sample["edges"], root=sample["root"],
             target=sample["target"], neg_target=sample["neg_target"],
             original_depth=len(sample["steps"]), distances=distances[index],
             optimal=sample["neighbor_k"])
        for index, sample in enumerate(data)
    ]
    (args.output / "graphs.json").write_text(
        json.dumps(graphs, indent=2), encoding="utf-8"
    )
    (args.output / "selection.json").write_text(
        json.dumps(dict(dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                        seed=args.seed, selected_graphs=selected,
                        eligible_graph_count=len(eligible),
                        n_original_queries=len(originals), n_queries=len(queries)),
                   indent=2),
        encoding="utf-8",
    )
    print(f"Prepared {len(originals)} original and {len(queries)} depth queries")


if __name__ == "__main__":
    main()
