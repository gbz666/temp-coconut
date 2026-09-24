"""Evaluate the original ProsQA questions with their native 3/4 latent budget."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch

from prepare_eval_queries import DEFAULT_OUTPUT as DEFAULT_PREPARED
from run_depth_selectivity import CHECKPOINT_DIR, CONFIG_DIR, DATA_DIR, choose_device, load, sha


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/native_baseline"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def estimate(rows: list[dict]) -> dict:
    by_graph: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        by_graph[row["graph_id"]].append(int(row["correct"]))
    graph_values = np.asarray([np.mean(values) for values in by_graph.values()])
    assert all(len(values) == 2 for values in by_graph.values())
    rng = np.random.default_rng(20260920)
    n = len(graph_values)
    weights = rng.multinomial(n, np.ones(n) / n, size=2000) / n
    bootstrap = weights @ graph_values
    return dict(
        accuracy_percent=float(graph_values.mean() * 100),
        ci95_low=float(np.quantile(bootstrap, .025) * 100),
        ci95_high=float(np.quantile(bootstrap, .975) * 100),
        n_graphs=n,
        n_queries=len(rows),
        n_correct=int(sum(row["correct"] for row in rows)),
    )


@torch.inference_mode()
def evaluate(model, tokenizer, rows: list[dict], batch_size: int) -> list[dict]:
    device = next(model.parameters()).device
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["prompt_length"], row["original_depth"])].append(row)
    predictions = []
    for (prompt_length, steps), group in sorted(groups.items()):
        for offset in range(0, len(group), batch_size):
            batch = group[offset:offset + batch_size]
            ids = torch.tensor([
                tokenizer.encode(row["prompt"], add_special_tokens=False)
                + [33] * steps + [37]
                for row in batch
            ], dtype=torch.long, device=device)
            assert ids.shape == (len(batch), prompt_length + steps + 1)
            output = model(
                ids,
                torch.ones_like(ids),
                ids,
                torch.arange(ids.shape[1], device=device)[None].expand(len(batch), -1),
                inference_only=True,
            )
            tokens = output.logits[:, -1].argmax(dim=-1).tolist()
            predictions.extend(dict(
                graph_id=row["graph_id"],
                query_id=row["query_id"],
                original_depth=steps,
                target=row["target"],
                predicted_token=token,
                correct=int(token == row["target"]),
            ) for row, token in zip(batch, tokens))
    # Retain the original evaluation order for its graph-level bootstrap.
    input_order = {row["query_id"]: index for index, row in enumerate(rows)}
    prompt_length = {row["query_id"]: row["prompt_length"] for row in rows}
    predictions.sort(key=lambda row: (
        prompt_length[row["query_id"]], input_order[row["query_id"]]
    ))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new output directory; results are not overwritten.")
    selection = json.loads((args.prepared / "selection.json").read_text(encoding="utf-8"))
    dataset = DATA_DIR / "prosqa_test_graph_4_coconut_shuffled_with_bfs.json"
    if selection["dataset_sha256"] != sha(dataset):
        raise ValueError("Prepared questions do not match the current test dataset")
    originals = read_jsonl(args.prepared / "originals.jsonl")
    if len(originals) != selection["n_original_queries"] or not originals:
        raise ValueError("Original question count does not match selection.json")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = {}
    device = choose_device(args.device)
    for layers in (2, 6):
        model, tokenizer = load(
            CHECKPOINT_DIR / f"{layers}layer/checkpoint_350",
            CONFIG_DIR / f"model_{layers}layer.json",
            device=device,
        )
        predictions = evaluate(model, tokenizer, originals, args.batch_size)
        with (args.output / f"{layers}layer.jsonl").open("w", encoding="utf-8") as stream:
            for row in predictions:
                stream.write(json.dumps(row) + "\n")
        results[layers] = predictions
        del model
    with (args.output / "native_baseline.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "budget", "model", "target_depth", "accuracy_percent", "ci95_low",
            "ci95_high", "n_graphs", "n_queries", "n_correct",
        ])
        for layers in (2, 6):
            for depth in (3, 4, "All"):
                selected = (results[layers] if depth == "All" else [
                    row for row in results[layers] if row["original_depth"] == depth
                ])
                result = estimate(selected)
                writer.writerow([
                    "depth-matched", f"{layers}-layer", depth,
                    *(result[key] for key in (
                        "accuracy_percent", "ci95_low", "ci95_high", "n_graphs",
                        "n_queries", "n_correct",
                    )),
                ])
                print(f"{layers}-layer depth={depth}: {result['accuracy_percent']:.2f}%")


if __name__ == "__main__":
    main()
