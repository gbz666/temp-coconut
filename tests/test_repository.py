from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RepositoryTest(unittest.TestCase):
    def test_model_configs(self) -> None:
        for layers in (2, 6):
            config = json.loads((ROOT / f"configs/model_{layers}layer.json").read_text())
            self.assertEqual(config["n_layer"], layers)
            self.assertEqual(config["n_head"], 8)
            self.assertEqual(config["n_embd"], 768)
            self.assertEqual(config["vocab_size"], 40)

    def test_training_configs_resolve(self) -> None:
        for mode in ("train", "eval"):
            for layers in (2, 6):
                config = yaml.safe_load(
                    (ROOT / f"configs/{mode}_{layers}layer.yaml").read_text()
                )
                self.assertTrue((ROOT / config["model_id"]).is_file())
                self.assertTrue((ROOT / config["train_path"]).is_file())
                self.assertTrue((ROOT / config["val_path"]).is_file())
                if mode == "eval" and (ROOT / "checkpoints").exists():
                    self.assertTrue((ROOT / config["load_model_path"]).is_file())

    def test_test_dataset(self) -> None:
        path = ROOT / "data/prosqa_test_graph_4_coconut_shuffled_with_bfs.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 419)
        required = {"edges", "root", "target", "neg_target", "steps", "neighbor_k"}
        self.assertTrue(all(required <= set(row) for row in rows))
        self.assertEqual(
            sha256(path),
            "da778133a3f8d1c094ae02773b7a495e7578a5647b112d1103d7ef29954aa4c9",
        )

    def test_saved_native_baseline(self) -> None:
        path = ROOT / "results/source_data/native_baseline.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        overall = {
            row["model"]: float(row["accuracy_percent"])
            for row in rows
            if row["budget"] == "depth-matched" and row["target_depth"] == "All"
        }
        self.assertAlmostEqual(overall["2-layer"], 97.97136038186157)
        self.assertAlmostEqual(overall["6-layer"], 61.09785202863962)

    def test_saved_depth_selectivity(self) -> None:
        path = ROOT / "results/source_data/depth_selectivity.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        values = {
            (row["panel"], row["target_bfs_depth"], row["condition"]): float(row["value"])
            for row in rows
        }
        self.assertAlmostEqual(values[("A", "2", "only_P2")], 92.7541035354)
        self.assertAlmostEqual(values[("A", "3", "only_P3")], 99.1420905483)
        self.assertAlmostEqual(values[("A", "4", "only_P4")], 98.6426767677)
        self.assertAlmostEqual(values[("B", "1", "z1")], 0.996562118437)

    def test_checkpoint_hashes(self) -> None:
        expected = {
            2: "af23f0b0729c8bfa0a0417c8ed45ae9fd2bcc09b002c555f7ebfc0f2cb0d3a3c",
            6: "7fb31ccafab134781ccb5dbb297b3522f3a15a8c8cf221481050723259c94259",
        }
        for layers, digest in expected.items():
            checkpoint = ROOT / f"checkpoints/{layers}layer/checkpoint_350"
            if not checkpoint.exists():
                self.skipTest("local checkpoints are not part of the Git repository")
            self.assertEqual(sha256(checkpoint), digest)

    def test_checkpoints_load_on_cpu(self) -> None:
        runner = load_module(
            "run_depth_selectivity",
            ROOT / "experiments/run_depth_selectivity.py",
        )
        for layers in (2, 6):
            checkpoint = ROOT / f"checkpoints/{layers}layer/checkpoint_350"
            if not checkpoint.exists():
                self.skipTest("local checkpoints are not part of the Git repository")
            model, _ = runner.load(
                checkpoint,
                ROOT / f"configs/model_{layers}layer.json",
                device="cpu",
            )
            self.assertEqual(model.base_causallm.config.n_layer, layers)


if __name__ == "__main__":
    unittest.main()
