"""无需机器人依赖，核对提交的原始报告与展示数字是否一致。"""

import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((PROJECT / "configs/experiment.json").read_text())
    test = json.loads((PROJECT / "results/final_test.json").read_text())
    selection = json.loads((PROJECT / "results/model_selection.json").read_text())
    seeds = list(
        range(config["test_seed_start"], config["test_seed_start"] + config["test_episodes"])
    )
    assert test["seeds"] == seeds
    summary = json.loads((PROJECT / "results/data/collection_summary.json").read_text())
    seed_groups = [set(seeds)]
    seed_groups.append(
        set(
            range(
                config["selection_seed_start"],
                config["selection_seed_start"] + config["selection_episodes"],
            )
        )
    )
    for split in ("train", "val"):
        manifest = json.loads((PROJECT / f"results/data/{split}_manifest.json").read_text())
        assert len(manifest) == config[f"{split}_episodes"] == summary[split]["episodes"]
        assert sum(row["steps"] for row in manifest) == summary[split]["frames"]
        assert all(row["success"] for row in manifest)
        group = {row["seed"] for row in manifest}
        assert len(group) == len(manifest)
        seed_groups.append(group)
    for index, group in enumerate(seed_groups):
        assert all(not group.intersection(other) for other in seed_groups[index + 1 :])
    assert test["initial_state_comparison"] == "passed"
    for mode, result in test["results"].items():
        rows = result["episodes"]
        assert [row["seed"] for row in rows] == seeds
        assert result["successes"] == sum(row["success"] for row in rows)
        assert result["success_rate"] == result["successes"] / len(rows)
        assert result["mean_steps"] == sum(row["steps"] for row in rows) / len(rows)
        for row in rows:
            assert 1 <= row["steps"] <= test["max_steps"]
        print(mode, f"{result['successes']}/{len(rows)}")
    best = max(
        selection["candidates"],
        key=lambda row: (row["selection_success_rate"], -row["validation_l1"], row["step"]),
    )
    assert selection["selected"] == best
    assert f"{best['step']:06d}" in test["checkpoint"]
    with (PROJECT / "results/training_metrics.csv").open() as source:
        metrics = list(csv.DictReader(source))
    assert [int(row["step"]) for row in metrics] == list(range(100, 5001, 100))
    print("Selected step:", best["step"])
    print("Evidence consistency checks passed.")


if __name__ == "__main__":
    main()
