"""串联短训练、采集、正式训练、验证选模、独立测试；阶段失败即停止。"""

import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((PROJECT / "configs/experiment.json").read_text())
STATE_PATH = PROJECT / "status.json"
STATE = {
    "status": "running",
    "completed_stages": [],
    "started_at": datetime.now(timezone.utc).isoformat(),
}


def update(phase, status="running", **details):
    STATE.update(
        status=status, phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **details
    )
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(STATE, indent=2, ensure_ascii=False))
    temporary.replace(STATE_PATH)
    print(f"[{STATE['updated_at']}] {phase}: {status}", flush=True)


def run(phase, command, extra_env=None):
    update(phase)
    env = dict(os.environ)
    env.update(extra_env or {})
    path = PROJECT / "logs" / f"{phase}.log"
    (PROJECT / "logs" / f"{phase}_command.json").write_text(json.dumps(command, indent=2))
    with path.open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    STATE["completed_stages"].append(phase)
    update(phase)


def main():
    if STATE_PATH.exists():
        raise FileExistsError("Pipeline status already exists; inspect it before restarting")
    for directory in ("logs", "outputs", "datasets"):
        (PROJECT / directory).mkdir(exist_ok=True)
    assert importlib.metadata.version("lerobot") == CONFIG["lerobot_version"]
    python = sys.executable
    scripts = PROJECT / "scripts"
    gpu_ids = ",".join(map(str, CONFIG["physical_gpus"]))
    run("collect", [python, str(scripts / "collect.py")], {"CUDA_VISIBLE_DEVICES": ""})
    data = PROJECT / "datasets/push_v2"
    run("smoke_ddp100", [python, str(scripts / "train.py"), "--smoke"])
    smoke = PROJECT / "outputs/smoke_ddp100/checkpoints/000100/pretrained_model"
    run(
        "smoke_load",
        [
            python,
            str(scripts / "evaluate.py"),
            "--checkpoint",
            str(smoke),
            "--output",
            str(PROJECT / "outputs/smoke_load"),
            "--episodes",
            "1",
            "--seed-start",
            "999999",
            "--max-steps",
            "2",
            "--gpus",
            "0",
            "--skip-validation",
        ],
    )
    run(
        "inspect",
        [
            python,
            str(scripts / "inspect_dataset.py"),
            "--root",
            str(data),
            "--chunk-size",
            str(CONFIG["chunk_size"]),
            "--output",
            str(PROJECT / "outputs/dataset_inspection"),
        ],
        {"CUDA_VISIBLE_DEVICES": ""},
    )
    run("formal_train", [python, str(scripts / "train.py")])
    candidates = []
    checkpoints = PROJECT / "outputs/act_ddp5000/checkpoints"
    for step in range(CONFIG["save_freq"], CONFIG["training_steps"] + 1, CONFIG["save_freq"]):
        checkpoint = checkpoints / f"{step:06d}" / "pretrained_model"
        output = PROJECT / "outputs" / f"selection_{step:06d}"
        run(
            f"selection_{step:06d}",
            [
                python,
                str(scripts / "evaluate.py"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--data-root",
                str(data),
                "--episodes",
                str(CONFIG["selection_episodes"]),
                "--seed-start",
                str(CONFIG["selection_seed_start"]),
                "--max-steps",
                str(CONFIG["max_steps"]),
                "--workers",
                str(CONFIG["evaluation_workers"]),
                "--gpus",
                gpu_ids,
            ],
        )
        report = json.loads((output / "report.json").read_text())
        candidates.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "selection_success_rate": report["results"]["act"]["success_rate"],
                "validation_l1": report["validation"]["normalized_action_l1"],
            }
        )
    selected = max(
        candidates,
        key=lambda item: (item["selection_success_rate"], -item["validation_l1"], item["step"]),
    )
    (PROJECT / "outputs/model_selection.json").write_text(
        json.dumps(
            {
                "rule": CONFIG["checkpoint_selection"],
                "candidates": candidates,
                "selected": selected,
            },
            indent=2,
        )
    )
    run(
        "final_test",
        [
            python,
            str(scripts / "evaluate.py"),
            "--checkpoint",
            selected["checkpoint"],
            "--output",
            str(PROJECT / "outputs/final_test"),
            "--episodes",
            str(CONFIG["test_episodes"]),
            "--seed-start",
            str(CONFIG["test_seed_start"]),
            "--max-steps",
            str(CONFIG["max_steps"]),
            "--workers",
            str(CONFIG["evaluation_workers"]),
            "--gpus",
            gpu_ids,
            "--modes",
            "random",
            "expert",
            "act",
            "--skip-validation",
        ],
    )
    report = json.loads((PROJECT / "outputs/final_test/report.json").read_text())
    update(
        "complete",
        "complete",
        selected_checkpoint=selected,
        test_success_rates={
            mode: values["success_rate"] for mode, values in report["results"].items()
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # 若状态文件已存在，拒绝重复启动时不覆盖原任务状态。
        if STATE.get("phase"):
            update(STATE["phase"], "failed", error=f"{type(error).__name__}: {error}")
        raise
