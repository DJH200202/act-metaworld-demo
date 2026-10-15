# ruff: noqa: E402
# MUJOCO_GL 必须在导入仿真环境之前设置。
"""并行仿真，按种子顺序保存成功演示；学生只获得图像和自身状态。"""

import argparse
import json
import multiprocessing as mp
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

os.environ.setdefault("MUJOCO_GL", "osmesa")
from lerobot.envs.metaworld import MetaworldEnv

PROJECT = Path(__file__).resolve().parents[1]
FEATURES = {
    "observation.image": {
        "dtype": "video",
        "shape": (480, 480, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (4,),
        "names": ["eef_x", "eef_y", "eef_z", "gripper_opening"],
    },
    "action": {
        "dtype": "float32",
        "shape": (4,),
        "names": ["delta_x", "delta_y", "delta_z", "gripper_command"],
    },
}
ENV = None


def init_worker():
    global ENV
    ENV = MetaworldEnv(task="push-v3", obs_type="pixels_agent_pos")


def simulate(seed, max_steps):
    obs, _ = ENV.reset(seed=seed)
    initial = ENV._env._get_obs().copy()
    frames, reward_sum = [], 0.0
    success = False
    for _ in range(max_steps):
        action = np.clip(ENV.expert_policy.get_action(ENV._env._get_obs()), -1, 1).astype(
            np.float32
        )
        frames.append(
            {
                "observation.image": obs["pixels"].copy(),
                "observation.state": obs["agent_pos"].astype(np.float32),
                "action": action.copy(),
                "task": ENV.task_description,
            }
        )
        obs, reward, terminated, truncated, info = ENV.step(action)
        reward_sum += float(reward)
        success = bool(info.get("is_success", False))
        if success or terminated or truncated:
            break
    row = {
        "seed": seed,
        "steps": len(frames),
        "success": success,
        "return": reward_sum,
        "initial_robot_state": initial[:4].tolist(),
        "initial_object_position": initial[4:7].tolist(),
        "goal_position": initial[-3:].tolist(),
    }
    return frames if success else [], row


def collect_split(config, split, output):
    count = config[f"{split}_episodes"]
    seed_start = config[f"{split}_seed_start"]
    workers = config["collection_workers"]
    dataset = LeRobotDataset.create(
        repo_id=f"local/act8_push_{split}",
        root=output / split,
        fps=80,
        robot_type="metaworld_sawyer",
        features=FEATURES,
        use_videos=True,
        video_backend="pyav",
        image_writer_threads=2,
        encoder_threads=2,
    )
    manifest, rejected = [], []
    next_seed = seed_start
    # 队列最多容纳 workers 条轨迹，避免一次性在内存堆积所有图像。
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn"), initializer=init_worker
    ) as executor:
        pending = deque()
        for _ in range(workers):
            pending.append(executor.submit(simulate, next_seed, config["max_steps"]))
            next_seed += 1
        while len(manifest) < count:
            frames, row = pending.popleft().result()
            if row["success"]:
                for frame in frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                row["episode_index"] = len(manifest)
                manifest.append(row)
                (output / f"{split}_manifest.json").write_text(json.dumps(manifest, indent=2))
                print(
                    f"{split}: saved {len(manifest)}/{count}, seed={row['seed']}, "
                    f"steps={row['steps']}",
                    flush=True,
                )
            else:
                rejected.append(row)
                print(f"{split}: rejected seed={row['seed']}", flush=True)
            if len(manifest) < count:
                if next_seed >= seed_start + count * 5:
                    raise RuntimeError("Too many unsuccessful expert attempts")
                pending.append(executor.submit(simulate, next_seed, config["max_steps"]))
                next_seed += 1
        for future in pending:
            future.cancel()
    dataset.finalize()
    (output / f"{split}_rejected.json").write_text(json.dumps(rejected, indent=2))
    return {"episodes": len(manifest), "frames": sum(row["steps"] for row in manifest)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "datasets/push_v2")
    parser.add_argument("--train-episodes", type=int)
    parser.add_argument("--val-episodes", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    config = json.loads((PROJECT / "configs/experiment.json").read_text())
    for argument, key in (
        (args.train_episodes, "train_episodes"),
        (args.val_episodes, "val_episodes"),
        (args.workers, "collection_workers"),
    ):
        if argument is not None:
            if argument < 1:
                parser.error("counts must be positive")
            config[key] = argument
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    report = {"task": config["task"], "fps": 80, "protocol": "successful expert demonstrations"}
    for split in ("train", "val"):
        report[split] = collect_split(config, split, output)
    (output / "collection_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
