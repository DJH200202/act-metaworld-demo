"""独立种子的闭环评测；验证误差按有效动作分量数加权汇总。"""

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
PROJECT = Path(__file__).resolve().parents[1]


def load_policy(checkpoint):
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    torch.set_num_threads(2)
    torch.manual_seed(42)
    policy = ACTPolicy.from_pretrained(checkpoint).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    return policy, pre, post


def validation(checkpoint, root, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader

    policy, pre, _ = load_policy(checkpoint)
    # 数据频率与采集配置一致：80Hz。
    ds = LeRobotDataset(
        "local/act8_push_val",
        root=root / "val",
        video_backend="pyav",
        delta_timestamps={"action": [i / 80 for i in range(policy.config.chunk_size)]},
    )
    error_sum, elements, frames = 0.0, 0, 0
    with torch.inference_mode():
        for batch in DataLoader(ds, batch_size=8, shuffle=False, num_workers=0):
            batch = pre(batch)
            _, metrics = policy.forward(batch)
            count = int((~batch["action_is_pad"]).sum().item()) * batch["action"].shape[-1]
            error_sum += metrics["l1_loss"] * count
            elements += count
            frames += batch["action"].shape[0]
    del policy
    torch.cuda.empty_cache()
    return {
        "normalized_action_l1": error_sum / elements,
        "frames_evaluated": frames,
        "valid_action_elements": elements,
        "aggregation": "sum of absolute errors / all valid action elements",
    }


def write_video(frames, path):
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=80)
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = 1
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def rollout_shard(job):
    # spawn 子进程各自绑定物理 GPU；其 cuda:0 对应分配的物理卡。
    os.environ["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    import torch
    from lerobot.envs.metaworld import MetaworldEnv
    from lerobot.envs.utils import preprocess_observation

    torch.set_num_threads(2)
    if "act" in job["modes"]:
        policy, pre, post = load_policy(Path(job["checkpoint"]))
    output = Path(job["output"])
    results = {}
    for mode in job["modes"]:
        env = MetaworldEnv(task="push-v3", obs_type="pixels_agent_pos")
        rows, saved = [], {True: 0, False: 0}
        try:
            for seed in job["seeds"]:
                obs, _ = env.reset(seed=seed)
                initial = env._env._get_obs().copy()
                if mode == "act":
                    policy.reset()
                rng = np.random.default_rng(seed)
                # 每个工作进程每种控制器保留最多一段成功和一段失败视频。
                record = not all(saved.values())
                frames = [obs["pixels"].copy()] if record else []
                total_reward, success = 0.0, False
                for step in range(job["max_steps"]):
                    if mode == "expert":
                        action = env.expert_policy.get_action(env._env._get_obs())
                    elif mode == "random":
                        action = rng.uniform(-1, 1, size=4)
                    else:
                        with torch.inference_mode():
                            action = post(policy.select_action(pre(preprocess_observation(obs))))
                        action = action[0].cpu().numpy()
                    action = np.clip(action, -1, 1).astype(np.float32)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    success = bool(info.get("is_success", False))
                    if record and not (terminated or truncated):
                        frames.append(obs["pixels"].copy())
                    if success or terminated or truncated:
                        break
                row = {
                    "seed": seed,
                    "success": success,
                    "steps": step + 1,
                    "return": total_reward,
                    "initial_state": initial.tolist(),
                }
                rows.append(row)
                if record and not saved[success]:
                    path = (
                        output
                        / "videos"
                        / f"{mode}_seed_{seed}_{'success' if success else 'failure'}.mp4"
                    )
                    write_video(frames, path)
                    saved[success] += 1
                print(f"{mode} seed={seed} success={success} steps={step + 1}", flush=True)
                # 分片报告持续写出，便于查看进度和定位失败。
                (output / f"progress_gpu{job['gpu']}_{mode}.json").write_text(
                    json.dumps(rows, indent=2)
                )
        finally:
            env.close()
        results[mode] = rows
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "datasets/push_v2")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--gpus", default="0", help="physical GPU IDs, comma separated")
    parser.add_argument("--modes", nargs="+", choices=("random", "expert", "act"), default=["act"])
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if any(not gpu.isdigit() for gpu in gpus) or len(set(gpus)) != len(gpus):
        parser.error("GPU IDs must be unique non-negative integers")
    args.workers = args.workers or len(gpus)
    if min(args.episodes, args.max_steps, args.workers) < 1 or args.workers > len(gpus):
        parser.error("positive counts required; workers must not exceed available GPU IDs")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "videos").mkdir()
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "seeds": seeds,
        "max_steps": args.max_steps,
        "task": "push-v3",
        "protocol": f"single-task {args.max_steps}-step simulation evaluation, not MT50",
    }
    if not args.skip_validation:
        report["validation"] = validation(args.checkpoint, args.data_root, gpus[0])
        print("Full validation:", report["validation"], flush=True)
    workers = min(args.workers, args.episodes)
    jobs = [
        {
            "gpu": int(gpus[rank]),
            "seeds": seeds[rank::workers],
            "modes": args.modes,
            "checkpoint": str(args.checkpoint),
            "output": str(args.output),
            "max_steps": args.max_steps,
        }
        for rank in range(workers)
    ]
    rows = {mode: [] for mode in args.modes}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
        for result in executor.map(rollout_shard, jobs):
            for mode, values in result.items():
                rows[mode].extend(values)
    report["results"] = {}
    for mode, values in rows.items():
        values.sort(key=lambda row: row["seed"])
        assert [row["seed"] for row in values] == seeds
        report["results"][mode] = {
            "success_rate": sum(row["success"] for row in values) / len(values),
            "successes": sum(row["success"] for row in values),
            "episodes": values,
            "mean_return": float(np.mean([row["return"] for row in values])),
            "mean_steps": float(np.mean([row["steps"] for row in values])),
        }
    if "expert" in rows:
        for mode in args.modes:
            for index in range(len(seeds)):
                np.testing.assert_allclose(
                    rows["expert"][index]["initial_state"],
                    rows[mode][index]["initial_state"],
                    rtol=0,
                    atol=1e-8,
                )
        report["initial_state_comparison"] = "passed"
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(
        "SUCCESS RATES:",
        {mode: data["success_rate"] for mode, data in report["results"].items()},
        flush=True,
    )


if __name__ == "__main__":
    main()
