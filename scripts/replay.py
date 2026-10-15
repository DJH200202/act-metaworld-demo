# ruff: noqa: E402
# MUJOCO_GL 必须在导入仿真环境之前设置。
"""加载 checkpoint，在新初始化的仿真中执行策略，保存视频与结果。"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import torch
from evaluate import write_video
from lerobot.envs.metaworld import MetaworldEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/act_push"))
    parser.add_argument("--mode", choices=("act", "expert", "random"), default="act")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=300000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    if min(args.episodes, args.max_steps) < 1:
        parser.error("episode and step counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if args.mode == "act":
        config = ACTConfig.from_pretrained(args.checkpoint)
        config.device = device
        policy = ACTPolicy.from_pretrained(args.checkpoint, config=config).to(device).eval()
        pre, post = make_pre_post_processors(
            policy.config,
            pretrained_path=str(args.checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
    env = MetaworldEnv(task="push-v3", obs_type="pixels_agent_pos")
    results = []
    try:
        for index in range(args.episodes):
            seed = args.seed + index
            obs, _ = env.reset(seed=seed)
            if args.mode == "act":
                policy.reset()
            rng = np.random.default_rng(seed)
            frames = [obs["pixels"].copy()]
            reward_sum, success = 0.0, False
            for step in range(args.max_steps):
                if args.mode == "expert":
                    action = env.expert_policy.get_action(env._env._get_obs())
                elif args.mode == "random":
                    action = rng.uniform(-1, 1, size=4)
                else:
                    with torch.inference_mode():
                        action = post(policy.select_action(pre(preprocess_observation(obs))))
                    action = action[0].cpu().numpy()
                action = np.clip(action, -1, 1).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
                success = bool(info.get("is_success", False))
                if not (terminated or truncated):
                    frames.append(obs["pixels"].copy())
                if success or terminated or truncated:
                    break
            video = f"{args.mode}_seed_{seed}.mp4"
            write_video(frames, args.output / video)
            row = {
                "seed": seed,
                "success": success,
                "steps": step + 1,
                "return": reward_sum,
                "video": video,
            }
            results.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    (args.output / "report.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "device": device,
                "max_steps": args.max_steps,
                "success_rate": sum(row["success"] for row in results) / len(results),
                "episodes": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
