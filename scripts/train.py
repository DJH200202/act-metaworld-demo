"""用 torchrun 启动八进程 DDP；batch_size 是每卡大小。"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpus", help="physical IDs, e.g. 0 or 0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    config = json.loads((PROJECT / "configs/experiment.json").read_text())
    steps = config["smoke_steps"] if args.smoke else config["training_steps"]
    name = "smoke_ddp100" if args.smoke else "act_ddp5000"
    gpus = args.gpus.split(",") if args.gpus else list(map(str, config["physical_gpus"]))
    if not gpus or any(not item.isdigit() for item in gpus) or len(set(gpus)) != len(gpus):
        parser.error("GPU IDs must be unique non-negative integers")
    output = args.output or PROJECT / "outputs" / name
    if output.exists():
        raise FileExistsError(output)
    data = args.data_root or PROJECT / "datasets/push_v2/train"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(gpus)}",
        "--module",
        "lerobot.scripts.lerobot_train",
        "--dataset.repo_id=local/act8_push_train",
        f"--dataset.root={data}",
        "--dataset.video_backend=pyav",
        "--dataset.use_imagenet_stats=true",
        "--policy.type=act",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--policy.use_amp=false",
        f"--policy.chunk_size={config['chunk_size']}",
        f"--policy.n_action_steps={config['n_action_steps']}",
        f"--policy.dim_model={config['dim_model']}",
        f"--policy.n_heads={config['n_heads']}",
        f"--policy.dim_feedforward={config['dim_feedforward']}",
        f"--policy.n_encoder_layers={config['n_encoder_layers']}",
        f"--policy.n_decoder_layers={config['n_decoder_layers']}",
        f"--policy.latent_dim={config['latent_dim']}",
        f"--policy.kl_weight={config['kl_weight']}",
        f"--policy.optimizer_lr={config['learning_rate']}",
        f"--policy.optimizer_lr_backbone={config['learning_rate']}",
        f"--batch_size={config['per_gpu_batch_size']}",
        f"--num_workers={config['dataloader_workers_per_rank']}",
        f"--steps={steps}",
        "--log_freq=20" if args.smoke else "--log_freq=100",
        f"--save_freq={steps if args.smoke else config['save_freq']}",
        "--env_eval_freq=0",
        "--wandb.enable=false",
        f"--seed={config['training_seed']}",
        f"--output_dir={output}",
    ]
    (PROJECT / "logs").mkdir(exist_ok=True)
    (PROJECT / "logs" / f"{name}_command.json").write_text(json.dumps(command, indent=2))
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    print(
        f"Starting {name}: GPUs={gpus}, global batch={len(gpus) * config['per_gpu_batch_size']}, steps={steps}",
        flush=True,
    )
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
