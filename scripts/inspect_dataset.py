"""Check stored data and the temporal action chunks used by ACT."""

# 命令行参数：把终端中的 --output 等选项交给 Python。
import argparse

# JSON：用于保存可读的实验配置、统计和评测结果。
import json

# Path 表示文件路径；用 / 拼接目录，不是数值除法。
from pathlib import Path

# NumPy 负责仿真环境使用的数组；np 是它的简写。
import numpy as np

# PyTorch 负责张量、神经网络和梯度。
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image


# 主入口负责解析参数、组织调用；把具体计算放在其他函数里。
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets/push_v2",
    )
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dataset_inspection"),
    )
    # 读取命令行参数；没有提供的选项使用 default。
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    # 依次检查训练集和验证集，二者共享字段定义。
    for split in ("train", "val"):
        # 从磁盘读取已有数据；此处没有创建或采集新演示。
        ds = LeRobotDataset(f"local/push_{split}", root=args.root / split, video_backend="pyav")
        # 下标 0 读取第一个时间步；sample 是字段名称到张量的字典。
        sample = ds[0]
        print(f"{split}: episodes={ds.num_episodes}, frames={len(ds)}, fps={ds.fps}")
        shapes = {k: list(v.shape) for k, v in sample.items() if isinstance(v, torch.Tensor)}
        print("sample shapes:", shapes)
        # assert 用于核对实际形状，条件为假时立即报错。
        assert tuple(sample["observation.image"].shape) == (3, 480, 480)
        # assert 用于核对实际形状，条件为假时立即报错。
        assert tuple(sample["observation.state"].shape) == (4,)
        # assert 用于核对实际形状，条件为假时立即报错。
        assert tuple(sample["action"].shape) == (4,)
        assert torch.isfinite(sample["observation.state"]).all()
        raw_actions = np.array(ds.hf_dataset["action"])
        assert np.isfinite(raw_actions).all() and np.abs(raw_actions).max() <= 1.00001
        image = sample["observation.image"].permute(1, 2, 0).numpy()
        if image.dtype != np.uint8:
            image = np.round(image * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(image).save(args.output / f"{split}_first_frame.png")
        # 再次打开同一数据集，但本次让读取器组合未来动作序列。
        temporal = LeRobotDataset(
            f"local/push_{split}",
            root=args.root / split,
            video_backend="pyav",
            delta_timestamps={"action": [i / ds.fps for i in range(args.chunk_size)]},
        )
        # 当前图像不变，action 从一个动作扩展为一段未来动作。
        chunk = temporal[0]
        assert tuple(chunk["action"].shape) == (args.chunk_size, 4)
        assert tuple(chunk["action_is_pad"].shape) == (args.chunk_size,)
        # The chunk's first label must match the same frame's single action.
        torch.testing.assert_close(chunk["action"][0], sample["action"])
        print(
            "ACT action chunk:",
            tuple(chunk["action"].shape),
            "pad mask:",
            chunk["action_is_pad"].tolist(),
        )
        manifest = json.loads((args.root / f"{split}_manifest.json").read_text())
        # 定位第一条轨迹的最后一帧，专门检查跨轨迹边界处理。
        last_index = manifest[0]["steps"] - 1
        # 末尾缺少未来真实动作；action_is_pad=True 的位置不能计入动作损失。
        tail_chunk = temporal[last_index]
        assert not tail_chunk["action_is_pad"][0].item()
        assert tail_chunk["action_is_pad"][1:].all().item()
        print("First episode's final-frame mask:", tail_chunk["action_is_pad"].tolist())
        report[split] = {
            "episodes": ds.num_episodes,
            "frames": len(ds),
            "fps": ds.fps,
            "sample_shapes": shapes,
            "chunk_shape": list(chunk["action"].shape),
            "episode_boundary_padding": "passed",
        }
    train_manifest = json.loads((args.root / "train_manifest.json").read_text())
    val_manifest = json.loads((args.root / "val_manifest.json").read_text())
    assert not ({r["seed"] for r in train_manifest} & {r["seed"] for r in val_manifest})
    (args.output / "dataset_check.json").write_text(json.dumps(report, indent=2))
    print("All dataset checks passed; train and validation seeds are disjoint.")


# 仅在直接运行本文件时调用 main；被其他文件 import 时不会自动运行实验。
if __name__ == "__main__":
    main()
