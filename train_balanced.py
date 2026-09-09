import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from model_cnn import ChessCNN
from train_mate_mix import ShuffledDataset
from train_real import ChessDataset, run_epoch


def main():
    random.seed(42)
    torch.manual_seed(42)

    root = Path(__file__).resolve().parent

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用的 CUDA 显卡")

    device = torch.device("cuda")

    print("正在加载两套训练数据……", flush=True)

    normal_data = ShuffledDataset(root / "train.jsonl")
    puzzle_data = ShuffledDataset(root / "puzzles_mate_train.jsonl")
    
    # 随机抽取约 18.3 万条战术题，占混合数据约 20%
    count = min(len(puzzle_data), len(normal_data) // 4)
    indices = random.Random(42).sample(
        range(len(puzzle_data)), count
    )
    puzzle_data = Subset(puzzle_data, indices)

    # 合并后统一打乱，每轮每条记录使用一次
    train_loader = DataLoader(
        ConcatDataset([normal_data, puzzle_data]),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        num_workers=0,
    )

    normal_val_loader = DataLoader(
        ChessDataset(root / "val.jsonl"),
        batch_size=256,
        num_workers=0,
    )

    puzzle_val_loader = DataLoader(
        ChessDataset(root / "puzzles_mate_val.jsonl"),
        batch_size=256,
        num_workers=0,
    )

    model = ChessCNN().to(device)
    model.load_state_dict(torch.load(
        root / "chess_model_puzzles_epoch2.pt",
        map_location=device,
        weights_only=True,
    ))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.00005
    )
    loss_function = nn.CrossEntropyLoss()

    print(f"普通训练样本：{len(normal_data):,}")
    print(f"战术训练样本：{len(puzzle_data):,}")
    print("开始混合训练……", flush=True)

    for epoch in range(1, 2):
        print(f"\n===== 平衡微调第 {epoch}/1 轮 =====", flush=True)

        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_function, device, optimizer
        )

        print("\n评估普通验证集……", flush=True)
        normal_loss, normal_accuracy = run_epoch(
            model, normal_val_loader, loss_function, device
        )

        print("\n评估战术验证集……", flush=True)
        puzzle_loss, puzzle_accuracy = run_epoch(
            model, puzzle_val_loader, loss_function, device
        )

        print(f"\n第 {epoch} 轮汇总")
        print(
            f"混合训练损失：{train_loss:.4f} | "
            f"准确率：{train_accuracy:.2%}"
        )
        print(
            f"普通验证损失：{normal_loss:.4f} | "
            f"准确率：{normal_accuracy:.2%}"
        )
        print(
            f"战术验证损失：{puzzle_loss:.4f} | "
            f"题库答案一致率：{puzzle_accuracy:.2%}"
        )

        output = root / "chess_model_balanced.pt"
        torch.save(model.state_dict(), output)
        print("已保存：", output)

    print("\n训练完成！")


if __name__ == "__main__":
    main()