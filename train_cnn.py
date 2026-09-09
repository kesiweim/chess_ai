import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from model_cnn import ChessCNN
from train_real import ChessDataset, run_epoch


BATCH_SIZE = 256
EPOCHS = 6


def main():
    random.seed(42)
    torch.manual_seed(42)

    root = Path(__file__).resolve().parent

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用的 CUDA 显卡")

    device = torch.device("cuda")

    # 复用之前的数据读取和评估方法
    train_loader = DataLoader(
        ChessDataset(root / "train.jsonl", shuffle=True),
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    val_loader = DataLoader(
        ChessDataset(root / "val.jsonl"),
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    # 新结构必须从头训练，不能加载原模型的参数
    model = ChessCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    best_loss = float("inf")
    output_path = root / "chess_model_cnn.pt"

    parameter_count = sum(p.numel() for p in model.parameters())

    print("显卡：", torch.cuda.get_device_name(0))
    print(f"模型参数量：{parameter_count:,}")
    print("开始训练 CNN……", flush=True)

    for epoch in range(1, EPOCHS + 1):
        # 与原模型保持相近的学习率安排：
        # 前 3 轮 0.001，后 3 轮 0.0003，并重建优化器
        if epoch == 4:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.0003
            )

        print(f"\n===== CNN 第 {epoch}/{EPOCHS} 轮 =====", flush=True)

        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_function, device, optimizer
        )

        val_loss, val_accuracy = run_epoch(
            model, val_loader, loss_function, device
        )

        print(
            f"\n训练损失：{train_loss:.4f} | "
            f"训练准确率：{train_accuracy:.2%}"
        )
        print(
            f"验证损失：{val_loss:.4f} | "
            f"验证准确率：{val_accuracy:.2%}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), output_path)
            print("已保存当前最佳 CNN：", output_path)

    print("\nCNN 训练完成！")


if __name__ == "__main__":
    main()