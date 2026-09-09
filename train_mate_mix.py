import json
import random
from pathlib import Path

import chess
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from board_encoder import encode_board
from move_encoder import encode_move
from model_cnn import ChessCNN
from train_real import ChessDataset, run_epoch


class ShuffledDataset(Dataset):
    def __init__(self, path):
        # 只把文本放入内存，不预先生成全部棋盘张量
        with path.open("r", encoding="utf-8") as file:
            self.lines = [
                line for line in file if line.strip()
            ]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        sample = json.loads(self.lines[index])

        board = chess.Board(sample["position"])
        move = chess.Move.from_uci(sample["next_move"])

        return encode_board(board), encode_move(move)


def main():
    random.seed(42)
    torch.manual_seed(42)

    root = Path(__file__).resolve().parent

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用的 CUDA 显卡")

    device = torch.device("cuda")

    print("正在加载混合数据……", flush=True)
    train_dataset = ShuffledDataset(
        root / "train_mate_mix.jsonl"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        num_workers=0,
    )

    val_loader = DataLoader(
        ChessDataset(root / "val.jsonl"),
        batch_size=256,
        num_workers=0,
    )

    model = ChessCNN().to(device)
    model.load_state_dict(torch.load(
        root / "chess_model_cnn.pt",
        map_location=device,
        weights_only=True,
    ))

    # 用较小学习率微调；优化器重新初始化
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001
    )
    loss_function = nn.CrossEntropyLoss()

    print(f"训练样本：{len(train_dataset):,}")
    print("开始微调……", flush=True)

    for epoch in range(1, 3):
        print(f"\n===== 微调第 {epoch}/2 轮 =====", flush=True)

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

        # 两轮分别保留，之后检查普通表现与战术表现
        output = root / f"chess_model_mate_epoch{epoch}.pt"
        torch.save(model.state_dict(), output)
        print("已保存：", output)

    print("\n微调完成！")


if __name__ == "__main__":
    main()