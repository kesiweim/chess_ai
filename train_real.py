import json
import random
import time
from pathlib import Path

import chess
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader

from board_encoder import encode_board
from move_encoder import encode_move
from model import ChessModel


ROOT = Path(__file__).resolve().parent
BATCH_SIZE = 256
EPOCHS = 3


class ChessDataset(IterableDataset):
    def __init__(self, path, shuffle=False):
        self.path = path
        self.shuffle = shuffle

    def __iter__(self):
        # 使用小缓冲区打乱样本，不把全部数据放进内存
        buffer = []

        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                sample = json.loads(line)

                if self.shuffle:
                    buffer.append(sample)

                    if len(buffer) < 4096:
                        continue

                    index = random.randrange(len(buffer))
                    sample = buffer[index]
                    buffer[index] = buffer[-1]
                    buffer.pop()

                yield self.encode(sample)

        if self.shuffle:
            random.shuffle(buffer)
            for sample in buffer:
                yield self.encode(sample)

    @staticmethod
    def encode(sample):
        board = chess.Board(sample["position"])
        move = chess.Move.from_uci(sample["next_move"])

        return encode_board(board), encode_move(move)


def run_epoch(model, loader, loss_function, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start = time.perf_counter()

    with torch.set_grad_enabled(training):
        for batch, (inputs, targets) in enumerate(loader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            scores = model(inputs)
            loss = loss_function(scores, targets)

            if training:
                loss.backward()
                optimizer.step()

            count = targets.size(0)
            total_samples += count
            total_loss += loss.item() * count
            total_correct += (
                scores.argmax(dim=1) == targets
            ).sum().item()

            if batch % 100 == 0:
                label = "训练" if training else "验证"
                elapsed = time.perf_counter() - start

                print(
                    f"  {label}：已处理 {total_samples:,} 条 | "
                    f"平均损失 {total_loss / total_samples:.4f} | "
                    f"用时 {elapsed:.0f} 秒",
                    flush=True,
                )

    if total_samples == 0:
        raise ValueError("数据文件没有样本")

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


def main():
    random.seed(42)
    torch.manual_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用的 CUDA 显卡")

    device = torch.device("cuda")

    train_loader = DataLoader(
        ChessDataset(ROOT / "train.jsonl", shuffle=True),
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    val_loader = DataLoader(
        ChessDataset(ROOT / "val.jsonl"),
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    # 加载真实对局模型，继续训练
    model = ChessModel().to(device)
    model.load_state_dict(torch.load(
        ROOT / "chess_model_real.pt",
        map_location=device,
        weights_only=True,
    ))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0003
    )
    loss_function = nn.CrossEntropyLoss()

    output_path = ROOT / "chess_model_continued.pt"

    print("先评估原模型，作为比较基准……", flush=True)
    best_val_loss, baseline_accuracy = run_epoch(
        model, val_loader, loss_function, device
    )

    # 先保存原模型；后续只有验证损失更低才更新
    torch.save(model.state_dict(), output_path)

    print(
        f"原模型验证损失：{best_val_loss:.4f} | "
        f"准确率：{baseline_accuracy:.2%}"
    )

    print("显卡：", torch.cuda.get_device_name(0))
    print("每批样本数：", BATCH_SIZE)
    print("开始训练，首次进度输出需要等待一会儿。", flush=True)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== 第 {epoch}/{EPOCHS} 轮 =====", flush=True)

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            print("已保存当前最佳模型：", output_path)

    print("\n训练完成！")


if __name__ == "__main__":
    main()