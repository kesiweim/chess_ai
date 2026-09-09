import json
from pathlib import Path

import chess
import torch
from torch import nn

from board_encoder import encode_board
from move_encoder import encode_move, decode_move
from model import ChessModel


def main():
    torch.manual_seed(42)

    device = torch.device("cuda")
    root = Path(__file__).resolve().parent

    # 读取训练样本
    with (root / "samples.json").open("r", encoding="utf-8") as file:
        samples = json.load(file)

    # 输入：局面；目标：棋谱中的走法编号
    inputs = torch.stack([
        encode_board(chess.Board(sample["position"]))
        for sample in samples
    ]).to(device)

    targets = torch.tensor([
        encode_move(chess.Move.from_uci(sample["next_move"]))
        for sample in samples
    ], dtype=torch.long, device=device)

    model = ChessModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    print("使用显卡：", torch.cuda.get_device_name(0))
    print("训练样本数：", len(samples))

    # 每一轮都用这 6 条样本学习一次
    model.train()

    for epoch in range(1, 301):
        optimizer.zero_grad()

        scores = model(inputs)
        loss = loss_function(scores, targets)

        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 25 == 0:
            # 用更新后的模型重新检查
            with torch.no_grad():
                updated_scores = model(inputs)
                updated_loss = loss_function(updated_scores, targets)
                predictions = updated_scores.argmax(dim=1)
                correct = (predictions == targets).sum().item()

            print(
                f"第 {epoch:3d} 轮 | "
                f"损失 {updated_loss.item():.4f} | "
                f"训练集答对 {correct}/{len(samples)}"
            )

    # 展示训练后的实际预测
    model.eval()

    with torch.no_grad():
        predictions = model(inputs).argmax(dim=1).cpu().tolist()

    print("\n训练后的预测：")

    for sample, prediction in zip(samples, predictions):
        predicted_move = decode_move(prediction).uci()
        expected_move = sample["next_move"]

        print(f"目标：{expected_move}，预测：{predicted_move}")

    # 保存模型学到的参数；重复运行会更新这个文件
    output_path = root / "chess_model.pt"
    torch.save(model.state_dict(), output_path)

    print(f"\n模型已保存：{output_path}")


if __name__ == "__main__":
    main()