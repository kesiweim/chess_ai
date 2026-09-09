import json
import random
from pathlib import Path

import chess
import torch
from torch import nn

from board_encoder import encode_board
from move_encoder import encode_move, decode_move
from model_cnn import ChessCNN


def encode_batch(samples, device):
    inputs = torch.stack([
        encode_board(chess.Board(s["position"]))
        for s in samples
    ]).to(device)

    targets = torch.tensor([
        encode_move(chess.Move.from_uci(s["next_move"]))
        for s in samples
    ], dtype=torch.long, device=device)

    return inputs, targets


def show_predictions(model, samples, device):
    model.eval()

    with torch.inference_mode():
        inputs, _ = encode_batch(samples, device)
        scores = model(inputs).cpu()

    for row, sample in enumerate(samples):
        board = chess.Board(sample["position"])
        legal_ids = [
            encode_move(move) for move in board.legal_moves
        ]
        index = scores[row, legal_ids].argmax().item()
        predicted = decode_move(legal_ids[index])
        target = chess.Move.from_uci(sample["next_move"])

        print(
            f"第 {sample['move_number']} 回合："
            f"预测 {board.san(predicted)}，"
            f"纠正目标 {board.san(target)}",
            flush=True,
        )


def main():
    root = Path(__file__).resolve().parent
    rng = random.Random(42)
    torch.manual_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA 显卡")
    device = torch.device("cuda")

    selected = {6, 10, 11, 12, 14, 15, 16}

    with (root / "loss_corrections.jsonl").open(
        "r", encoding="utf-8"
    ) as file:
        corrections = [
            json.loads(line) for line in file if line.strip()
        ]

    corrections = [
        s for s in corrections if s["move_number"] in selected
    ]

    if {s["move_number"] for s in corrections} != selected:
        raise ValueError("纠正文件缺少预期回合，请检查文件")

    for sample in corrections:
        board = chess.Board(sample["position"])
        move = chess.Move.from_uci(sample["next_move"])
        if move not in board.legal_moves:
            raise ValueError("发现非法纠正走法")

    # 从普通训练集均匀抽取最多 8192 条作为复习样本
    replay = []
    count = 0

    print("正在抽取普通复习样本……", flush=True)
    with (root / "train.jsonl").open(
        "r", encoding="utf-8"
    ) as file:
        for line in file:
            if not line.strip():
                continue

            count += 1
            if len(replay) < 8192:
                replay.append(json.loads(line))
            else:
                index = rng.randrange(count)
                if index < len(replay):
                    replay[index] = json.loads(line)

    if len(replay) < 128:
        raise ValueError("普通训练数据不足")

    model = ChessCNN().to(device)
    model.load_state_dict(torch.load(
        root / "chess_model_balanced.pt",
        map_location=device,
        weights_only=True,
    ))

    print("\n纠正前：")
    show_predictions(model, corrections, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()
    correction_x, correction_y = encode_batch(corrections, device)

    # 固定做 100 次小幅更新，不追求反复训练至全部记住
    model.train()

    for step in range(1, 101):
        ordinary = rng.sample(replay, 128)
        inputs, targets = encode_batch(ordinary, device)

        optimizer.zero_grad(set_to_none=True)

        ordinary_loss = criterion(model(inputs), targets)
        correction_loss = criterion(
            model(correction_x), correction_y
        )

        loss = ordinary_loss + 0.25 * correction_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0:
            print(
                f"更新 {step}/100 | "
                f"普通损失 {ordinary_loss.item():.4f} | "
                f"纠正损失 {correction_loss.item():.4f}",
                flush=True,
            )

    print("\n纠正后：")
    show_predictions(model, corrections, device)

    output = root / "chess_model_corrected.pt"
    torch.save(model.state_dict(), output)
    print("\n候选模型已保存：", output)


if __name__ == "__main__":
    main()