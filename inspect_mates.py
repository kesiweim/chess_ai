import json
from pathlib import Path

import chess


def inspect(path):
    total = 0
    mating_targets = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            sample = json.loads(line)
            board = chess.Board(sample["position"])
            move = chess.Move.from_uci(sample["next_move"])

            # 检查棋谱实际走的这一步是否将死
            board.push(move)

            total += 1
            if board.is_checkmate():
                mating_targets += 1

            if total % 100000 == 0:
                print(
                    f"{path.name}：已检查 {total:,} 条",
                    flush=True,
                )

    print(f"\n文件：{path.name}")
    print(f"样本总数：{total:,}")
    print(f"目标走法直接将死的样本：{mating_targets:,}")

    if total:
        print(f"占比：{mating_targets / total:.4%}")

    print()


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    inspect(root / "train.jsonl")
    inspect(root / "val.jsonl")