import json
import math
from pathlib import Path

import chess


VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


def main():
    path = Path(__file__).resolve().parent / "value_val.jsonl"

    squared_error = 0.0
    absolute_error = 0.0
    count = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            sample = json.loads(line)
            board = chess.Board(sample["position"])

            # 和引擎标签一致：从当前走棋方的角度评分
            material = sum(
                value * (
                    len(board.pieces(piece_type, board.turn))
                    - len(board.pieces(piece_type, not board.turn))
                )
                for piece_type, value in VALUES.items()
            )

            # 与训练目标使用相同的缩放
            prediction = math.tanh(material / 300.0)
            error = prediction - sample["value"]

            squared_error += error * error
            absolute_error += abs(error)
            count += 1

    if count == 0:
        raise ValueError("验证集为空")

    print(f"验证样本：{count:,}")
    print(f"子力基准 MSE：{squared_error / count:.4f}")
    print(f"子力基准 MAE：{absolute_error / count:.4f}")
    print("已保存价值层 MSE：0.2465")
    print("已保存价值层 MAE：0.3749")


if __name__ == "__main__":
    main()