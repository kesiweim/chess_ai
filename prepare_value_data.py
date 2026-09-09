import json
import math
from pathlib import Path

import chess


def position_key(board):
    # 忽略回合计数；无实际合法吃过路兵时统一表示为 -
    return " ".join(board.fen().split()[:4])


def load_records(path):
    records = []
    seen = set()
    duplicates = 0

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("score_pov") != "side_to_move":
                raise ValueError(
                    f"{path.name} 第 {line_number} 行评分视角不符"
                )

            board = chess.Board(record["position"])
            if not board.is_valid() or board.is_game_over():
                raise ValueError(
                    f"{path.name} 第 {line_number} 行局面异常"
                )

            key = position_key(board)
            if key in seen:
                duplicates += 1
                continue

            candidates = record["candidates"]
            if not candidates:
                raise ValueError("缺少引擎候选走法")

            best = candidates[0]
            move = chess.Move.from_uci(best["move"])

            if move not in board.legal_moves:
                raise ValueError("引擎目标走法不合法")

            cp = best.get("cp")
            mate = best.get("mate")

            if mate is not None:
                if mate == 0:
                    raise ValueError("非终局出现 mate=0，请检查")
                value = 1.0 if mate > 0 else -1.0
            elif cp is not None:
                # ±300 分约映射到 ±0.76
                value = math.tanh(cp / 300.0)
            else:
                raise ValueError("缺少评分")

            if not math.isfinite(value):
                raise ValueError("评分不是有限数值")

            seen.add(key)
            records.append({
                "key": key,
                "position": board.fen(),
                "game_id": record.get("game_id"),
                "next_move": move.uci(),
                "value": value,
                "cp": cp,
                "mate": mate,
            })

    return records, duplicates


def save_records(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            # key 仅用于去重，无需存入训练数据
            output = {
                key: value for key, value in record.items()
                if key != "key"
            }
            file.write(json.dumps(output) + "\n")


def main():
    root = Path(__file__).resolve().parent

    train, train_duplicates = load_records(
        root / "engine_labels_train.jsonl"
    )
    val, val_duplicates = load_records(
        root / "engine_labels_val.jsonl"
    )

    # 检查来源对局是否跨组
    train_games = {r["game_id"] for r in train if r["game_id"]}
    val_games = {r["game_id"] for r in val if r["game_id"]}

    if train_games & val_games:
        raise ValueError("发现来源对局跨组，请先检查划分")

    # 不同对局也可能走到相同局面，移除验证集中的重合项
    train_keys = {r["key"] for r in train}
    clean_val = [r for r in val if r["key"] not in train_keys]
    overlap = len(val) - len(clean_val)

    if not train or not clean_val:
        raise ValueError("清理后至少一个集合为空")

    save_records(root / "value_train.jsonl", train)
    save_records(root / "value_val.jsonl", clean_val)

    print("数据准备完成")
    print(f"训练组内去重：{train_duplicates}")
    print(f"验证组内去重：{val_duplicates}")
    print(f"移除跨组重复局面：{overlap}")
    print(f"最终训练样本：{len(train):,}")
    print(f"最终验证样本：{len(clean_val):,}")

    for name, records in [("训练", train), ("验证", clean_val)]:
        mate_count = sum(r["mate"] is not None for r in records)
        mean_value = sum(r["value"] for r in records) / len(records)
        print(
            f"{name}：将死评分 {mate_count} 条，"
            f"平均价值目标 {mean_value:+.4f}"
        )

    print("已保存：value_train.jsonl、value_val.jsonl")


if __name__ == "__main__":
    main()