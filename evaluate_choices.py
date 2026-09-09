import json
from collections import defaultdict
from pathlib import Path

import chess
import torch

from board_encoder import encode_board
from model_residual_value import ResidualValueModel


VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


def key(fen):
    board = chess.Board(fen)
    return " ".join(board.fen().split()[:4])


def read_records(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line) for line in file if line.strip()
        ]


def material(board):
    return sum(
        value * (
            len(board.pieces(piece_type, board.turn))
            - len(board.pieces(piece_type, not board.turn))
        )
        for piece_type, value in VALUES.items()
    )


def main():
    root = Path(__file__).resolve().parent
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    training = read_records(root / "move_choices_train.jsonl")
    validation = read_records(root / "move_choices_val.jsonl")
    old_training = read_records(root / "value_train.jsonl")

    # 排除与价值训练局面、候选训练子局面相同的验证局面
    train_keys = {
        key(record["position"])
        for record in old_training + training
    }

    groups = defaultdict(list)
    for record in validation:
        groups[record["parent_id"]].append(record)

    model = ResidualValueModel().to(device)
    model.load_state_dict(torch.load(
        root / "chess_model_residual_value.pt",
        map_location=device,
        weights_only=True,
    ))
    model.eval()

    results = {
        "子力评价": {"hits": 0, "regret": 0.0},
        "新价值评价": {"hits": 0, "regret": 0.0},
    }
    count = 0
    skipped_overlap = 0
    skipped_incomplete = 0

    with torch.inference_mode():
        for records in groups.values():
            parent = chess.Board(records[0]["parent_position"])

            expected = {m.uci() for m in parent.legal_moves}
            actual = [r["move"] for r in records]

            # 必须包含全部合法候选，且没有重复记录
            if set(actual) != expected or len(actual) != len(expected):
                skipped_incomplete += 1
                continue

            # 整组排除，不能删掉某个候选后再比较最优走法
            if (
                key(parent.fen()) in train_keys
                or any(key(r["position"]) in train_keys for r in records)
            ):
                skipped_overlap += 1
                continue

            boards = []
            for record in records:
                child = parent.copy()
                child.push_uci(record["move"])

                if child.fen() != chess.Board(record["position"]).fen():
                    raise ValueError("父局面、走法与子局面不匹配")

                boards.append(child)

            inputs = torch.stack([
                encode_board(board) for board in boards
            ]).to(device)

            materials = torch.tensor(
                [material(board) for board in boards],
                dtype=torch.float32,
                device=device,
            )

            values, _ = model(inputs, materials)

            # 子局面轮到对方，转成原走棋方的视角要反号
            neural_scores = (-values).cpu().tolist()
            material_scores = (
                -torch.tanh(materials / 300.0)
            ).cpu().tolist()

            # 终局直接使用规则结果，不让模型猜
            for index, board in enumerate(boards):
                outcome = board.outcome()
                if outcome is not None:
                    if outcome.winner is None:
                        value = 0.0
                    else:
                        value = (
                            1.0 if outcome.winner == parent.turn else -1.0
                        )
                    neural_scores[index] = value
                    material_scores[index] = value

            targets = [r["parent_move_value"] for r in records]
            best_target = max(targets)

            count += 1

            for name, scores in [
                ("子力评价", material_scores),
                ("新价值评价", neural_scores),
            ]:
                # 同分时按 UCI 字符串固定选择，保证可重复
                chosen = min(
                    range(len(records)),
                    key=lambda i: (-scores[i], records[i]["move"]),
                )

                regret = max(0.0, best_target - targets[chosen])
                results[name]["regret"] += regret
                results[name]["hits"] += int(regret <= 1e-6)

    print(f"验证父局面总数：{len(groups)}")
    print(f"排除跨组重合：{skipped_overlap}")
    print(f"排除候选不完整：{skipped_incomplete}")
    print(f"实际评估父局面：{count}")

    if count == 0:
        print("没有可评估局面，请把以上统计发来。")
        return

    for name, result in results.items():
        print(
            f"{name}：标签最优命中 {result['hits']}/{count}，"
            f"平均价值损失 {result['regret'] / count:.4f}"
        )


if __name__ == "__main__":
    main()