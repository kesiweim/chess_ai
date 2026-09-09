import json
from pathlib import Path

import chess
import torch

from board_encoder import encode_board
from move_encoder import encode_move
from model_cnn import ChessCNN


def main():
    root = Path(__file__).resolve().parent
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ChessCNN().to(device)
    model.load_state_dict(torch.load(
        root / "chess_model_cnn.pt",
        map_location=device,
        weights_only=True,
    ))
    model.eval()

    total = 0
    top1_hits = 0
    top3_hits = 0
    scanned = 0

    print("正在筛选并评估将死样本……", flush=True)

    with (root / "val.jsonl").open(
        "r", encoding="utf-8"
    ) as file, torch.inference_mode():

        for line in file:
            if not line.strip():
                continue

            sample = json.loads(line)
            board = chess.Board(sample["position"])
            target = chess.Move.from_uci(sample["next_move"])

            scanned += 1
            if scanned % 20000 == 0:
                print(f"已检查 {scanned:,} 条样本", flush=True)

            # 只选出棋谱实际走出将死的局面
            board.push(target)
            target_is_mate = board.is_checkmate()
            board.pop()

            if not target_is_mate:
                continue

            legal_moves = list(board.legal_moves)
            mating_ids = set()

            # 找出全部正确答案：所有一步将死走法
            for move in legal_moves:
                board.push(move)
                if board.is_checkmate():
                    mating_ids.add(encode_move(move))
                board.pop()

            # 模型独立评分，不使用将死搜索结果
            inputs = encode_board(board).unsqueeze(0).to(device)
            scores = model(inputs)[0].cpu()

            legal_ids = [
                encode_move(move) for move in legal_moves
            ]
            indices = scores[legal_ids].topk(
                min(3, len(legal_ids))
            ).indices.tolist()

            recommended = [
                legal_ids[index] for index in indices
            ]

            total += 1
            top1_hits += int(recommended[0] in mating_ids)
            top3_hits += int(
                any(move_id in mating_ids for move_id in recommended)
            )

    if total == 0:
        print("没有找到符合条件的样本")
        return

    print(f"\n将死样本数：{total}")
    print(
        f"首选找到将死：{top1_hits}/{total}"
        f"（{top1_hits / total:.2%}）"
    )
    print(
        f"前三包含将死：{top3_hits}/{total}"
        f"（{top3_hits / total:.2%}）"
    )


if __name__ == "__main__":
    main()