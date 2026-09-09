import json
from pathlib import Path

import chess
import torch

from board_encoder import encode_board
from move_encoder import encode_move
from model_cnn import ChessCNN


BATCH_SIZE = 256


def read_batches(path):
    batch = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            batch.append(json.loads(line))

            if len(batch) == BATCH_SIZE:
                yield batch
                batch = []

    if batch:
        yield batch


def main():
    root = Path(__file__).resolve().parent
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ChessCNN().to(device)
    model.load_state_dict(torch.load(
        root / "chess_model_corrected.pt",
        map_location=device,
        weights_only=True,
    ))
    model.eval()

    total = 0
    raw_legal = 0
    raw_correct = 0
    legal_correct = 0
    top3_correct = 0

    print("使用设备：", device)
    print("开始评估，请等待进度输出……", flush=True)

    with torch.inference_mode():
        for batch_number, samples in enumerate(
            read_batches(root / "val.jsonl"),
            start=1,
        ):
            boards = [
                chess.Board(sample["position"])
                for sample in samples
            ]

            inputs = torch.stack([
                encode_board(board) for board in boards
            ]).to(device)

            # 一次预测一批，随后在 CPU 上统计
            scores = model(inputs).cpu()
            raw_predictions = scores.argmax(dim=1).tolist()

            for row, (board, sample) in enumerate(
                zip(boards, samples)
            ):
                target_move = chess.Move.from_uci(
                    sample["next_move"]
                )
                legal_moves = list(board.legal_moves)

                if target_move not in legal_moves:
                    raise ValueError(
                        "验证样本中的目标走法不合法，请检查数据"
                    )

                target_id = encode_move(target_move)
                legal_ids = [
                    encode_move(move) for move in legal_moves
                ]

                # 不过滤时，模型首选是否合法、是否与棋谱一致
                raw_id = raw_predictions[row]
                raw_legal += int(raw_id in legal_ids)
                raw_correct += int(raw_id == target_id)

                # 只在合法走法中选分数最高的三个
                legal_scores = scores[row, legal_ids]
                best_indices = legal_scores.topk(
                    min(3, len(legal_ids))
                ).indices.tolist()

                recommended_ids = [
                    legal_ids[index] for index in best_indices
                ]

                legal_correct += int(
                    recommended_ids[0] == target_id
                )
                top3_correct += int(
                    target_id in recommended_ids
                )

                total += 1

            if batch_number % 20 == 0:
                print(f"已评估 {total:,} 条样本", flush=True)

    if total == 0:
        raise ValueError("验证文件中没有样本")

    print(f"\n评估完成，共 {total:,} 条样本")
    print(f"原始首选合法率：{raw_legal / total:.2%}")
    print(f"原始首选准确率：{raw_correct / total:.2%}")
    print(f"合法走法首选准确率：{legal_correct / total:.2%}")
    print(f"合法走法前三命中率：{top3_correct / total:.2%}")


if __name__ == "__main__":
    main()