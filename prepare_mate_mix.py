import json
from pathlib import Path

import chess


def main():
    root = Path(__file__).resolve().parent
    source = root / "train.jsonl"
    output = root / "train_mate_mix.jsonl"

    mating_samples = []
    total = 0

    # 保留全部原始训练样本
    with source.open("r", encoding="utf-8") as infile, \
            output.open("w", encoding="utf-8") as outfile:

        for line in infile:
            if not line.strip():
                continue

            sample = json.loads(line)
            outfile.write(json.dumps(sample) + "\n")
            total += 1

            board = chess.Board(sample["position"])
            board.push(chess.Move.from_uci(sample["next_move"]))

            if board.is_checkmate():
                mating_samples.append(sample)

            if total % 100000 == 0:
                print(f"已处理 {total:,} 条", flush=True)

        # 每条将死样本额外加入 19 份，总计出现 20 次
        for _ in range(19):
            for sample in mating_samples:
                outfile.write(json.dumps(sample) + "\n")

    added = len(mating_samples) * 19
    final_total = total + added

    print(f"\n原始样本：{total:,}")
    print(f"不同训练记录中的将死样本：{len(mating_samples):,}")
    print(f"额外加入：{added:,}")
    print(f"最终样本：{final_total:,}")

    if final_total:
        ratio = len(mating_samples) * 20 / final_total
        print(f"将死样本占比：{ratio:.2%}")

    print(f"已保存：{output}")


if __name__ == "__main__":
    main()