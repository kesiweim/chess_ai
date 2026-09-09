import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlparse

import chess
import zstandard as zstd


def position_key(board):
    # 忽略回合计数，避免同一棋子布局因计数不同而重复
    text = " ".join(board.fen().split()[:4])
    return hashlib.sha256(text.encode()).digest()


def main():
    root = Path(__file__).resolve().parent

    # 排除已经出现在原始对局数据中的局面
    excluded = set()

    for name in ["train.jsonl", "val.jsonl"]:
        print(f"读取旧数据：{name}", flush=True)

        with (root / name).open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    sample = json.loads(line)
                    board = chess.Board(sample["position"])
                    excluded.add(position_key(board))

    seen = set()
    records = {"train": [], "val": [], "test": []}
    scanned = 0
    candidates = 0
    invalid = 0
    overlaps = 0
    duplicates = 0

    source = root / "lichess_db_puzzle.csv.zst"

    print("开始读取压缩题库……", flush=True)

    with source.open("rb") as compressed:
        with zstd.ZstdDecompressor().stream_reader(compressed) as stream:
            with io.TextIOWrapper(stream, encoding="utf-8") as text:
                reader = csv.DictReader(text)

                required = {"PuzzleId", "FEN", "Moves", "Themes", "GameUrl"}
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError("题库表头与预期不符")

                for row in reader:
                    scanned += 1

                    if scanned % 200000 == 0:
                        kept = sum(len(items) for items in records.values())
                        print(
                            f"已读取 {scanned:,} 题，保留 {kept:,} 题",
                            flush=True,
                        )

                    if "mateIn1" not in row["Themes"].split():
                        continue

                    candidates += 1

                    try:
                        board = chess.Board(row["FEN"])
                        moves = row["Moves"].split()

                        if not board.is_valid() or len(moves) < 2:
                            raise ValueError("局面或走法不完整")

                        # 第一步是对手落子，执行后才是待解局面
                        setup = chess.Move.from_uci(moves[0])
                        if setup not in board.legal_moves:
                            raise ValueError("前置走法不合法")
                        board.push(setup)

                        fen = board.fen()
                        key = position_key(board)

                        # 第二步是题库提供的答案
                        answer = chess.Move.from_uci(moves[1])
                        if answer not in board.legal_moves:
                            raise ValueError("答案不合法")

                        board.push(answer)
                        if not board.is_checkmate():
                            raise ValueError("答案没有将死")

                    except ValueError:
                        invalid += 1
                        continue

                    if key in excluded:
                        overlaps += 1
                        continue

                    if key in seen:
                        duplicates += 1
                        continue

                    seen.add(key)

                    # 用来源对局编号分组，避免同盘题目跨集合
                    parts = urlparse(row["GameUrl"]).path.strip("/").split("/")
                    game_id = parts[0][:8]

                    if not game_id:
                        game_id = row["PuzzleId"]

                    bucket = int(
                        hashlib.sha256(game_id.encode()).hexdigest(), 16
                    ) % 100

                    if bucket < 80:
                        split = "train"
                    elif bucket < 90:
                        split = "val"
                    else:
                        split = "test"

                    records[split].append({
                        "puzzle_id": row["PuzzleId"],
                        "game_id": game_id,
                        "position": fen,
                        "next_move": answer.uci(),
                    })

    if any(not items for items in records.values()):
        raise ValueError("至少一个集合为空，请检查题库和筛选结果")

    # 完整读取成功后再写出数据
    for split, items in records.items():
        path = root / f"puzzles_mate_{split}.jsonl"

        with path.open("w", encoding="utf-8") as file:
            for item in items:
                file.write(json.dumps(item) + "\n")

    print("\n处理完成")
    print(f"读取题目：{scanned:,}")
    print(f"一步将死候选：{candidates:,}")
    print(f"核验失败：{invalid:,}")
    print(f"与旧数据重合：{overlaps:,}")
    print(f"重复局面：{duplicates:,}")

    for split, items in records.items():
        print(f"{split}：{len(items):,} 条")

    print("已保存三个 puzzles_mate_*.jsonl 文件")


if __name__ == "__main__":
    main()