import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import chess
import chess.engine


NODES = 100_000


def load_roots(path, count):
    with path.open("r", encoding="utf-8") as file:
        records = [
            json.loads(line) for line in file if line.strip()
        ]

    # 固定顺序，方便中断后继续
    random.Random(42).shuffle(records)
    return records[:count]


def score_to_value(cp, mate):
    if mate is not None:
        return 1.0 if mate > 0 else -1.0
    return math.tanh(cp / 300.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=["train", "val"], required=True
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    count = 2_000 if args.split == "train" else 200

    source = root / f"value_{args.split}.jsonl"
    output = root / f"move_choices_{args.split}.jsonl"

    engine_path = (
        root
        / "stockfish-windows-x86-64-universal"
        / "stockfish"
        / "stockfish-windows-x86-64-universal.exe"
    )

    if not engine_path.is_file():
        raise FileNotFoundError(f"找不到引擎：{engine_path}")

    roots = load_roots(source, count)

    completed = set()
    if output.exists():
        with output.open("r", encoding="utf-8") as file:
            for number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"输出文件第 {number} 行不完整，请保留文件并反馈"
                    ) from error

                if record["node_budget"] != NODES:
                    raise ValueError("已有记录的搜索预算不一致")

                completed.add(record["sample_id"])

    # 在内存中准备待分析的父局面和走法
    pending = []

    for record in roots:
        board = chess.Board(record["position"])

        if not board.is_valid() or board.is_game_over():
            continue

        parent_fen = board.fen()
        parent_id = hashlib.sha256(
            parent_fen.encode()
        ).hexdigest()

        for move in board.legal_moves:
            sample_id = f"{parent_id}:{move.uci()}"

            if sample_id not in completed:
                pending.append((
                    record, parent_fen, parent_id, move.uci(), sample_id
                ))

    print(f"父局面数量：{len(roots)}")
    print(f"本次待分析走法：{len(pending):,}", flush=True)

    if not pending:
        print("这批数据已经完成。")
        return

    # 若旧文件末行完整但缺少换行，补上换行再追加
    needs_newline = False
    if output.exists() and output.stat().st_size:
        with output.open("rb") as file:
            file.seek(-1, 2)
            needs_newline = file.read(1) != b"\n"

    written = 0
    start = time.perf_counter()

    try:
        with chess.engine.SimpleEngine.popen_uci(
            str(engine_path)
        ) as engine, output.open("a", encoding="utf-8") as file:

            engine.configure({"Threads": 2, "Hash": 256})

            if needs_newline:
                file.write("\n")
                file.flush()

            for source_record, parent_fen, parent_id, uci, sid in pending:
                board = chess.Board(parent_fen)
                mover = board.turn
                board.push_uci(uci)

                # 走完后轮到对方：下面保存的是对方视角的价值
                outcome = board.outcome()
                cp = None
                mate = None
                depth = None

                if outcome is not None:
                    if outcome.winner is None:
                        child_value = 0.0
                    else:
                        child_value = (
                            1.0 if outcome.winner == board.turn else -1.0
                        )
                else:
                    info = engine.analyse(
                        board,
                        chess.engine.Limit(nodes=NODES),
                        game=object(),
                    )
                    score = info["score"].pov(board.turn)
                    cp = score.score()
                    mate = score.mate()
                    depth = info.get("depth")

                    if cp is None and mate is None:
                        raise ValueError("引擎没有返回评分")

                    child_value = score_to_value(cp, mate)

                record = {
                    "sample_id": sid,
                    "parent_id": parent_id,
                    "parent_position": parent_fen,
                    "game_id": source_record.get("game_id"),
                    "move": uci,
                    "position": board.fen(),
                    "score_pov": "side_to_move",
                    "value": child_value,
                    # 评价这一步对原走棋方的好坏，需要反号
                    "parent_move_value": -child_value,
                    "cp": cp,
                    "mate": mate,
                    "terminal": outcome is not None,
                    "depth": depth,
                    "node_budget": NODES,
                    "engine": engine.id.get("name", "Stockfish"),
                }

                file.write(json.dumps(record) + "\n")
                file.flush()
                written += 1

                if written == 1 or written % 100 == 0:
                    elapsed = time.perf_counter() - start
                    print(
                        f"完成 {written:,}/{len(pending):,} | "
                        f"平均 {elapsed / written:.3f} 秒/走法",
                        flush=True,
                    )

    except KeyboardInterrupt:
        print("\n已停止，完整记录已保留，重复同一命令可继续。")
        return

    print(f"\n本次新增：{written:,} 条")
    print(f"耗时：{time.perf_counter() - start:.1f} 秒")
    print("保存位置：", output)


if __name__ == "__main__":
    main()