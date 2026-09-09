import hashlib
import json
import random
import time
from pathlib import Path

import chess
import chess.engine


SAMPLE_COUNT = 500
NODE_BUDGET = 100_000
CANDIDATES = 3


def main():
    root = Path(__file__).resolve().parent
    engine_path = (
        root
        / "stockfish-windows-x86-64-universal"
        / "stockfish"
        / "stockfish-windows-x86-64-universal.exe"
    )

    if not engine_path.is_file():
        raise FileNotFoundError(f"找不到引擎：{engine_path}")

    # 固定随机种子，重复运行时抽到相同的局面
    rng = random.Random(42)
    seen = set()
    samples = []
    unique_count = 0

    print("正在从训练集抽取不同局面……", flush=True)

    with (root / "train.jsonl").open(
        "r", encoding="utf-8"
    ) as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            sample = json.loads(line)
            fen = sample["position"]

            # 去重时忽略回合计数，保存时保留完整 FEN
            key_text = " ".join(fen.split()[:4])
            key = hashlib.sha256(key_text.encode()).hexdigest()

            if key in seen:
                continue
            seen.add(key)

            unique_count += 1
            record = {
                "position_id": key,
                "position": fen,
                "game_id": sample.get("game_id"),
            }

            # 均匀抽取不同局面，无需保留全部文本
            if len(samples) < SAMPLE_COUNT:
                samples.append(record)
            else:
                index = rng.randrange(unique_count)
                if index < SAMPLE_COUNT:
                    samples[index] = record

            if line_number % 100000 == 0:
                print(f"已读取 {line_number:,} 行", flush=True)

    output = root / "engine_labels_pilot.jsonl"

    # 已完成的局面不再分析
    completed = set()
    if output.exists():
        with output.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    record = json.loads(line)
                    if (
                        record["node_budget"] != NODE_BUDGET
                        or record["requested_candidates"] != CANDIDATES
                    ):
                        raise ValueError("已有文件的标注设置不同")
                    completed.add(record["position_id"])

    pending = [
        sample for sample in samples
        if sample["position_id"] not in completed
    ]

    print(f"抽取局面：{len(samples)}")
    print(f"本次待标注：{len(pending)}", flush=True)

    if not pending:
        print("这些局面已全部标注，无需重复运行。")
        return

    start = time.perf_counter()
    written = 0

    try:
        with chess.engine.SimpleEngine.popen_uci(
            str(engine_path)
        ) as engine, output.open("a", encoding="utf-8") as file:

            engine.configure({"Threads": 2, "Hash": 256})
            engine_name = engine.id.get("name", "Stockfish")
            print("引擎：", engine_name, flush=True)

            for sample in pending:
                board = chess.Board(sample["position"])

                if not board.is_valid() or board.is_game_over():
                    raise ValueError("抽取到无效或已结束的局面")

                infos = engine.analyse(
                    board,
                    chess.engine.Limit(nodes=NODE_BUDGET),
                    multipv=min(
                        CANDIDATES, board.legal_moves.count()
                    ),
                    game=object(),
                )

                candidates = []

                for info in sorted(
                    infos, key=lambda item: item.get("multipv", 1)
                ):
                    move = info["pv"][0]
                    if move not in board.legal_moves:
                        raise ValueError("引擎返回非法走法")

                    score = info["score"].pov(board.turn)

                    candidates.append({
                        "move": move.uci(),
                        "cp": score.score(),
                        "mate": score.mate(),
                        "depth": info.get("depth"),
                        "pv": [m.uci() for m in info["pv"]],
                    })

                record = {
                    **sample,
                    "engine": engine_name,
                    "score_pov": "side_to_move",
                    "node_budget": NODE_BUDGET,
                    "requested_candidates": CANDIDATES,
                    "next_move": candidates[0]["move"],
                    "candidates": candidates,
                }

                file.write(json.dumps(record) + "\n")
                file.flush()
                written += 1

                if written == 1 or written % 25 == 0:
                    elapsed = time.perf_counter() - start
                    print(
                        f"本次完成 {written}/{len(pending)} | "
                        f"平均 {elapsed / written:.2f} 秒/局面",
                        flush=True,
                    )

    except KeyboardInterrupt:
        print("\n已中断，已写入的完整记录保留。")
        return

    elapsed = time.perf_counter() - start
    print(f"\n标注完成，本次新增 {written} 条")
    print(f"耗时：{elapsed:.1f} 秒")
    print(f"平均：{elapsed / written:.2f} 秒/局面")
    print("保存位置：", output)


if __name__ == "__main__":
    main()