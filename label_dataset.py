import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import chess
import chess.engine


NODE_BUDGET = 100_000
CANDIDATES = 3


def position_id(fen):
    # 与之前版本保持一致：去重时忽略回合计数
    text = " ".join(fen.split()[:4])
    return hashlib.sha256(text.encode()).hexdigest()


def collect_samples(source, split, sample_count):
    rng = random.Random(42)
    seen = set()
    samples = []
    unique_count = 0

    print("正在从训练集抽取不同局面……", flush=True)

    with source.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line_number % 100000 == 0:
                print(f"已读取 {line_number:,} 行", flush=True)

            if not line.strip():
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"train.jsonl 第 {line_number} 行格式错误"
                ) from error

            game_id = sample.get("game_id")
            if not game_id:
                raise ValueError(
                    f"第 {line_number} 行缺少 game_id"
                )

            # 同一盘对局始终属于同一组
            bucket = int(
                hashlib.sha256(game_id.encode()).hexdigest(), 16
            ) % 10

            belongs_to_val = bucket == 0
            if belongs_to_val != (split == "val"):
                continue

            fen = sample["position"]
            key = position_id(fen)

            if key in seen:
                continue
            seen.add(key)

            unique_count += 1
            record = {
                "position_id": key,
                "position": fen,
                "game_id": game_id,
            }

            # 保持原来的抽样顺序，便于复用已完成的标注
            if len(samples) < sample_count:
                samples.append(record)
            else:
                index = rng.randrange(unique_count)
                if index < sample_count:
                    samples[index] = record

    print(f"该组不同局面：{unique_count:,}")
    print(f"抽取局面：{len(samples):,}", flush=True)
    return samples


def read_completed(output):
    completed = set()

    if not output.exists():
        return completed

    with output.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{output.name} 第 {line_number} 行不完整。"
                    "请保留文件并把报错发来，不要删除已有数据。"
                ) from error

            if (
                record.get("node_budget") != NODE_BUDGET
                or record.get("requested_candidates") != CANDIDATES
            ):
                raise ValueError(
                    f"{output.name} 中已有记录的搜索设置不同"
                )

            completed.add(record["position_id"])

    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        required=True,
    )
    args = parser.parse_args()

    sample_count = 50_000 if args.split == "train" else 5_000
    root = Path(__file__).resolve().parent

    source = root / "train.jsonl"
    output = root / f"engine_labels_{args.split}.jsonl"
    skipped_output = root / f"engine_labels_{args.split}_skipped.json"

    engine_path = (
        root
        / "stockfish-windows-x86-64-universal"
        / "stockfish"
        / "stockfish-windows-x86-64-universal.exe"
    )

    if not source.is_file():
        raise FileNotFoundError(f"找不到训练数据：{source}")

    if not engine_path.is_file():
        raise FileNotFoundError(f"找不到引擎：{engine_path}")

    samples = collect_samples(source, args.split, sample_count)
    completed = read_completed(output)

    # 在启动引擎前检查局面，异常局面记录后跳过
    valid_samples = []
    skipped = []

    for sample in samples:
        try:
            board = chess.Board(sample["position"])

            if not board.is_valid():
                reason = f"无效局面，状态码 {int(board.status())}"
            else:
                outcome = board.outcome()
                reason = (
                    f"已结束：{outcome.termination.name}"
                    if outcome is not None else None
                )

        except ValueError:
            reason = "FEN 无法解析"

        if reason is not None:
            skipped.append({**sample, "reason": reason})
        else:
            valid_samples.append(sample)

    # 这只是跳过原因报告，重复运行会更新
    with skipped_output.open("w", encoding="utf-8") as file:
        json.dump(skipped, file, ensure_ascii=False, indent=2)

    pending = [
        sample for sample in valid_samples
        if sample["position_id"] not in completed
    ]

    already_done = len(valid_samples) - len(pending)

    print(f"跳过异常或终局：{len(skipped):,}")
    print(f"本批已标注：{already_done:,}")
    print(f"本次待标注：{len(pending):,}", flush=True)

    if not pending:
        print("本批有效局面已全部标注。")
        print("数据文件：", output)
        return

    # 兼容末行是完整 JSON、但没有换行符的旧文件
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

            if needs_newline:
                file.write("\n")
                file.flush()

            engine.configure({"Threads": 2, "Hash": 256})
            engine_name = engine.id.get("name", "Stockfish")

            print("引擎：", engine_name)
            print("开始标注……", flush=True)

            for sample in pending:
                board = chess.Board(sample["position"])

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
                    infos,
                    key=lambda item: item.get("multipv", 1),
                ):
                    pv = info.get("pv")
                    if not pv or "score" not in info:
                        raise ValueError("引擎未返回完整分析结果")

                    move = pv[0]
                    if move not in board.legal_moves:
                        raise ValueError("引擎返回了非法走法")

                    score = info["score"].pov(board.turn)

                    candidates.append({
                        "move": move.uci(),
                        "cp": score.score(),
                        "mate": score.mate(),
                        "depth": info.get("depth"),
                        "pv": [m.uci() for m in pv],
                    })

                if not candidates:
                    raise ValueError("引擎没有返回候选走法")

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

                if written == 1 or written % 100 == 0:
                    elapsed = time.perf_counter() - start
                    average = elapsed / written
                    remaining_minutes = (
                        (len(pending) - written) * average / 60
                    )

                    print(
                        f"本次完成 {written:,}/{len(pending):,} | "
                        f"平均 {average:.3f} 秒/局面 | "
                        f"预计剩余 {remaining_minutes:.1f} 分钟",
                        flush=True,
                    )

    except KeyboardInterrupt:
        print(
            "\n已停止。再次运行同一命令可接着标注；"
            "已写入的记录保留。",
            flush=True,
        )
        return

    elapsed = time.perf_counter() - start

    print(f"\n标注完成，本次新增 {written:,} 条")
    print(f"本批累计完成：{already_done + written:,} 条")
    print(f"跳过局面：{len(skipped):,} 条")
    print(f"耗时：{elapsed:.1f} 秒")
    print(f"平均：{elapsed / written:.3f} 秒/局面")
    print("保存位置：", output)


if __name__ == "__main__":
    main()