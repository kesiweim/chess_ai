import json
import random
import time
from pathlib import Path

import chess
import chess.engine
import torch

from model_cnn import ChessCNN
from model_residual_value import ResidualValueModel
from search_engine import SearchEngine, neural_evaluator, policy_order


def read_records(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line) for line in file if line.strip()
        ]


def position_key(fen):
    return " ".join(chess.Board(fen).fen().split()[:4])


def main():
    root = Path(__file__).resolve().parent
    torch.set_num_threads(4)
    device = torch.device("cpu")

    # 排除模型评分训练数据及候选训练子局面
    excluded = {
        position_key(r["position"])
        for name in ["value_train.jsonl", "move_choices_train.jsonl"]
        for r in read_records(root / name)
    }

    # 排除已经用于候选排序验证的父局面
    excluded.update(
        position_key(r["parent_position"])
        for r in read_records(root / "move_choices_val.jsonl")
    )

    samples = [
        r for r in read_records(root / "value_val.jsonl")
        if position_key(r["position"]) not in excluded
    ]

    random.Random(2026).shuffle(samples)
    samples = samples[:20]

    if not samples:
        raise ValueError("没有可用的对照局面")

    policy = ChessCNN().to(device)
    policy.load_state_dict(torch.load(
        root / "chess_model_balanced.pt",
        map_location=device,
        weights_only=True,
    ))
    policy.eval()

    value_model = ResidualValueModel().to(device)
    value_model.load_state_dict(torch.load(
        root / "chess_model_ranked_epoch4.pt",
        map_location=device,
        weights_only=True,
    ))
    value_model.eval()

    evaluator = neural_evaluator(value_model, device)

    stockfish_path = (
        root
        / "stockfish-windows-x86-64-universal"
        / "stockfish"
        / "stockfish-windows-x86-64-universal.exe"
    )

    # 预热，避免首次加载开销集中在第一题
    warmup = chess.Board()
    policy_order(warmup, policy, device)
    evaluator(warmup)

    results = []
    counts = {"material": 0, "neural": 0, "close": 0, "same": 0}

    with chess.engine.SimpleEngine.popen_uci(
        str(stockfish_path)
    ) as engine:
        engine.configure({"Threads": 2, "Hash": 256})

        for number, sample in enumerate(samples, start=1):
            board = chess.Board(sample["position"])
            preferred = policy_order(board, policy, device)
            choices = {}

            print(f"\n局面 {number}/{len(samples)}", flush=True)

            # 交替先后顺序，减少持续运行顺序的影响
            modes = ["material", "neural"]
            if number % 2 == 0:
                modes.reverse()

            for mode in modes:
                search = SearchEngine(
                    evaluator=evaluator if mode == "neural" else None,
                    seconds=5,
                    max_depth=5,
                )

                start = time.perf_counter()
                move, _, stats = search.choose(board, preferred)
                elapsed = time.perf_counter() - start

                choices[mode] = {
                    "move": move.uci(),
                    "san": board.san(move),
                    "depth": stats["depth"],
                    "seconds": elapsed,
                }

                print(
                    f"{mode}：{board.san(move)} | "
                    f"深度 {stats['depth']} | {elapsed:.2f} 秒",
                    flush=True,
                )

            unique_moves = list(dict.fromkeys(
                chess.Move.from_uci(choices[mode]["move"])
                for mode in ["material", "neural"]
            ))

            if len(unique_moves) == 1:
                verdict = "same"
                print("两版选择相同。", flush=True)
            else:
                # 在同一次 MultiPV 搜索中比较双方候选
                infos = engine.analyse(
                    board,
                    chess.engine.Limit(time=5),
                    root_moves=unique_moves,
                    multipv=2,
                    game=object(),
                )

                scores = {}
                for info in infos:
                    move = info["pv"][0].uci()
                    score = info["score"].pov(board.turn)
                    scores[move] = score

                for mode in ["material", "neural"]:
                    score = scores[choices[mode]["move"]]
                    choices[mode]["cp"] = score.score()
                    choices[mode]["mate"] = score.mate()

                material_score = scores[choices["material"]["move"]]
                neural_score = scores[choices["neural"]["move"]]

                if (
                    material_score.mate() is None
                    and neural_score.mate() is None
                ):
                    difference = (
                        neural_score.score() - material_score.score()
                    )

                    # 小于等于 0.30 兵的差距归为接近
                    if abs(difference) <= 30:
                        verdict = "close"
                    else:
                        verdict = "neural" if difference > 0 else "material"
                else:
                    if neural_score == material_score:
                        verdict = "close"
                    else:
                        verdict = (
                            "neural" if neural_score > material_score
                            else "material"
                        )

                print(
                    f"引擎评分（当前走棋方）："
                    f"子力 {material_score}，价值 {neural_score}",
                    flush=True,
                )

            counts[verdict] += 1
            results.append({
                "position": board.fen(),
                "choices": choices,
                "verdict": verdict,
            })

            # 每完成一题就保存；重新运行本程序会重新测试
            output = root / "search_comparison_cpu.json"
            output.write_text(
                json.dumps(
                    {"summary": counts, "results": results},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    print("\n===== 对照汇总 =====")
    print("选择相同：", counts["same"])
    print("不同走法但评分接近：", counts["close"])
    print("价值版更好：", counts["neural"])
    print("子力版更好：", counts["material"])

    for mode in ["material", "neural"]:
        average_depth = sum(
            r["choices"][mode]["depth"] for r in results
        ) / len(results)
        print(f"{mode} 平均完成深度：{average_depth:.2f}")

    print("详细结果：search_comparison.json")


if __name__ == "__main__":
    main()