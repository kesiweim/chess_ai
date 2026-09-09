from pathlib import Path

import chess
import chess.engine
import torch

from model_cnn import ChessCNN
from model_residual_value import ResidualValueModel
from search_engine import SearchEngine, neural_evaluator, policy_order


CASES = [
    (
        "第一盘，第7回合",
        "r1bqkb1r/ppppppnp/6p1/3PP3/3Q2P1/8/PPP2P1P/RNB1KBNR w KQkq - 1 7",
        "e1e2",
    ),
    (
        "第二盘，第3回合",
        "rnbqkbnr/ppp1pp1p/6p1/3p4/3P4/3Q4/PPP1PPPP/RNB1KBNR w KQkq - 0 3",
        "e1d1",
    ),
    (
        "第三盘，第3回合",
        "rnbqk1nr/pppp1ppp/4p3/8/1b1P4/3Q4/PPP1PPPP/RNB1KBNR w KQkq - 2 3",
        "e1d1",
    ),
]


def describe(info):
    score = info["score"].white()
    mate = score.mate()

    if mate is not None:
        return f"将死分数 {mate}"

    return f"{score.score() / 100:+.2f}"


def main():
    root = Path(__file__).resolve().parent
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

    engine_path = (
        root
        / "stockfish-windows-x86-64-universal"
        / "stockfish"
        / "stockfish-windows-x86-64-universal.exe"
    )

    with chess.engine.SimpleEngine.popen_uci(
        str(engine_path)
    ) as stockfish:
        stockfish.configure({"Threads": 2, "Hash": 256})
        print("引擎：", stockfish.id.get("name"))

        for title, fen, actual_uci in CASES:
            board = chess.Board(fen)
            actual = chess.Move.from_uci(actual_uci)
            preferred = policy_order(board, policy, device)

            print(f"\n===== {title} =====", flush=True)
            print("实际走法：", board.san(actual))
            print(
                "纯策略前三：",
                ", ".join(board.san(m) for m in preferred[:3]),
            )

            # 同样的三手深度上限，给足一些时间完成
            # 若没有完成三手，会显示实际完成深度
            choices = {"实际走法": actual}

            for name, evaluation in [
                ("子力搜索", None),
                ("价值搜索", evaluator),
            ]:
                search = SearchEngine(
                    evaluator=evaluation,
                    seconds=30,
                    max_depth=3,
                )
                move, _, stats = search.choose(board, preferred)
                choices[name] = move

                print(
                    f"{name}：{board.san(move)} | "
                    f"完成深度 {stats['depth']} | "
                    f"耗时 {stats['seconds']:.2f} 秒",
                    flush=True,
                )

            best_info = stockfish.analyse(
                board,
                chess.engine.Limit(time=5),
                game=object(),
            )
            best_move = best_info["pv"][0]
            print(
                f"Stockfish建议：{board.san(best_move)} | "
                f"白方评分 {describe(best_info)}",
                flush=True,
            )

            # 独立分析各个实际候选，相同走法只计算一次
            analysed = {}

            for name, move in choices.items():
                if move not in analysed:
                    analysed[move] = stockfish.analyse(
                        board,
                        chess.engine.Limit(time=5),
                        root_moves=[move],
                        game=object(),
                    )

                print(
                    f"{name} {board.san(move)}："
                    f"白方评分 {describe(analysed[move])}",
                    flush=True,
                )


if __name__ == "__main__":
    main()