import io
import json
from pathlib import Path

import chess
import chess.engine
import chess.pgn


PGN = """
[White "maxwl666"]
[Black "Salman911111"]
[Result "0-1"]

1. e4 d5 2. exd5 Qxd5 3. Nc3 Qa5 4. d4 Bf5
5. Nf3 Nc6 6. Bc4 e6 7. O-O O-O-O 8. Re1 Bg4
9. h3 Bh5 10. Be3 Nf6 11. Qd2 h6 12. Rad1 Bb4
13. a3 Bd6 14. Ne2 g5 15. c3 Ne4 16. Ng3 Nxd2
17. Rxd2 g4 18. hxg4 Bxg4 19. Rb1 Rdg8 20. Ne2 h5
21. b3 Qxa3 22. Nf4 h4 23. Nd3 h3 24. gxh3 Bxf3+
25. Kf1 Rxh3 26. Ra1 Rh1# 0-1
"""


def describe(score):
    mate = score.mate()

    if mate is not None:
        if mate > 0:
            return f"白方可强制将死，距离 {mate}"
        if mate < 0:
            return f"白方将被强制将死，距离 {abs(mate)}"
        return "将死终局"

    return f"{score.score() / 100:+.2f}"


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

    game = chess.pgn.read_game(io.StringIO(PGN))
    if game is None or game.errors:
        raise ValueError("棋谱解析失败")

    board = game.board()
    corrections = []
    report = []

    # 每个局面分别分析最佳走法与实际走法
    limit = chess.engine.Limit(time=2.0)

    with chess.engine.SimpleEngine.popen_uci(
        str(engine_path)
    ) as engine:
        engine.configure({"Threads": 2, "Hash": 256})
        print("引擎：", engine.id.get("name", "Stockfish"))
        print("分数均从白方角度解释，正数对白方有利。")
        print("开始分析……", flush=True)

        for actual_move in game.mainline_moves():
            if board.turn == chess.WHITE:
                number = board.fullmove_number
                actual_san = board.san(actual_move)

                best_info = engine.analyse(board, limit)
                best_move = best_info["pv"][0]
                best_score = best_info["score"].white()

                if best_move == actual_move:
                    actual_score = best_score
                else:
                    actual_info = engine.analyse(
                        board,
                        limit,
                        root_moves=[actual_move],
                    )
                    actual_score = actual_info["score"].white()

                best_san = board.san(best_move)
                reason = None
                cp_loss = None

                # 将死分数单独处理，不混同普通子力评分
                best_mate = best_score.mate()
                actual_mate = actual_score.mate()

                if best_move != actual_move:
                    if (
                        best_mate is not None
                        and best_mate > 0
                        and not (
                            actual_mate is not None
                            and actual_mate > 0
                        )
                    ):
                        reason = "可能错失强制将死"

                    elif (
                        actual_mate is not None
                        and actual_mate < 0
                        and best_mate is None
                    ):
                        reason = "可能允许对手强制将死"

                    elif best_mate is None and actual_mate is None:
                        cp_loss = (
                            best_score.score() - actual_score.score()
                        )
                        if cp_loss >= 100:
                            reason = "引擎估计损失至少一个兵的评分"

                entry = {
                    "move_number": number,
                    "position": board.fen(),
                    "actual_move": actual_move.uci(),
                    "suggested_move": best_move.uci(),
                    "actual_score": describe(actual_score),
                    "best_score": describe(best_score),
                    "centipawn_loss": cp_loss,
                    "reason": reason,
                }
                report.append(entry)

                print(
                    f"{number}. 实走 {actual_san}，建议 {best_san} | "
                    f"实走评分 {describe(actual_score)} | "
                    f"建议评分 {describe(best_score)}",
                    flush=True,
                )

                if reason:
                    corrections.append({
                        "position": board.fen(),
                        "next_move": best_move.uci(),
                        "actual_move": actual_move.uci(),
                        "move_number": number,
                        "reason": reason,
                    })
                    print(f"    待纠正：{reason}", flush=True)

            board.push(actual_move)

    with (root / "loss_analysis.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    with (root / "loss_corrections.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for sample in corrections:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n分析完成，生成 {len(corrections)} 条候选纠正样本。")
    print("完整报告：loss_analysis.json")
    print("候选样本：loss_corrections.jsonl")
    print("先审核结果，再用于训练。")


if __name__ == "__main__":
    main()