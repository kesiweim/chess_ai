import json
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn
import torch

from model_cnn import ChessCNN
from model_residual_value import ResidualValueModel
from search_engine import SearchEngine, neural_evaluator, policy_order


OPENINGS = [
    ("开放开局", ["e4", "e5", "Nf3", "Nc6"]),
    ("后兵开局", ["d4", "d5", "c4", "e6"]),
    ("西西里", ["e4", "c5", "Nf3", "d6"]),
    ("法国防御", ["e4", "e6", "d4", "d5"]),
    ("卡罗康", ["e4", "c6", "d4", "d5"]),
    ("斯堪的纳维亚", ["e4", "d5", "exd5", "Qxd5"]),
    ("印度防御", ["d4", "Nf6", "c4", "e6"]),
    ("斯拉夫防御", ["d4", "d5", "c4", "c6"]),
    ("英国式开局", ["c4", "e5", "Nc3", "Nf6"]),
    ("列蒂开局", ["Nf3", "d5", "g3", "Nf6"]),
]

SECONDS = 5.0
DEPTH = 5
MAX_PLIES = 240


def draw_claim_details(board):
    """Return a verifiable claim, including an intended move when needed.
    The intended move is declared, not played; PGN stays at the claiming position.
    """
    if board.is_repetition(3):
        return {"reason": "threefold_repetition", "intended_move": None}
    if board.is_fifty_moves():
        return {"reason": "fifty_moves", "intended_move": None}
    for move in list(board.legal_moves):
        board.push(move)
        try:
            if board.is_repetition(3):
                return {"reason": "threefold_repetition", "intended_move": move.uci()}
            if board.is_fifty_moves():
                return {"reason": "fifty_moves", "intended_move": move.uci()}
        finally:
            board.pop()
    return None


def main():
    root = Path(__file__).resolve().parent
    torch.set_num_threads(4)
    device = torch.device("cpu")

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

    engines = {
        "neural": SearchEngine(
            neural_evaluator(value_model, device),
            seconds=SECONDS,
            max_depth=DEPTH,
            claim_draw=True,
        ),
        "material": SearchEngine(
            seconds=SECONDS,
            max_depth=DEPTH,
            claim_draw=True,
        ),
    }

    folder = root / "arena_games" / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    folder.mkdir(parents=True)

    summary = {
        "neural_wins": 0,
        "material_wins": 0,
        "draws": 0,
        "unfinished": 0,
    }

    # 预热，减少首次调用开销
    warmup = chess.Board()
    policy_order(warmup, policy, device)
    engines["neural"].evaluator(warmup)

    game_number = 0

    for opening_name, opening_moves in OPENINGS:
        for neural_is_white in [True, False]:
            game_number += 1
            board = chess.Board()

            for san in opening_moves:
                board.push_san(san)

            white = "neural" if neural_is_white else "material"
            black = "material" if neural_is_white else "neural"

            path = folder / f"game_{game_number}.pgn"
            log = []
            result = "*"
            termination = "unfinished"

            def save():
                game = chess.pgn.Game.from_board(board)
                game.headers["Event"] = "Local AI comparison"
                game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
                game.headers["White"] = white
                game.headers["Black"] = black
                game.headers["Result"] = result
                game.headers["Termination"] = termination
                game.headers["OpeningLabel"] = opening_name
                game.headers["SearchSeconds"] = str(SECONDS)
                game.headers["SearchDepth"] = str(DEPTH)
                game.headers["PolicyModel"] = "chess_model_balanced.pt"
                game.headers["DrawPolicy"] = "optional_claim"
                game.headers["CPUThreads"] = "4"
                game.headers["ValueModel"] = "chess_model_ranked_epoch4.pt"

                temporary = path.with_suffix(".tmp")
                temporary.write_text(str(game) + "\n", encoding="utf-8")
                temporary.replace(path)

                log_path = path.with_suffix(".json")
                log_tmp = path.with_suffix(".json.tmp")
                log_tmp.write_text(
                    json.dumps(log, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                log_tmp.replace(log_path)

            print(
                f"\n===== 第 {game_number}/{len(OPENINGS) * 2} 盘："
                f"{opening_name}，白方 {white}，黑方 {black} =====",
                flush=True,
            )
            save()

            try:
                while True:
                    # 这里只处理自动结束；申领和棋由引擎选择。
                    outcome = board.outcome()

                    if outcome is not None:
                        result = outcome.result()
                        termination = outcome.termination.name
                        break

                    if len(board.move_stack) >= MAX_PLIES:
                        # 达到运行上限不冒充规则和棋
                        termination = "move_limit_unfinished"
                        break

                    side = white if board.turn else black
                    preferred = policy_order(board, policy, device)

                    move, _, stats = engines[side].choose(
                        board, preferred
                    )

                    if move is None:
                        claim = draw_claim_details(board)
                        if claim is None:
                            raise ValueError("搜索申请了无效和棋")
                        result = "1/2-1/2"
                        termination = "claimed_" + claim["reason"]
                        log.append({
                            "position": board.fen(), "side": side,
                            "move": None, **stats, **claim,
                        })
                        print(f"{side} 申请和棋：{claim}", flush=True)
                        break

                    if move not in board.legal_moves:
                        raise ValueError("搜索返回非法走法")

                    san = board.san(move)
                    prefix = (
                        f"{board.fullmove_number}."
                        if board.turn else f"{board.fullmove_number}..."
                    )

                    log.append({
                        "position": board.fen(),
                        "side": side,
                        "move": move.uci(),
                        **stats,
                    })

                    board.push(move)
                    save()

                    print(
                        f"{prefix} {side}：{san} | "
                        f"深度 {stats['depth']} | "
                        f"{stats['seconds']:.2f} 秒",
                        flush=True,
                    )

            except KeyboardInterrupt:
                termination = "user_interrupted"
                save()
                print("\n已停止，当前棋谱已保存：", path)
                print("重新运行会创建一轮新比赛，不会续下本盘。")
                return

            save()

            if result == "*":
                summary["unfinished"] += 1
            elif result == "1/2-1/2":
                summary["draws"] += 1
            else:
                winner = white if result == "1-0" else black
                summary[f"{winner}_wins"] += 1

            (folder / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print(f"本盘结果：{result}，原因：{termination}", flush=True)

    print("\n===== 比赛结束 =====")
    print("价值版获胜：", summary["neural_wins"])
    print("子力版获胜：", summary["material_wins"])
    print("和棋：", summary["draws"])
    print("达到步数上限未结束：", summary["unfinished"])
    print("棋谱与搜索记录：", folder)


if __name__ == "__main__":
    main()