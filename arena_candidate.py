import json
import argparse
import hashlib
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn
import torch

from model_cnn import ChessCNN
from model_residual_value import ResidualValueModel
from search_engine import SearchEngine, neural_evaluator, policy_order


OPENINGS = [
    ("意大利式", ["e4","e5","Nf3","Nc6","Bc4","Bc5"]),
    ("后翼弃兵发展", ["d4","d5","c4","e6","Nc3","Nf6"]),
    ("对称英国式", ["c4","c5","Nc3","Nc6","Nf3","Nf6"]),
]

SECONDS = 7.0
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    model_path = args.model
    if model_path is None:
        candidates = sorted((root/"value_balanced_runs").glob("*/value_balanced_epoch3.pt"))
        if not candidates:
            parser.error("找不到加权训练第3轮权重，请用 --model 指定")
        model_path = candidates[-1]
    model_path = model_path.resolve()
    if not model_path.is_file():
        parser.error("模型路径不存在")
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    policy_sha = hashlib.sha256((root/"chess_model_balanced.pt").read_bytes()).hexdigest()
    print("本次candidate：", model_path, flush=True)
    print("6局；双方CPU4线程，每步7秒、深度上限5。", flush=True)
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
        model_path,
        map_location=device,
        weights_only=True,
    ))
    value_model.eval()

    engines = {
        "candidate": SearchEngine(
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

    folder = root / "candidate_arena_games" / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    folder.mkdir(parents=True)

    summary = {
        "candidate_wins": 0,
        "material_wins": 0,
        "draws": 0,
        "unfinished": 0,
    }

    manifest = {
        "candidate": str(model_path), "candidate_sha256": model_sha,
        "policy_sha256": policy_sha, "seconds": SECONDS, "depth": DEPTH,
        "threads": 4, "max_plies": MAX_PLIES, "openings": OPENINGS,
        "search_sha256": hashlib.sha256((root/"search_engine.py").read_bytes()).hexdigest(),
        "scope": "不同于此前20局的起点；未证明这些开局从未出现在训练棋谱中。"
    }
    (folder/"config.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")

    def save_summary():
        summary["finished_games"] = summary["candidate_wins"] + summary["material_wins"] + summary["draws"] + summary["unfinished"]
        tmp = folder/"summary.tmp"
        tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
        tmp.replace(folder/"summary.json")

    def pack():
        archive = folder.with_suffix(".zip")
        temporary = archive.with_suffix(".zip.tmp")
        with ZipFile(temporary,"w",ZIP_DEFLATED) as z:
            for file in sorted(folder.iterdir()):
                if file.suffix in [".pgn",".json"]:
                    z.write(file,file.name)
        temporary.replace(archive)
        print("结果压缩包：",archive,flush=True)

    save_summary()
    # 预热，减少首次调用开销
    warmup = chess.Board()
    policy_order(warmup, policy, device)
    engines["candidate"].evaluator(warmup)

    game_number = 0

    for opening_name, opening_moves in OPENINGS:
        for neural_is_white in [True, False]:
            game_number += 1
            board = chess.Board()

            for san in opening_moves:
                board.push_san(san)

            white = "candidate" if neural_is_white else "material"
            black = "material" if neural_is_white else "candidate"

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
                game.headers["ValueModel"] = str(model_path)
                game.headers["ValueModelSHA256"] = model_sha
                game.headers["PolicyModelSHA256"] = policy_sha

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
                summary["interrupted_game"] = game_number
                save_summary()
                pack()
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

            save_summary()

            print(f"本盘结果：{result}，原因：{termination}", flush=True)

    pack()
    print("\n===== 比赛结束 =====")
    print("价值版获胜：", summary["candidate_wins"])
    print("子力版获胜：", summary["material_wins"])
    print("和棋：", summary["draws"])
    print("达到步数上限未结束：", summary["unfinished"])
    print("棋谱与搜索记录：", folder)


if __name__ == "__main__":
    main()