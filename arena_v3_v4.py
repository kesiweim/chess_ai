import json
import argparse
import subprocess
import sys
import os
import time
import signal
import hashlib
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn
import torch

from model_cnn import ChessCNN
from model_residual_value import ResidualValueModel
from search_engine import policy_order
from neural_search_v2 import NeuralSearchV2
from neural_search_v4 import NeuralSearchV4
from neural_inference_v3 import TracedEvaluator


OPENINGS = [
    ("意大利式", ["e4","e5","Nf3","Nc6","Bc4","Bc5"]),
    ("后翼弃兵发展", ["d4","d5","c4","e6","Nc3","Nf6"]),
    ("对称英国式", ["c4","c5","Nc3","Nc6","Nf3","Nf6"]),
]

OPENINGS += [
    ("西班牙式", ["e4","e5","Nf3","Nc6","Bb5","a6"]),
    ("西西里式", ["e4","c5","Nf3","d6","d4","cxd4"]),
    ("法兰西式", ["e4","e6","d4","d5","Nc3","Nf6"]),
    ("卡罗康式", ["e4","c6","d4","d5","Nc3","dxe4"]),
    ("斯拉夫式", ["d4","d5","c4","c6","Nf3","Nf6"]),
    ("王印式", ["d4","Nf6","c4","g6","Nc3","Bg7"]),
    ("苏格兰式", ["e4","e5","Nf3","Nc6","d4","exd4"]),
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


def run_pair(args):
    root = Path(__file__).resolve().parent
    model_path = args.model
    model_path=model_path.resolve()
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    torch.set_grad_enabled(False)
    policy_sha = hashlib.sha256((root/"chess_model_balanced.pt").read_bytes()).hexdigest()
    print("本次v4：", model_path, flush=True)
    print("交换颜色两局；双方CPU4线程，每步5秒、深度上限5。", flush=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
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

    evaluator=TracedEvaluator(value_model)
    probes=[chess.Board()]
    for _,moves in OPENINGS:
        board=chess.Board()
        for san in moves:board.push_san(san)
        probes.append(board)
    evaluator.validate(probes)
    engines={
        "v4":NeuralSearchV4(evaluator,seconds=SECONDS,max_depth=DEPTH,claim_draw=True),
        "v3":NeuralSearchV2(evaluator,seconds=SECONDS,max_depth=DEPTH,claim_draw=True),
    }

    folder = args.output / f"pair_{args.pair+1:02d}"
    folder.mkdir(parents=True)

    summary = {
        "v4_wins": 0,
        "v3_wins": 0,
        "draws": 0,
        "unfinished": 0,
    }

    manifest = {
        "v4": str(model_path), "v4_sha256": model_sha,
        "policy_sha256": policy_sha, "seconds": SECONDS, "depth": DEPTH,
        "threads": 4, "max_plies": MAX_PLIES, "openings": OPENINGS,
        "search_sha256": hashlib.sha256((root/"search_engine.py").read_bytes()).hexdigest(),
        "scope": "v3/v4同权重、同策略排序、同线程与时间；未证明这些开局未用于训练。"
    }
    (folder/"config.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")

    def save_summary():
        summary["finished_games"] = summary["v4_wins"] + summary["v3_wins"] + summary["draws"] + summary["unfinished"]
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
    engines["v4"].evaluator(warmup)

    game_number = args.pair * 2

    for opening_name, opening_moves in [OPENINGS[args.pair]]:
        for neural_is_white in [True, False]:
            game_number += 1
            board = chess.Board()

            for san in opening_moves:
                board.push_san(san)

            white = "v4" if neural_is_white else "v3"
            black = "v3" if neural_is_white else "v4"

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
    print("v4获胜：", summary["v4_wins"])
    print("v3获胜：", summary["v3_wins"])
    print("和棋：", summary["draws"])
    print("达到步数上限未结束：", summary["unfinished"])
    print("棋谱与搜索记录：", folder)


def main():
    root=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(description="20局：全量第2轮 vs v3；5秒，最高深度5")
    ap.add_argument("--model",type=Path)
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--pair",type=int,default=None,help=argparse.SUPPRESS)
    ap.add_argument("--output",type=Path,help=argparse.SUPPRESS)
    args=ap.parse_args()
    if args.pair is not None:
        run_pair(args)
        return
    if not 1<=args.workers<=10:ap.error("workers须为1至10")
    if args.model is None:
        v4s=sorted((root/"value_full_runs").glob("*/value_full_epoch2.pt"))
        if not v4s:ap.error("找不到全量第2轮模型；请用 --model 指定")
        args.model=v4s[-1]
    args.model=args.model.resolve()
    if not args.model.is_file():ap.error("模型文件不存在")
    for name,moves in OPENINGS:
        board=chess.Board()
        for san in moves:board.push_san(san)
    # Fail early before starting ten model-loading jobs.
    for name in ["chess_model_balanced.pt","search_engine.py"]:
        if not (root/name).is_file():ap.error("缺少文件："+name)
    output=root/"arena_v3_v4_games"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    config={"v4":str(args.model),"v4_sha256":hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "policy_sha256":hashlib.sha256((root/"chess_model_balanced.pt").read_bytes()).hexdigest(),
            "search_sha256":hashlib.sha256((root/"search_engine.py").read_bytes()).hexdigest(),
            "workers":args.workers,"threads_per_worker":4,"games":20,"seconds":SECONDS,"depth":DEPTH,
            "max_plies":MAX_PLIES,"openings":OPENINGS,
            "scope":"本机并发墙钟限时比赛；资源竞争可能改变完成深度；不是Elo认证。"}
    config['implementation_sha256']={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
        ['neural_fast.py','neural_search_v2.py','neural_search_v4.py','neural_inference_v3.py',Path(__file__).name]}
    (output/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    print("模型：",args.model,flush=True)
    print(f"20局；同时最多{args.workers}局；每进程4线程；5秒/步，最高深度5",flush=True)
    print("逐步日志和棋谱：",output,flush=True)
    active={};next_pair=0;failed=[];completed=[];interrupted=False
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1")
    try:
        while next_pair<10 or active:
            while next_pair<10 and len(active)<args.workers:
                i=next_pair;next_pair+=1
                log=(output/f"pair_{i+1:02d}.log").open("w",encoding="utf-8")
                try:
                    proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--pair",str(i),
                        "--output",str(output),"--model",str(args.model)],cwd=root,env=env,
                        stdout=log,stderr=subprocess.STDOUT)
                except BaseException:
                    log.close();raise
                active[i]=(proc,log)
                print(f"启动第{i*2+1}、{i*2+2}局（顺序交换颜色）",flush=True)
            for i,(proc,log) in list(active.items()):
                code=proc.poll()
                if code is not None:
                    log.close();del active[i]
                    (completed if code==0 else failed).append(i+1)
                    print(f"开局组{i+1}结束，退出码{code}；已结束{len(completed)+len(failed)}/10组",flush=True)
            time.sleep(.25)
    except KeyboardInterrupt:
        interrupted=True
        print("停止比赛，保留已落盘棋谱；重新运行会新建比赛。",flush=True)
    finally:
        for proc,log in active.values():
            if proc.poll() is None:proc.terminate()
        for proc,log in active.values():
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()
        summary={"v4_wins":0,"v3_wins":0,"draws":0,"unfinished":0,
                 "completed_pairs":completed,"failed_pairs":failed,"interrupted":interrupted}
        for p in output.glob("pair_*/summary.json"):
            part=json.loads(p.read_text(encoding="utf-8"))
            for k in ["v4_wins","v3_wins","draws","unfinished"]:summary[k]+=part[k]
        summary["accounted_games"]=sum(summary[k] for k in ["v4_wins","v3_wins","draws","unfinished"])
        summary["unaccounted_games"]=20-summary["accounted_games"]
        (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
        with ZipFile(output.with_suffix(".zip"),"w",ZIP_DEFLATED) as z:
            for p in sorted(output.rglob("*")):
                if p.is_file() and p.suffix in [".json",".pgn",".log"]:z.write(p,p.relative_to(output))
        print(json.dumps(summary,ensure_ascii=False),flush=True)
        print("发我这个压缩包：",output.with_suffix(".zip"),flush=True)
    if failed:raise SystemExit("有开局组出错，请发压缩包排查，不计作输棋")


if __name__ == "__main__":
    main()
