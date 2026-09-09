"""Review the frozen arena20 games; never edits weights or search code."""
import argparse
import hashlib
import json
import math
import shutil
import sqlite3
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tasks():
    for n in range(1, 21):
        p = ROOT / "arena20" / f"game_{n}.pgn"
        with p.open(encoding="utf-8-sig") as f:
            game = chess.pgn.read_game(f)
        if game is None or game.errors:
            raise ValueError(f"坏棋谱：{p}")
        log = json.loads(p.with_suffix(".json").read_text(encoding="utf-8-sig"))
        b = game.board()
        for m in list(game.mainline_moves())[:4]:
            b.push(m)
        for idx, row in enumerate(log):
            if row["position"] != b.fen():
                raise ValueError(f"第{n}局第{idx}条记录不匹配")
            if row["side"] == "neural":
                yield f"{n}:{idx}", n, row, b.copy(stack=True)
            if row["move"] is not None:
                move = chess.Move.from_uci(row["move"])
                if move not in b.legal_moves:
                    raise ValueError("非法走法")
                b.push(move)


def score_record(score):
    cp, mate = score.score(), score.mate()
    return {"cp": cp, "mate": mate,
            "value": math.tanh(cp / 300) if cp is not None else
                     (1.0 if score.score(mate_score=100000) > 0 else -1.0)}


def analyze(engine, board, nodes, moves=None):
    # Fresh hash for each alternative: avoid unequal warm-cache advantage.
    engine.configure({"Clear Hash": None})
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes),
                          root_moves=moves)
    result = score_record(info["score"].pov(board.turn))
    result.update(depth=info.get("depth"), nodes=info.get("nodes"),
                  move=info["pv"][0].uci() if info.get("pv") else None)
    return result


def review(engine, board, row, game, nodes):
    best = analyze(engine, board, nodes)
    if best["move"] is None:
        raise ValueError("引擎未返回走法")
    actual = row["move"]
    if actual is None:
        if not board.can_claim_draw():
            raise ValueError("无效申领")
        played = {"cp": 0, "mate": None, "value": 0, "move": None}
    elif actual == best["move"]:
        played = dict(best)
    else:
        played = analyze(engine, board, nodes, [chess.Move.from_uci(actual)])
    # Optional claim is a real zero-valued alternative for the current player.
    best_value = max(best["value"], 0) if board.can_claim_draw() else best["value"]
    cp_loss = None
    if best["cp"] is not None and played["cp"] is not None:
        cp_loss = (max(0, best["cp"]) if board.can_claim_draw() else best["cp"]) - played["cp"]
    material = {chess.PAWN:100,chess.KNIGHT:320,chess.BISHOP:330,chess.ROOK:500,chess.QUEEN:900}
    material_cp = sum(v * (len(board.pieces(p,board.turn))-len(board.pieces(p,not board.turn)))
                      for p,v in material.items())
    return {
        "game":game, "fullmove":board.fullmove_number, "fen":board.fen(),
        "history":[m.uci() for m in board.move_stack],
        "actual_move":actual, "actual_san":board.san(chess.Move.from_uci(actual)) if actual else "claim_draw",
        "best":best, "best_san":board.san(chess.Move.from_uci(best["move"])),
        "played":played, "cp_loss":cp_loss,
        "value_loss":best_value-played["value"],
        "can_claim_draw":board.can_claim_draw(),
        "pieces":len(board.piece_map()), "material_cp":material_cp,
        "search_depth":row["depth"], "search_score":row.get("score"),
        "note":"有限节点估计；负损失表示两次搜索不一致，不能直接作为纠正标签。",
    }


def export(db, out, total):
    rows=[json.loads(r[0]) for r in db.execute("SELECT payload FROM reviews ORDER BY rowid")]
    dest=out/"neural_review.json"
    tmp=dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(dest)
    suspects=[r for r in rows if (r["cp_loss"] is not None and r["cp_loss"]>=100)
              or r["value_loss"]>=.2
              or (r["best"]["mate"] is not None and r["best"]["mate"]>0
                  and (r["played"]["mate"] is None or r["played"]["mate"]<=0))]
    summary={"completed":len(rows),"total":total,"complete":len(rows)==total,
             "suspected_errors":len(suspects),
             "low_piece_records_le_7":sum(r["pieces"]<=7 for r in rows),
             "top_suspects":sorted(suspects,key=lambda r:r["value_loss"],reverse=True)[:30],
             "usage":"开发复盘数据，不是独立测试集。需复核后再生成训练样本。"}
    tmp=out/"review_summary.tmp"
    tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(out/"review_summary.json")


def backup():
    # Snapshot code AND weights; material uses the policy model for ordering.
    folder=ROOT/"frozen_before_neural_review"
    names=["arena.py","search_engine.py","model_cnn.py","board_encoder.py",
           "move_encoder.py","model_residual_value.py","chess_model_balanced.pt",
           "chess_model_ranked_epoch4.pt"]
    folder.mkdir(exist_ok=True)
    for name in names:
        source=ROOT/name
        if source.exists() and not (folder/name).exists():
            shutil.copy2(source,folder/name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--engine",type=Path)
    ap.add_argument("--nodes",type=int,default=300000)
    ap.add_argument("--limit",type=int,default=0,help="本次最多新增几条；0 表示全部")
    ap.add_argument("--check",action="store_true")
    args=ap.parse_args()
    if args.nodes<1 or args.limit<0:
        ap.error("nodes 必须为正，limit 不能为负")
    items=list(tasks())
    print(f"核验通过：20 局，neural 决策 {len(items)} 条",flush=True)
    if args.check:
        return
    engine_path=args.engine
    if engine_path is None:
        found=list(ROOT.glob("stockfish*/**/stockfish*.exe"))
        if len(found)!=1:
            ap.error("请通过 --engine 指定 Stockfish exe 路径")
        engine_path=found[0]
    engine_path=engine_path.resolve()
    if not engine_path.is_file():
        ap.error(f"找不到引擎：{engine_path}")
    backup()
    out=ROOT/"neural_review_output"
    out.mkdir(exist_ok=True)
    config={"nodes":args.nodes,"engine_sha256":digest(engine_path),
            "data":[digest(ROOT/"arena20"/f"game_{i}.{ext}")
                    for i in range(1,21) for ext in ["pgn","json"]],
            "schema":1}
    db=sqlite3.connect(out/"progress.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY,payload TEXT)")
    old=db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
    if old and json.loads(old[0])!=config:
        raise ValueError("本次配置或文件与旧进度不同。请保留并重命名 neural_review_output 后重跑。")
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('config',?)",(json.dumps(config),))
    db.commit()
    done={r[0] for r in db.execute("SELECT id FROM reviews")}
    added=0; start=time.perf_counter()
    try:
        with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
            engine.configure({"Threads":1,"Hash":128})
            print("引擎：",engine.id,flush=True)
            for key,n,row,board in items:
                if key in done:
                    continue
                result=review(engine,board,row,n,args.nodes)
                db.execute("INSERT INTO reviews VALUES (?,?)",
                           (key,json.dumps(result,ensure_ascii=False)))
                db.commit()
                added+=1
                if added%10==0:
                    print(f"累计 {len(done)+added}/{len(items)}；本次 {added}；"
                          f"平均 {(time.perf_counter()-start)/added:.2f} 秒/条",flush=True)
                if args.limit and added>=args.limit:
                    break
    except KeyboardInterrupt:
        print("\n已暂停，完成记录已保存。相同命令可继续。",flush=True)
    finally:
        export(db,out,len(items))
        db.close()
    print(f"本次新增 {added} 条，输出：{out}",flush=True)


if __name__=="__main__":
    main()
