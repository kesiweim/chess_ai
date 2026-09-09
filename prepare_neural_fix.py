"""Recheck arena errors; creates paired CHILD-side value labels."""
import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path
import chess
import chess.engine

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "neural_fix_data"


def file_hash(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p, obj):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def load_candidates():
    return json.loads((ROOT / "arena_fix_candidates.json").read_text(encoding="utf-8"))


def score(engine, b, nodes):
    outcome = b.outcome()
    if outcome is not None:
        v = 0.0 if outcome.winner is None else (1.0 if outcome.winner == b.turn else -1.0)
        return {"value": v, "cp": None, "mate": None, "terminal": True}
    engine.configure({"Clear Hash": None})
    info = engine.analyse(b, chess.engine.Limit(nodes=nodes))
    s = info["score"].pov(b.turn)
    cp = s.score()
    return {"value": math.tanh(cp / 300) if cp is not None else
            (1.0 if s.score(mate_score=100000) > 0 else -1.0),
            "cp": cp, "mate": s.mate(), "terminal": False,
            "depth": info.get("depth"), "nodes": info.get("nodes")}


def recheck(engine, row, nodes):
    parent = chess.Board()
    for uci in row["history"]:
        parent.push_uci(uci)
    if parent.fen() != row["fen"]:
        raise ValueError("历史与局面不一致")
    if parent.can_claim_draw():
        return {"accepted": False, "reason": "claimable_parent"}
    children, historical = [], []
    for move in [row["best"]["move"], row["actual_move"]]:
        if move is None or chess.Move.from_uci(move) not in parent.legal_moves:
            return {"accepted": False, "reason": "not_a_move_pair"}
        child = parent.copy(stack=True)
        child.push_uci(move)
        if child.can_claim_draw() or child.is_repetition(2):
            return {"accepted": False, "reason": "history_sensitive_child"}
        historical.append(child)
        children.append(chess.Board(child.fen()))
    # Values are for the OPPONENT; a better original move has LOWER child value.
    low = [score(engine, b, 300000) for b in children]
    high = [score(engine, b, nodes) for b in children]
    hist = [score(engine, b, nodes) for b in historical]
    gaps = [p[1]["value"] - p[0]["value"] for p in [low, high, hist]]
    stable = all(g >= .10 for g in gaps)
    stable = stable and all(abs(low[i]["value"]-high[i]["value"]) <= .20
                            and abs(hist[i]["value"]-high[i]["value"]) <= .15
                            for i in range(2))
    base = {"accepted": stable, "game": row["game"], "fullmove": row["fullmove"],
            "parent_position": row["fen"], "gaps": gaps,
            "reason": "confirmed_pair" if stable else "unstable_or_small_value_gap"}
    base["children"] = [
        {"position": b.fen(), "move": move, **label}
        for b, move, label in zip(children, [row["best"]["move"], row["actual_move"]], high)]
    return base


def export(db, total):
    rows = [json.loads(x[0]) for x in db.execute("SELECT payload FROM records ORDER BY rowid")]
    accepted = [r for r in rows if r["accepted"]]
    write_json(OUT / "confirmed_pairs.json", accepted)
    write_json(OUT / "preparation_report.json", {
        "complete": len(rows) == total, "processed": len(rows), "total": total,
        "accepted_pairs": len(accepted), "rejected": len(rows)-len(accepted),
        "results": rows,
        "scope": "开发训练数据；不是独立验证集。未通过的候选不用于本轮训练。"
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path)
    ap.add_argument("--nodes", type=int, default=1000000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.nodes <= 300000 or args.limit < 0:
        ap.error("nodes 必须大于 300000，limit 必须非负")
    path = args.engine
    if path is None:
        paths = list(ROOT.glob("stockfish*/**/stockfish*.exe"))
        if len(paths) != 1:
            ap.error("请用 --engine 指定 Stockfish exe")
        path = paths[0]
    if not path.is_file():
        ap.error("找不到 Stockfish 文件")
    rows = load_candidates()
    OUT.mkdir(exist_ok=True)
    config = {"nodes": args.nodes, "engine": file_hash(path),
              "input": file_hash(ROOT / "arena_fix_candidates.json"), "version": 1}
    db = sqlite3.connect(OUT / "progress.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS metadata (config TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY,payload TEXT)")
    old = db.execute("SELECT config FROM metadata").fetchone()
    if old and json.loads(old[0]) != config:
        raise ValueError("配置变化：请先重命名并保留 neural_fix_data 目录，再重新运行")
    if not old:
        db.execute("INSERT INTO metadata VALUES (?)", (json.dumps(config),))
        db.commit()
    done = {r[0] for r in db.execute("SELECT id FROM records")}
    added = 0
    print(f"候选 {len(rows)}，已完成 {len(done)}", flush=True)
    try:
        with chess.engine.SimpleEngine.popen_uci(str(path.resolve())) as engine:
            engine.configure({"Threads": 1, "Hash": 128})
            for i, row in enumerate(rows):
                if i in done:
                    continue
                r = recheck(engine, row, args.nodes)
                db.execute("INSERT INTO records VALUES (?,?)", (i,json.dumps(r)))
                db.commit()
                added += 1
                print(f"{len(done)+added}/{len(rows)}：{r['reason']}", flush=True)
                if args.limit and added >= args.limit:
                    break
    except KeyboardInterrupt:
        print("已暂停，同一命令可继续。", flush=True)
    finally:
        export(db, len(rows))
        db.close()
    print("输出：", OUT, flush=True)


if __name__ == "__main__":
    main()
