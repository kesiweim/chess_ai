"""Read completed collection, label fresh FENs, resume durably in a separate DB."""
import argparse
import hashlib
import json
import math
import sqlite3
import time
from pathlib import Path
import chess
import chess.engine

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"value_expansion_data"
OUT=ROOT/"value_expansion_labels"


def filehash(path):
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def write(path,obj):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(path)


def label(engine,fen,nodes):
    b=chess.Board(fen)
    if not b.is_valid():raise ValueError("采集到无效局面："+fen)
    outcome=b.outcome()
    if outcome:
        value=0. if outcome.winner is None else (1. if outcome.winner==b.turn else -1.)
        return {"value":value,"cp":None,"mate":None,"terminal":True,
                "termination":outcome.termination.name,"depth":0,"nodes":0}
    engine.configure({"Clear Hash":None})
    info=engine.analyse(b,chess.engine.Limit(nodes=nodes))
    score=info["score"].pov(b.turn)
    cp=score.score()
    value=math.tanh(cp/300) if cp is not None else (1. if score.score(mate_score=100000)>0 else -1.)
    # Static FEN interface: no reconstructed repetition history.
    return {"value":value,"cp":cp,"mate":score.mate(),"terminal":False,
            "depth":info.get("depth"),"nodes":info.get("nodes"),
            "best_move":info["pv"][0].uci() if info.get("pv") else None}


def export(db,total,elapsed,added):
    counts=dict(db.execute("SELECT split,count(*) FROM labels GROUP BY split"))
    completed=sum(counts.values())
    stats={"completed":completed,"total":total,"complete":completed==total,
           "split_counts":counts,"new_this_run":added,"seconds_this_run":elapsed,
           "seconds_per_new":elapsed/added if added else None}
    details={"mate_labels":0,"terminal_labels":0,"value_sum":0.}
    for (payload,) in db.execute("SELECT payload FROM labels"):
        r=json.loads(payload)
        details["mate_labels"]+=int(r["mate"] is not None)
        details["terminal_labels"]+=int(r["terminal"])
        details["value_sum"]+=r["value"]
    stats.update(details)
    stats["mean_value"]=stats.pop("value_sum")/completed if completed else None
    stats["remaining_hours_estimate"]=(total-completed)*elapsed/added/3600 if added else None
    # Training exports become available only after the full fixed corpus is labelled.
    if completed==total:
        for split in ["train","val"]:
            path=OUT/f"expanded_value_{split}.jsonl"
            tmp=path.with_suffix(".tmp")
            with tmp.open("w",encoding="utf-8") as f:
                for (payload,) in db.execute("SELECT payload FROM labels WHERE split=? ORDER BY k",(split,)):
                    f.write(payload+"\n")
            tmp.replace(path)
    write(OUT/"label_report.json",stats)
    return stats


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--engine",type=Path)
    ap.add_argument("--nodes",type=int,default=300000)
    ap.add_argument("--limit",type=int,default=1000,help="本次新增上限；0为全部")
    args=ap.parse_args()
    if args.nodes<1 or args.limit<0:ap.error("nodes必须为正，limit不能为负")
    source=DATA/"collection.sqlite"
    seedpath=DATA/"seeds.json"
    if not source.is_file() or not seedpath.is_file():ap.error("找不到采集数据库或seeds.json")
    seeds=json.loads(seedpath.read_text(encoding="utf-8"))
    seedmap={r["id"]:r for r in seeds}
    src=sqlite3.connect(source.resolve().as_uri()+"?mode=ro",uri=True)
    if src.execute("SELECT count(*) FROM done").fetchone()[0]!=len(seeds):
        raise ValueError("搜索起点尚未采集完，请先完成采集")
    rows=src.execute("SELECT k,fen,split,seed,source FROM positions WHERE split IN ('train','val') ORDER BY k").fetchall()
    source_config=src.execute("SELECT value FROM config").fetchone()[0]
    src.close()
    fingerprint=hashlib.sha256()
    for row in rows:
        fingerprint.update(json.dumps(row,ensure_ascii=False).encode())
        seed=seedmap[row[3]]
        if seed["split"]!=row[2]:raise ValueError("局面与来源组别不一致")
    path=args.engine
    if path is None:
        found=list(ROOT.glob("stockfish*/**/stockfish*.exe"))
        if len(found)!=1:ap.error("请使用 --engine 指定 Stockfish exe")
        path=found[0]
    if not path.is_file():ap.error("找不到Stockfish："+str(path))
    config={"version":1,"nodes":args.nodes,"threads":1,"hash_mb":128,
            "engine_sha256":filehash(path),"corpus_sha256":fingerprint.hexdigest(),
            "seeds_sha256":filehash(seedpath),"collection_config":source_config,
            "script_sha256":filehash(Path(__file__).resolve()),
            "value":"tanh(cp/300); mate +/-1; side_to_move; fresh_fen"}
    OUT.mkdir(exist_ok=True)
    db=sqlite3.connect(OUT/"progress.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS config(payload TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS labels(k TEXT PRIMARY KEY,split TEXT,payload TEXT)")
    old=db.execute("SELECT payload FROM config").fetchone()
    if old and json.loads(old[0])!=config:
        raise ValueError("输入、脚本、引擎或参数已改变；请保留并重命名 value_expansion_labels 后新建标注")
    if not old:
        db.execute("INSERT INTO config VALUES(?)",(json.dumps(config),));db.commit()
        write(OUT/"config.json",config)
    done={r[0] for r in db.execute("SELECT k FROM labels")}
    print(f"总局面 {len(rows):,}；已标注 {len(done):,}；每局面 {args.nodes:,} 节点",flush=True)
    added=0;start=time.perf_counter()
    try:
        with chess.engine.SimpleEngine.popen_uci(str(path.resolve())) as engine:
            engine.configure({"Threads":1,"Hash":128})
            print("引擎：",engine.id,flush=True)
            for k,fen,split,seed,source_kind in rows:
                if k in done:continue
                result=label(engine,fen,args.nodes)
                record={"sample_id":k,"position":fen,"split":split,"seed_id":seed,
                        "game_id":seedmap[seed]["game_id"],"source":source_kind,
                        "score_pov":"side_to_move","node_budget":args.nodes,**result}
                db.execute("INSERT INTO labels VALUES(?,?,?)",(k,split,json.dumps(record,ensure_ascii=False)))
                db.commit()
                added+=1
                if added%100==0:
                    elapsed=time.perf_counter()-start
                    print(f"累计 {len(done)+added:,}/{len(rows):,} | 平均 {elapsed/added:.3f} 秒/条",flush=True)
                if args.limit and added>=args.limit:break
    except KeyboardInterrupt:
        db.rollback()
        print("已暂停；已完成记录保留，原命令可继续。",flush=True)
    finally:
        summary=export(db,len(rows),time.perf_counter()-start,added)
        db.close()
    print(summary,flush=True)
    print("报告：",OUT/"label_report.json",flush=True)


if __name__=="__main__":main()
