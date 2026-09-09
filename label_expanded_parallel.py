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


# Reuse the original, hash-verified labelling semantics and exports.
import label_expanded_value as legacy
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading

filehash=legacy.filehash
write=legacy.write
label=legacy.label
export=legacy.export


def run_parallel(tasks, workers, path, nodes, consume):
    local=threading.local()
    engines=[]
    lock=threading.Lock()
    def work(row):
        if not hasattr(local,"engine"):
            engine=chess.engine.SimpleEngine.popen_uci(str(path.resolve()))
            with lock:
                engines.append(engine)
            local.engine=engine
            engine.configure({"Threads":1,"Hash":128})
        return row,label(local.engine,row[1],nodes)
    pool=ThreadPoolExecutor(max_workers=workers)
    pending=set()
    iterator=iter(tasks)
    def submit():
        try: row=next(iterator)
        except StopIteration:return False
        pending.add(pool.submit(work,row))
        return True
    try:
        for _ in range(2*workers):
            if not submit():break
        while pending:
            finished,_=wait(pending,timeout=.25,return_when=FIRST_COMPLETED)
            for future in finished:
                pending.remove(future)
                row,result=future.result()
                consume(row,result)
                submit()
    finally:
        for future in pending:future.cancel()
        # Finish the bounded in-flight searches; unconsumed results get retried on resume.
        pool.shutdown(wait=True,cancel_futures=True)
        for engine in engines:
            try:engine.quit()
            except Exception:engine.close()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--engine",type=Path)
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--nodes",type=int,default=300000)
    ap.add_argument("--limit",type=int,default=1000,help="本次新增上限；0为全部")
    args=ap.parse_args()
    if not 1<=args.workers<=32 or args.nodes<1 or args.limit<0:ap.error("workers须为1至32，nodes为正，limit非负")
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
            "script_sha256":filehash(Path(legacy.__file__).resolve()),
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
    write(OUT/"last_parallel_run.json",{"workers":args.workers,
          "driver_sha256":filehash(Path(__file__).resolve()),"semantics":"original label() reused",
          "nodes":args.nodes,"single_engine_threads":1})
    done={r[0] for r in db.execute("SELECT k FROM labels")}
    print(f"总局面 {len(rows):,}；已标注 {len(done):,}；每局面 {args.nodes:,} 节点",flush=True)
    tasks=[r for r in rows if r[0] not in done]
    if args.limit:
        tasks=tasks[:args.limit]
    added=0
    start=time.perf_counter()
    last_time=start
    last_added=0
    print(f"并行进程 {args.workers}，每进程1线程/128MB Hash；本次最多 {len(tasks):,} 条",flush=True)
    def consume(row,result):
        nonlocal added,last_time,last_added
        k,fen,split,seed,source_kind=row
        record={"sample_id":k,"position":fen,"split":split,"seed_id":seed,
                "game_id":seedmap[seed]["game_id"],"source":source_kind,
                "score_pov":"side_to_move","node_budget":args.nodes,**result}
        db.execute("INSERT INTO labels VALUES(?,?,?)",(k,split,json.dumps(record,ensure_ascii=False)))
        added+=1
        # Only the main thread writes SQLite. Keep normal durability settings.
        if added%20==0:
            db.commit()
        if added%100==0:
            now=time.perf_counter()
            recent=(added-last_added)/(now-last_time)
            print(f"累计 {len(done)+added:,}/{len(rows):,} | "
                  f"最近吞吐 {recent:.2f} 条/秒 | 累计折算 {(now-start)/added:.3f} 秒/条",flush=True)
            last_time,last_added=now,added
    try:
        run_parallel(tasks,args.workers,path,args.nodes,consume)
    except KeyboardInterrupt:
        print("已暂停；保存已接收结果。未接收的在途局面下次重跑。",flush=True)
    finally:
        db.commit()
        summary=export(db,len(rows),time.perf_counter()-start,added)
        write(OUT/"parallel_report.json",{
            **summary,"workers":args.workers,
            "positions_per_second":added/(time.perf_counter()-start) if added else None,
            "note":"seconds_per_new是并行总耗时/新增条数，不是单个Stockfish的搜索延迟。"})
        db.close()
    print(summary,flush=True)
    print("报告：",OUT/"label_report.json",flush=True)


if __name__=="__main__":main()
