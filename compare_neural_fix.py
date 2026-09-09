import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
import chess
import chess.engine

ROOT = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,obj):
    t=p.with_suffix(".tmp")
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
    t.replace(p)


def board_from(row):
    b=chess.Board()
    for m in row["history"]:
        b.push_uci(m)
    if b.fen()!=row["fen"]:
        raise ValueError("局面与历史不一致")
    return b


def judge(engine,b,move,nodes):
    if move is None:
        if not b.can_claim_draw():
            raise ValueError("非法申领")
        return {"cp":0,"mate":None}
    child=b.copy(stack=True)
    child.push(move)
    outcome=child.outcome()
    if outcome is not None:
        if outcome.winner is None:
            return {"cp":0,"mate":None}
        return {"cp":None,"mate":1 if outcome.winner==b.turn else -1}
    engine.configure({"Clear Hash":None})
    info=engine.analyse(b,chess.engine.Limit(nodes=nodes),root_moves=[move])
    score=info["score"].pov(b.turn)
    return {"cp":score.score(),"mate":score.mate(),"depth":info.get("depth"),
            "nodes":info.get("nodes")}


def classify(old,new,same):
    if same:
        return "same_move"
    # Mate distances are not used to declare one mate-preserving move stronger.
    def band(s):
        if s["mate"] is not None:
            return 1 if s["mate"]>0 else -1
        return 0
    a,c=band(old),band(new)
    if a!=c:
        return "new_better" if c>a else "old_better"
    if a!=0:
        return "both_mate_scores"
    delta=new["cp"]-old["cp"]
    return "new_better" if delta>30 else "old_better" if delta < -30 else "close"


def export(db,out,total):
    rows=[json.loads(r[0]) for r in db.execute("SELECT payload FROM results ORDER BY id")]
    counts={k:sum(r["comparison"]==k for r in rows)
            for k in ["same_move","new_better","old_better","close","both_mate_scores"]}
    summary={"completed":len(rows),"total":total,"complete":len(rows)==total,**counts}
    for side in ["old","new"]:
        completed=[r[side]["seconds"] for r in rows]
        summary[side+"_mean_seconds"]=sum(completed)/len(completed) if completed else None
        summary[side+"_mate_shortcuts"]=sum(r[side]["mate_in_one"] for r in rows)
    summary["scope"]="训练局面的搜索回归检查，不是独立棋力验证；Stockfish为有限节点估计。"
    write(out/"comparison_details.json",rows)
    write(out/"comparison_summary.json",summary)
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate",type=Path)
    ap.add_argument("--engine",type=Path)
    ap.add_argument("--limit",type=int,default=0,help="本次新增上限，0为全部")
    ap.add_argument("--depth",type=int,default=3)
    ap.add_argument("--nodes",type=int,default=1000000)
    ap.add_argument("--check",action="store_true")
    args=ap.parse_args()
    if args.depth<1 or args.nodes<1 or args.limit<0:
        ap.error("参数范围错误")
    rows=json.loads((ROOT/"regression_positions.json").read_text(encoding="utf-8"))
    for r in rows:
        board_from(r)
    if args.check:
        print(f"核验通过：{len(rows)} 个完整历史局面")
        return
    out=ROOT/"neural_fix_comparison"
    oldpath=ROOT/"chess_model_ranked_epoch4.pt"
    candidate=args.candidate
    # Once started, pin the same candidate on resume.
    saved=out/"config.json"
    previous=json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else None
    if candidate is None and previous:
        candidate=Path(previous["candidate"])
    if candidate is None:
        found=sorted((ROOT/"neural_fix_runs").glob("*/neural_fix_epoch2.pt"))
        if not found:
            ap.error("找不到第2轮候选；请用 --candidate 指定")
        candidate=found[-1]
    enginepath=args.engine
    if enginepath is None:
        found=list(ROOT.glob("stockfish*/**/stockfish*.exe"))
        if len(found)!=1:
            ap.error("请用 --engine 指定 Stockfish")
        enginepath=found[0]
    files=[oldpath,candidate,ROOT/"chess_model_balanced.pt",enginepath,
           ROOT/"search_engine.py",ROOT/"model_residual_value.py",ROOT/"model_cnn.py",
           ROOT/"board_encoder.py",ROOT/"move_encoder.py",ROOT/"regression_positions.json",
           Path(__file__).resolve()]
    for p in files:
        if not p.is_file():
            ap.error(f"找不到：{p}")
    config={"candidate":str(candidate.resolve()),"depth":args.depth,"nodes":args.nodes,
            "hashes":[sha(p) for p in files],"threads":4,"device":"cpu"}
    if previous and config!=previous:
        raise ValueError("模型、代码或配置改变。请先重命名保留 neural_fix_comparison 目录，再重新运行")
    print("原模型：",oldpath,flush=True)
    print("候选：",candidate,flush=True)
    print(f"完整深度 {args.depth}，无搜索时间截断。Ctrl+C 可暂停。",flush=True)
    import torch
    from model_cnn import ChessCNN
    from model_residual_value import ResidualValueModel
    from search_engine import SearchEngine,neural_evaluator,policy_order
    torch.set_num_threads(4)
    device=torch.device("cpu")
    policy=ChessCNN().to(device)
    policy.load_state_dict(torch.load(ROOT/"chess_model_balanced.pt",map_location=device,weights_only=True))
    policy.eval()
    engines={}
    for side,path in [("old",oldpath),("new",candidate)]:
        model=ResidualValueModel().to(device)
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
        model.eval()
        engines[side]=SearchEngine(neural_evaluator(model,device),seconds=float("inf"),
                                   max_depth=args.depth,claim_draw=True)
    warm=chess.Board()
    policy_order(warm,policy,device)
    for e in engines.values():
        e.evaluator(warm)
    out.mkdir(exist_ok=True)
    if not previous:
        write(saved,config)
    db=sqlite3.connect(out/"progress.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS results(id INTEGER PRIMARY KEY,payload TEXT)")
    db.commit()
    done={r[0] for r in db.execute("SELECT id FROM results")}
    added=0
    try:
        with chess.engine.SimpleEngine.popen_uci(str(enginepath.resolve())) as sf:
            sf.configure({"Threads":1,"Hash":128})
            for i,row in enumerate(rows):
                if i in done:
                    continue
                b=board_from(row)
                preferred=policy_order(b,policy,device)
                choices={};record=dict(row)
                for side in (["old","new"] if i%2==0 else ["new","old"]):
                    copy=b.copy(stack=True)
                    before=(copy.fen(),list(copy.move_stack))
                    start=time.perf_counter()
                    move,mate,stats=engines[side].choose(copy,preferred)
                    elapsed=time.perf_counter()-start
                    if before!=(copy.fen(),list(copy.move_stack)):
                        raise ValueError("搜索没有恢复棋盘")
                    if not mate and stats["depth"]!=args.depth:
                        raise ValueError("未完成指定深度，停止而不混入浅层结果")
                    if move is None:
                        if not b.can_claim_draw():
                            raise ValueError("无效申领")
                    elif move not in b.legal_moves:
                        raise ValueError("非法走法")
                    choices[side]=move
                    record[side]={**stats,"seconds":elapsed,"mate_in_one":mate,
                                  "move":move.uci() if move else None,
                                  "san":b.san(move) if move else "claim_draw"}
                evaluations={}
                for side in (["old","new"] if i%2==0 else ["new","old"]):
                    m=choices[side]
                    if m not in evaluations:
                        evaluations[m]=judge(sf,b,m,args.nodes)
                    record[side]["stockfish"]=evaluations[m]
                record["comparison"]=classify(record["old"]["stockfish"],
                                               record["new"]["stockfish"],choices["old"]==choices["new"])
                db.execute("INSERT INTO results VALUES (?,?)",(i,json.dumps(record)))
                db.commit()
                added+=1
                print(f"{len(done)+added}/{len(rows)}：旧 {record['old']['san']} → 新 {record['new']['san']} | "
                      f"{record['comparison']}",flush=True)
                if added%10==0:
                    export(db,out,len(rows))
                if args.limit and added>=args.limit:
                    break
    except KeyboardInterrupt:
        print("已暂停；已完成局面保留，未完成的一个局面下次重跑。",flush=True)
    finally:
        summary=export(db,out,len(rows))
        db.close()
    print(summary,flush=True)
    print("报告目录：",out,flush=True)


if __name__=="__main__":
    main()
