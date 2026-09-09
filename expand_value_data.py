"""Stage 1: fresh PGNs -> game-split roots -> sampled neural evaluation positions."""
import argparse
import hashlib
import io
import json
import random
import sqlite3
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
import chess
import chess.pgn

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"value_expansion_data"


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def filehash(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def key(fen):
    return h(" ".join(fen.split()[:4]))


def write(p,v):
    tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(p)


def download():
    folder=DATA/"pgn"
    folder.mkdir(parents=True,exist_ok=True)
    for issue in range(1650,1660):
        path=folder/f"twic{issue}.pgn"
        if path.exists():
            print("已有：",path.name,flush=True)
            continue
        url=f"https://theweekinchess.com/zips/twic{issue}g.zip"
        print("下载：",url,flush=True)
        request=urllib.request.Request(url,headers={"User-Agent":"ChessLearningPersonal/1.0"})
        with urllib.request.urlopen(request,timeout=90) as response:
            raw=response.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names=[n for n in z.namelist() if n.lower().endswith(".pgn")]
            if len(names)!=1:
                raise ValueError("压缩包PGN数量异常")
            content=z.read(names[0])
        if b'[Event ' not in content[:10000]:
            raise ValueError("下载内容不像PGN")
        tmp=path.with_suffix(".tmp")
        tmp.write_bytes(content);tmp.replace(path)
    print("10期下载完成",flush=True)


def games(path):
    with path.open(encoding="utf-8-sig",errors="replace") as f:
        while True:
            g=chess.pgn.read_game(f)
            if g is None:break
            yield g


def gameid(g):
    return h(g.board().fen()+" "+" ".join(m.uci() for m in g.mainline_moves()))


def prepare():
    if (DATA/"seeds.json").exists():
        print("seeds.json 已存在，保持原分组。请继续 collect。");return
    paths=[DATA/"pgn"/f"twic{i}.pgn" for i in range(1650,1660)]
    if not all(p.exists() for p in paths):
        raise ValueError("请先完成 download")
    old=ROOT/"twic1660.pgn"
    if not old.exists():
        raise ValueError("需要原 twic1660.pgn 用于排除重复对局，请放在项目目录")
    seen={gameid(g) for g in games(old) if not g.errors}
    count=Counter()
    buckets={"train":[],"val":[]}
    # Deterministic reservoir: each eligible game contributes at most three roots.
    rng=random.Random(20260906);seen_seeds=Counter()
    caps={"train":8000,"val":800}
    for path in paths:
        for g in games(path):
            count["read"]+=1
            if g.errors or g.headers.get("Variant","Standard")!="Standard" or g.board().fen()!=chess.STARTING_FEN:
                count["invalid_or_variant"]+=1;continue
            try:
                strong=min(int(g.headers["WhiteElo"]),int(g.headers["BlackElo"]))>=2200
            except (KeyError,ValueError):
                strong=False
            if not strong:
                count["rating_filtered"]+=1;continue
            gid=gameid(g)
            if gid in seen:
                count["duplicate_or_old_game"]+=1;continue
            seen.add(gid)
            split="val" if int(gid[:8],16)%10==0 else "train"
            count["games_"+split]+=1
            b=g.board();history=[];phases=[[],[],[]]
            for move in g.mainline_moves():
                b.push(move);history.append(move.uci())
                if len(history)<12 or b.is_game_over():continue
                pieces=len(b.piece_map())
                phase=0 if len(history)<40 else 2 if pieces<=12 else 1
                phases[phase].append({"fen":b.fen(),"history":list(history),"game_id":gid,
                                      "split":split,"source_file":path.name,"phase":phase})
            grng=random.Random(gid)
            for phase in phases:
                if not phase:continue
                row=grng.choice(phase)
                seen_seeds[split]+=1
                bucket=buckets[split]
                if len(bucket)<caps[split]:bucket.append(row)
                else:
                    j=rng.randrange(seen_seeds[split])
                    if j<caps[split]:bucket[j]=row
        print(path.name,dict(count),flush=True)
    rows=buckets["train"]+buckets["val"]
    if not buckets["train"] or not buckets["val"]:raise ValueError("分组为空")
    random.Random(42).shuffle(rows)
    for i,r in enumerate(rows):r["id"]=i
    write(DATA/"seeds.json",rows)
    write(DATA/"seed_report.json",{"counts":dict(count),"train_roots":len(buckets["train"]),
                                  "val_roots":len(buckets["val"]),
                                  "note":"先整盘分组；衍生局面跨组重复在采集阶段删除"})
    print("起始局面：",len(rows),flush=True)


def put(db,fen,split,seed,source):
    k=key(fen)
    old=db.execute("SELECT split FROM positions WHERE k=?",(k,)).fetchone()
    if old:
        if old[0]!=split and old[0]!="blocked":
            db.execute("UPDATE positions SET split='blocked' WHERE k=?",(k,))
        return
    db.execute("INSERT INTO positions VALUES(?,?,?,?,?)",(k,fen,split,seed,source))


def report(db,seeds):
    counts=dict(db.execute("SELECT split,count(*) FROM positions GROUP BY split"))
    done=db.execute("SELECT count(*) FROM done").fetchone()[0]
    phases=Counter()
    for (fen,) in db.execute("SELECT fen FROM positions WHERE split!='blocked'"):
        b=chess.Board(fen)
        phases["pieces_le_7" if len(b.piece_map())<=7 else "pieces_8_12" if len(b.piece_map())<=12 else "pieces_gt_12"]+=1
    write(DATA/"collection_report.json",{"completed_roots":done,"total_roots":len(seeds),
          "positions":counts,"piece_distribution":dict(phases),"labelled":False,
          "scope":"采集数据，无Stockfish标签；blocked包含旧数据及跨组重合，后续不导出训练"})
    print("已完成搜索起点",done,"；局面数量",counts,flush=True)


def collect(limit):
    import torch
    from model_residual_value import ResidualValueModel
    from model_cnn import ChessCNN
    from search_engine import SearchEngine,neural_evaluator,policy_order
    seeds=json.loads((DATA/"seeds.json").read_text(encoding="utf-8"))
    paths=[ROOT/n for n in ["chess_model_ranked_epoch4.pt","chess_model_balanced.pt",
          "model_residual_value.py","model_cnn.py","board_encoder.py","move_encoder.py","search_engine.py"]]
    # Exclude old policy/value/choice inputs from the new corpus.
    oldpaths=[ROOT/n for n in ["train.jsonl","val.jsonl","value_train.jsonl","value_val.jsonl",
                               "move_choices_train.jsonl","move_choices_val.jsonl"]]
    if not all(p.exists() for p in oldpaths):
        raise ValueError("缺少旧train/val、value或move_choices JSONL，请保留这些文件用于去重")
    config={"hashes":[filehash(p) for p in paths+oldpaths+[DATA/"seeds.json",Path(__file__).resolve()]],
            "seconds":1,"depth":3,"sample_cap":24,"version":1}
    db=sqlite3.connect(DATA/"collection.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS config(value TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS done(id INTEGER PRIMARY KEY)")
    db.execute("CREATE TABLE IF NOT EXISTS positions(k TEXT PRIMARY KEY,fen TEXT,split TEXT,seed INTEGER,source TEXT)")
    old=db.execute("SELECT value FROM config").fetchone()
    if old and json.loads(old[0])!=config:raise ValueError("输入或模型改变，请保留并重命名数据目录后重建")
    if not old:
        print("首次建立旧数据去重索引，可能需要几分钟……",flush=True)
        for path in oldpaths:
            with path.open(encoding="utf-8-sig") as f:
                for line in f:
                    if not line.strip():continue
                    row=json.loads(line)
                    for field in ["position","parent_position"]:
                        if field in row:
                            fen=chess.Board(row[field]).fen()
                            db.execute("INSERT OR IGNORE INTO positions VALUES(?,?,'blocked',-1,'old')",(key(fen),fen))
            print("索引完成：",path.name,flush=True)
        db.execute("INSERT INTO config VALUES(?)",(json.dumps(config),));db.commit()
    torch.set_num_threads(4);device=torch.device("cpu")
    model=ResidualValueModel().to(device)
    model.load_state_dict(torch.load(paths[0],map_location=device,weights_only=True));model.eval()
    policy=ChessCNN().to(device)
    policy.load_state_dict(torch.load(paths[1],map_location=device,weights_only=True));policy.eval()
    evaluate=neural_evaluator(model,device)
    warm=chess.Board();evaluate(warm);policy_order(warm,policy,device)
    done={x[0] for x in db.execute("SELECT id FROM done")}
    added=0
    try:
        for seed in seeds:
            if seed["id"] in done:continue
            b=chess.Board()
            for m in seed["history"]:b.push_uci(m)
            if b.fen()!=seed["fen"]:raise ValueError("历史不匹配")
            rng=random.Random(seed["id"])
            reservoir=[];n=0;seen=set()
            def recording(board):
                nonlocal n
                fen=board.fen()
                # No terminal states reach this evaluator. Sampling never changes its score.
                if fen not in seen:
                    seen.add(fen);n+=1
                    if len(reservoir)<24:reservoir.append(fen)
                    else:
                        j=rng.randrange(n)
                        if j<24:reservoir[j]=fen
                return evaluate(board)
            engine=SearchEngine(recording,seconds=1,max_depth=3,claim_draw=True)
            engine.choose(b,policy_order(b,policy,device))
            with db:
                put(db,b.fen(),seed["split"],seed["id"],"game_root")
                for fen in reservoir:put(db,fen,seed["split"],seed["id"],"search_eval")
                db.execute("INSERT INTO done VALUES(?)",(seed["id"],))
            added+=1
            if added%10==0:print("本次完成起点：",added,flush=True)
            if limit and added>=limit:break
    except KeyboardInterrupt:
        db.rollback()
        print("已暂停，同一命令续跑。",flush=True)
    finally:
        report(db,seeds);db.close()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("step",choices=["download","prepare","collect"])
    ap.add_argument("--limit",type=int,default=100,help="collect本次起点数；0为剩余全部")
    a=ap.parse_args()
    if a.limit<0:ap.error("limit不能为负")
    DATA.mkdir(exist_ok=True)
    if a.step=="download":download()
    elif a.step=="prepare":prepare()
    else:collect(a.limit)


if __name__=="__main__":main()
