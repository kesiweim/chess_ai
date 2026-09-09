"""Small candidate-only fine tune with old score and ranking replay."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from board_encoder import encode_board
from model_residual_value import ResidualValueModel

ROOT = Path(__file__).resolve().parent
VALUES = {1:100,2:320,3:330,4:500,5:900,6:0}


def read(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def key(fen):
    return " ".join(chess.Board(fen).fen().split()[:4])


def material(b):
    return sum(v*(len(b.pieces(p,b.turn))-len(b.pieces(p,not b.turn))) for p,v in VALUES.items())


def groups(rows):
    result = defaultdict(list)
    for r in rows:
        result[r["parent_id"]].append(r)
    for group in result.values():
        b = chess.Board(group[0]["parent_position"])
        moves = [r["move"] for r in group]
        if len(moves)!=len(set(moves)) or set(moves)!={m.uci() for m in b.legal_moves}:
            raise ValueError("旧候选组不完整")
    return list(result.values())


def batch(rows, device):
    boards = [chess.Board(r["position"]) for r in rows]
    x = torch.stack([encode_board(b) for b in boards]).to(device)
    mat = torch.tensor([material(b) for b in boards],device=device,dtype=torch.float32)
    y = torch.tensor([r["value"] for r in rows],device=device,dtype=torch.float32)
    terminal = torch.tensor([bool(r.get("terminal",False)) for r in rows],device=device)
    return x, mat, y, terminal


def predictions(model, rows, device):
    x,mat,y,terminal = batch(rows,device)
    p,_ = model(x,mat)
    return torch.where(terminal,y,p),y


def ranking_loss(p,y):
    # Parent values have opposite sign from child values.
    gap = y[None,:]-y[:,None]
    pred_gap = p[None,:]-p[:,None]
    mask = gap > .10
    if mask.any():
        return F.relu(gap.clamp(max=.20)[mask]-pred_gap[mask]).mean()
    return p.sum()*0


def group_loss(model, rows, device):
    p,y = predictions(model,rows,device)
    return F.mse_loss(p,y) + .5*ranking_loss(p,y)


@torch.inference_mode()
def evaluate(model, pairs, val_groups, old_val, device):
    model.eval()
    pair_hits = 0
    for pair in pairs:
        p,_ = predictions(model,pair["children"],device)
        pair_hits += int(p[0].item()<p[1].item())
    hits, regret = 0,0.
    for group in val_groups:
        p,y = predictions(model,group,device)
        selected = min(range(len(group)),key=lambda i:(p[i].item(),group[i]["move"]))
        loss = max(0., y[selected].item()-y.min().item())
        hits += int(loss <= 1e-6)
        regret += loss
    squared = 0.
    for i in range(0,len(old_val),256):
        p,y = predictions(model,old_val[i:i+256],device)
        squared += (p-y).square().sum().item()
    return {"training_pair_hits":pair_hits,"training_pairs":len(pairs),
            "old_ranking_hits":hits,"old_ranking_groups":len(val_groups),
            "old_ranking_regret":regret/len(val_groups),
            "old_value_mse":squared/len(old_val),"old_value_count":len(old_val)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=3)
    args=ap.parse_args()
    if args.epochs<1:
        ap.error("epochs 必须为正")
    report=json.loads((ROOT/"neural_fix_data/preparation_report.json").read_text(encoding="utf-8"))
    if not report["complete"]:
        raise ValueError("请先完成全部复核，再开始训练")
    pairs=json.loads((ROOT/"neural_fix_data/confirmed_pairs.json").read_text(encoding="utf-8"))
    if not pairs:
        raise ValueError("没有通过复核的样本，请把 preparation_report.json 发回分析")
    # Collapse identical encoded input pairs, regardless of move counter.
    unique={}
    for p in pairs:
        ch=p["children"]
        if len(ch)!=2 or ch[1]["value"]-ch[0]["value"]<.10:
            raise ValueError("新训练对标签方向或间隔错误")
        unique.setdefault(tuple(" ".join(c["position"].split()[:5]) for c in ch),p)
    pairs=list(unique.values())
    old_train=read(ROOT/"value_train.jsonl")
    old_val=read(ROOT/"value_val.jsonl")
    choice_train=read(ROOT/"move_choices_train.jsonl")
    choice_val=read(ROOT/"move_choices_val.jsonl")
    train_groups=groups(choice_train)
    val_groups=groups(choice_val)
    train_keys={key(r["position"]) for r in old_train+choice_train}
    train_keys.update(key(r["parent_position"]) for r in choice_train)
    for p in pairs:
        train_keys.add(key(p["parent_position"]))
        train_keys.update(key(c["position"]) for c in p["children"])
    val_groups=[g for g in val_groups if key(g[0]["parent_position"]) not in train_keys
                and not any(key(r["position"]) in train_keys for r in g)]
    old_val=[r for r in old_val if key(r["position"]) not in train_keys]
    if not old_train or not train_groups or not val_groups or not old_val:
        raise ValueError("清理后有数据集为空，不能可靠评估")
    torch.manual_seed(42)
    rng=random.Random(42)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    base=ROOT/"chess_model_ranked_epoch4.pt"
    model=ResidualValueModel().to(device)
    model.load_state_dict(torch.load(base,map_location=device,weights_only=True))
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.000005,weight_decay=.001)
    folder=ROOT/"neural_fix_runs"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    metrics=[]
    baseline=evaluate(model,pairs,val_groups,old_val,device)
    metrics.append({"epoch":0,**baseline})
    print("设备：",device,flush=True)
    print("微调前：",baseline,flush=True)
    manifest={"base_sha256":hashlib.sha256(base.read_bytes()).hexdigest(),
              "pairs_sha256":hashlib.sha256((ROOT/"neural_fix_data/confirmed_pairs.json").read_bytes()).hexdigest(),
              "lr":.000005,"seed":42,"epochs":args.epochs,
              "scope":"训练对命中是记忆检查；旧验证集是保留能力检查，最终仍需新开局实战。"}
    (folder/"config.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    def save_metrics():
        tmp=folder/"training_report.tmp"
        tmp.write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
        tmp.replace(folder/"training_report.json")
    save_metrics()
    try:
        for epoch in range(1,args.epochs+1):
            rng.shuffle(pairs)
            model.train()
            total=0.
            for step,pair in enumerate(pairs,1):
                optimizer.zero_grad(set_to_none=True)
                correction=group_loss(model,pair["children"],device)
                replay_group=group_loss(model,rng.choice(train_groups),device)
                replay=rng.sample(old_train,min(64,len(old_train)))
                p,y=predictions(model,replay,device)
                # More weight on retention than on fitting the small error set.
                loss=.5*correction + replay_group + F.mse_loss(p,y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                total+=loss.item()
                if step%50==0:
                    print(f"第 {epoch} 轮：{step}/{len(pairs)}",flush=True)
            metric=evaluate(model,pairs,val_groups,old_val,device)
            metric.update(epoch=epoch,training_loss=total/len(pairs))
            metrics.append(metric)
            path=folder/f"neural_fix_epoch{epoch}.pt"
            tmp=path.with_suffix(".tmp")
            torch.save(model.state_dict(),tmp)
            tmp.replace(path)
            save_metrics()
            print(f"第 {epoch} 轮：",metric,flush=True)
            print("候选模型：",path,flush=True)
    except KeyboardInterrupt:
        print("已停止。完整轮次的候选已保存；重新运行会从原第4轮权重新建训练，不会续接中断步。")
    print("报告：",folder/"training_report.json",flush=True)
    print("没有自动替换实战模型。请先比较报告，再使用新开局验收。",flush=True)


if __name__=="__main__":
    main()
