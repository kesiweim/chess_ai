"""Controlled weighted replay experiment on the previous pilot snapshot."""
import argparse
import hashlib
import json
import math
import random
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent
VALUES={1:100,2:320,3:330,4:500,5:900,6:0}


def write(p,obj):
    tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(p)


def read(p):
    return [json.loads(line) for line in p.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def key(fen):
    return " ".join(fen.split()[:4])


def select_source(root, explicit=None):
    if explicit:
        candidates=[explicit]
    else:
        candidates=sorted((root/"value_pilot_runs").glob("*"),reverse=True)
    for folder in candidates:
        if not all((folder/n).is_file() for n in
                   ["snapshot_train.jsonl","snapshot_val.jsonl","data_report.json","training_report.json"]):
            continue
        info=json.loads((folder/"data_report.json").read_text(encoding="utf-8"))
        # Match the completed pilot reported by the user, not a newer partial run.
        if info["train"]["count"]==62179 and info["val"]["count"]==6257:
            return folder,info
    raise ValueError("未找到62179训练/6257验证的原试验；请用 --source 指定其时间戳目录")


def sample_weight(value):
    return 1.5 if value <= .1 else 1.0


def stats(rows):
    return {"count":len(rows),
            "mean_value":sum(r["value"] for r in rows)/len(rows),
            "negative":sum(r["value"]<-.1 for r in rows),
            "near_zero":sum(abs(r["value"])<=.1 for r in rows),
            "positive":sum(r["value"]>.1 for r in rows),
            "mate":sum(r.get("mate") is not None for r in rows)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=3)
    ap.add_argument("--source",type=Path)
    args=ap.parse_args()
    if args.epochs<1:ap.error("epochs必须为正")
    import chess
    import torch
    import torch.nn.functional as F
    from board_encoder import encode_board
    from model_residual_value import ResidualValueModel
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng=random.Random(42)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source,previous_info=select_source(ROOT,args.source)
    print("固定数据来源：",source,flush=True)
    train=read(source/"snapshot_train.jsonl")
    val=read(source/"snapshot_val.jsonl")
    if len(train)!=62179 or len(val)!=6257:
        raise ValueError("快照数量与报告不符")
    for records,split in [(train,"train"),(val,"val")]:
        for r in records:
            if r["split"]!=split or r["score_pov"]!="side_to_move" or not math.isfinite(r["value"]) or not -1<=r["value"]<=1:
                raise ValueError("快照标签或分组错误")
    rows=train+val
    label_config=previous_info.get("label_config")
    if len(train)<1000 or len(val)<200:raise ValueError("数据不足：至少1000训练、200验证")
    old_train=read(ROOT/"value_train.jsonl")
    old_val=read(ROOT/"value_val.jsonl")
    if not old_train or not old_val:raise ValueError("旧数据为空")
    if {r["game_id"] for r in train}&{r["game_id"] for r in val}:
        raise ValueError("发现新训练/验证组对局交叉")
    train_keys={key(r["position"]) for r in train+old_train}
    before=len(val)
    # Base epoch4 has also seen the old choice training data.
    path=ROOT/"move_choices_train.jsonl"
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                r=json.loads(line)
                train_keys.add(key(r["position"]))
                train_keys.add(key(r["parent_position"]))
    val=[r for r in val if key(r["position"]) not in train_keys]
    old_val=[r for r in old_val if key(r["position"]) not in train_keys]
    if len(val)!=before or len(old_val)!=previous_info["old_val"]:
        raise ValueError("验证集发生变化，停止以避免不可比；请恢复原旧数据文件")
    if not val or not old_val:raise ValueError("去重后验证数据为空")
    base=ROOT/"chess_model_ranked_epoch4.pt"
    if hashlib.sha256(base.read_bytes()).hexdigest()!=previous_info["base_sha256"]:
        raise ValueError("原第4轮权重已改变")
    weight_mean=sum(sample_weight(r["value"]) for r in train)/len(train)
    if len(old_train)!=previous_info["old_train"]:raise ValueError("旧训练数据数量改变")
    folder=ROOT/"value_balanced_runs"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    for name,records in [("snapshot_train.jsonl",train),("snapshot_val.jsonl",val)]:
        with (folder/name).open("w",encoding="utf-8") as f:
            for r in records:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    base=ROOT/"chess_model_ranked_epoch4.pt"
    info={"snapshot_total":len(rows),"train":stats(train),"val":stats(val),
          "removed_new_val_overlap":before-len(val),"old_train":len(old_train),"old_val":len(old_val),
          "label_config":label_config,"base":str(base),"base_sha256":hashlib.sha256(base.read_bytes()).hexdigest(),
          "epochs":args.epochs,"lr":1e-5,"batch_new":128,"batch_old":32,
          "source_pilot":str(source.resolve()),
          "source_hashes":{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in
                           ["snapshot_train.jsonl","snapshot_val.jsonl","training_report.json"]},
          "raw_weights":{"negative":1.5,"near_zero":1.5,"positive":1.0},
          "weight_normalizer":weight_mean,"old_loss_coefficient":1.0,
          "scope":"固定部分标注试验；新验证检查分布内评价，旧验证检查遗忘，不是独立棋力认证。"}
    write(folder/"data_report.json",info)
    print("设备：",device,"；固定新训练",len(train),"；新验证",len(val),flush=True)
    model=ResidualValueModel().to(device)
    model.load_state_dict(torch.load(base,map_location=device,weights_only=True))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.001)

    def batch(records):
        boards=[chess.Board(r["position"]) for r in records]
        x=torch.stack([encode_board(b) for b in boards]).to(device)
        mat=torch.tensor([sum(v*(len(b.pieces(p,b.turn))-len(b.pieces(p,not b.turn))) for p,v in VALUES.items())
                          for b in boards],device=device,dtype=torch.float32)
        y=torch.tensor([r["value"] for r in records],device=device,dtype=torch.float32)
        return x,mat,y

    @torch.inference_mode()
    def evaluate(records):
        model.eval()
        sq=ab=material_sq=0.
        sums={n:[0.,0] for n in ["negative","near_zero","positive"]}
        for i in range(0,len(records),256):
            x,mat,y=batch(records[i:i+256])
            p,_=model(x,mat)
            sq+=(p-y).square().sum().item()
            ab+=(p-y).abs().sum().item()
            material_sq+=(torch.tanh(mat/300)-y).square().sum().item()
            for name,mask in [("negative",y<-.1),("near_zero",y.abs()<=.1),("positive",y>.1)]:
                sums[name][0]+=(p[mask]-y[mask]).square().sum().item()
                sums[name][1]+=int(mask.sum().item())
        return {"mse":sq/len(records),"mae":ab/len(records),"material_mse":material_sq/len(records),
                "by_target":{n:{"mse":s/c if c else None,"count":c} for n,(s,c) in sums.items()}}
    metrics=[]
    def measure(epoch,loss=None):
        r={"epoch":epoch,"new_val":evaluate(val),"old_val":evaluate(old_val),"train_loss":loss}
        metrics.append(r)
        write(folder/"training_report.json",metrics)
        print(json.dumps(r,ensure_ascii=False),flush=True)
    measure(0)
    old_metrics=json.loads((source/"training_report.json").read_text(encoding="utf-8"))
    reference=next(r for r in old_metrics if r["epoch"]==0)
    for name in ["new_val","old_val"]:
        if abs(metrics[0][name]["mse"]-reference[name]["mse"])>1e-4:
            raise ValueError("起始验证MSE与原试验不一致，停止训练，请发报告核对编码、模型和数据")
    try:
        for epoch in range(1,args.epochs+1):
            rng.shuffle(train)
            model.train();total=0.;steps=0
            for i in range(0,len(train),128):
                new=train[i:i+128]
                replay=rng.sample(old_train,min(32,len(old_train)))
                x,mat,y=batch(new+replay)
                optimizer.zero_grad(set_to_none=True)
                p,_=model(x,mat)
                weights=torch.where(y[:len(new)]<=.1,1.5,1.0)/weight_mean
                new_loss=(weights*(p[:len(new)]-y[:len(new)]).square()).mean()
                loss=new_loss+F.mse_loss(p[len(new):],y[len(new):])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                total+=loss.item();steps+=1
                if steps%100==0:print(f"第{epoch}轮：新数据 {min(i+128,len(train))}/{len(train)}",flush=True)
            path=folder/f"value_balanced_epoch{epoch}.pt"
            tmp=path.with_suffix(".tmp")
            torch.save(model.state_dict(),tmp);tmp.replace(path)
            measure(epoch,total/steps)
            print("候选保存：",path,flush=True)
    except KeyboardInterrupt:
        print("已停止；保留完整轮次。再次运行将从原第4轮权重新建实验，不恢复中断批次。",flush=True)
    print("报告目录：",folder,flush=True)
    print("候选未自动接入实战。",flush=True)


if __name__=="__main__":main()
