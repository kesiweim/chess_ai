import argparse,json,random,hashlib
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parent

def read(p):
    with p.open(encoding='utf-8-sig') as f:return [json.loads(x) for x in f if x.strip()]
def write(p,r):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)

def main():
    import chess,torch
    from board_encoder import encode_board
    from model_residual_value import ResidualValueModel
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path);ap.add_argument('--epochs',type=int,default=3);a=ap.parse_args()
    if a.epochs<1:ap.error('epochs必须为正')
    if a.model is None:
        files=sorted((ROOT/'value_full_runs').glob('*/value_full_epoch2.pt'))
        if not files:ap.error('找不到基础模型')
        a.model=files[-1]
    sha=hashlib.sha256(a.model.read_bytes()).hexdigest()
    if sha!='4e44421b422c8467a32c06c85a31a5591b8397ae24c2f54fdd39c778fe7dd4a0':ap.error('请选择稳定全量第2轮权重')
    rng=random.Random(42);torch.manual_seed(42);torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=ROOT/'endgame_ranking_data';train=read(data/'train.jsonl');val=read(data/'val.jsonl')
    def keys(rs):return {x['group'] for r in rs for x in [r]+r['choices']}
    if keys(train)&keys(val):raise ValueError('父或子局面跨组重合')
    replay=read(ROOT/'value_train.jsonl')+read(ROOT/'value_expansion_labels'/'expanded_value_train.jsonl')
    replay=[r for r in replay if len(chess.Board(r['position']).piece_map())>3];rng.shuffle(replay);replay=replay[:20000]
    oldval=read(ROOT/'value_val.jsonl')
    folder=ROOT/'endgame_ranking_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');folder.mkdir(parents=True)
    model=ResidualValueModel().to(device);model.load_state_dict(torch.load(a.model,map_location=device,weights_only=True))
    # Keep the feature extractor fixed; update the value head only in this experiment.
    for p in model.features.parameters():p.requires_grad_(False)
    values={1:100,2:320,3:330,4:500,5:900,6:0};records=[]
    for r in train+val:records.extend(r['choices'])
    records+=replay+oldval
    xs=torch.empty(len(records),19,8,8,device=device);ms=torch.empty(len(records),device=device)
    print('编码缓存',len(records),'局面；设备',device,flush=True)
    for i in range(0,len(records),1024):
        rs=records[i:i+1024];bs=[chess.Board(r['position']) for r in rs]
        xs[i:i+len(rs)]=torch.stack([encode_board(b) for b in bs]).to(device)
        ms[i:i+len(rs)]=torch.tensor([sum(v*(len(b.pieces(p,b.turn))-len(b.pieces(p,not b.turn))) for p,v in values.items()) for b in bs],device=device)
        for j,r in enumerate(rs):r['_i']=i+j
    def batch(rs):
        ids=torch.tensor([r['_i'] for r in rs],device=device);return xs[ids],ms[ids]
    # Anchor old outputs rather than asking the model to relearn old targets.
    model.eval();anchors=[]
    with torch.inference_mode():
        for i in range(0,len(replay),512):
            x,m=batch(replay[i:i+512]);p,c=model(x,m);anchors.extend((m/300+c).cpu().tolist())
    for r,v in zip(replay,anchors):r['_anchor']=v
    @torch.inference_mode()
    def measure(epoch,loss=None):
        model.eval();sq=ab=0.;hits=preserve=0;gaps=[]
        for r in val:
            x,m=batch(r['choices']);p,c=model(x,m);scores=(-p).tolist()
            for i,child in enumerate(r['choices']):
                if child['terminal_score'] is not None:scores[i]=1000. if child['terminal_score']>0 else 0.
            choice=r['choices'][max(range(len(scores)),key=scores.__getitem__)]
            if choice['win']:
                preserve+=1;gaps.append(choice['distance']-r['best_distance']);hits+=choice['distance']==r['best_distance']
        for i in range(0,len(oldval),512):
            rs=oldval[i:i+512];x,m=batch(rs);p,_=model(x,m);y=torch.tensor([r['value'] for r in rs],device=device)
            sq+=(p-y).square().sum().item();ab+=(p-y).abs().sum().item()
        result={'epoch':epoch,'ranking_val_parents':len(val),'fastest_hits':hits,'preserves_win':preserve,
                'mean_extra_plies_among_preserved':sum(gaps)/len(gaps) if gaps else None,
                'old_mse':sq/len(oldval),'old_mae':ab/len(oldval),'loss':loss}
        report.append(result);write(folder/'training_report.json',report);print(result,flush=True)
    write(folder/'data_report.json',{'train_parents':len(train),'val_parents':len(val),'replay':len(replay),'oldval':len(oldval),
        'base_sha256':sha,'base':str(a.model.resolve()),'epochs':a.epochs,'lr':1e-5,'frozen':'features',
        'loss':'0.5*pairwise logit hinge + 1.0*old-logit distillation',
        'margin':'0.04*min(extra DTM halfplies,8); 0.5 for draw-vs-win',
        'data_hashes':{s:hashlib.sha256((data/f'{s}.jsonl').read_bytes()).hexdigest() for s in ['train','val']}})
    report=[];measure(0);opt=torch.optim.AdamW(model.head.parameters(),lr=1e-5,weight_decay=.001)
    try:
        for epoch in range(1,a.epochs+1):
            pairs=[]
            for r in train:
                for bad in r['bad']:
                    g=r['choices'][rng.choice(r['good'])];b=r['choices'][bad]
                    margin=.5 if not b['win'] else .04*min(b['distance']-r['best_distance'],8)
                    pairs.append((g,b,margin))
            rng.shuffle(pairs);total=steps=0;model.train();model.features.eval()
            for i in range(0,len(pairs),128):
                ps=pairs[i:i+128];n=len(ps);old=rng.sample(replay,min(128,len(replay)))
                x,m=batch([p[0] for p in ps]+[p[1] for p in ps]+old)
                opt.zero_grad(set_to_none=True);_,c=model(x,m);logits=m/300+c
                # Children are opponent-to-move, hence NEGATE logits for parent ranking.
                difference=-logits[:n]+logits[n:2*n]
                margin=torch.tensor([p[2] for p in ps],device=device)
                ranking=torch.relu(margin-difference).mean()
                anchors=torch.tensor([r['_anchor'] for r in old],device=device)
                loss=.5*ranking+(logits[2*n:]-anchors).square().mean()
                if not torch.isfinite(loss):raise ValueError('损失非有限')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.);opt.step();total+=loss.item();steps+=1
                if steps%100==0:print('轮',epoch,'对比样本',min(i+128,len(pairs)),'/',len(pairs),flush=True)
            p=folder/f'rook_ranking_epoch{epoch}.pt';tmp=p.with_suffix('.tmp');torch.save(model.state_dict(),tmp);tmp.replace(p)
            measure(epoch,total/steps)
    except KeyboardInterrupt:print('只保留完整轮次；重新运行会新建实验，不续训。')
    print('发我报告目录：',folder,flush=True)

if __name__=='__main__':main()
