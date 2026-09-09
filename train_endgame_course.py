import argparse,json,random,hashlib,time
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parent

def read(p):
    with p.open(encoding='utf-8-sig') as f:return [json.loads(x) for x in f if x.strip()]
def write(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)

def main():
    import chess,torch
    from board_encoder import encode_board
    from model_residual_value import ResidualValueModel
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path);ap.add_argument('--epochs',type=int,default=3);args=ap.parse_args()
    if args.epochs<1:ap.error('epochs必须为正')
    if args.model is None:
        files=sorted((ROOT/'value_full_runs').glob('*/value_full_epoch2.pt'))
        if not files:ap.error('缺少全量第2轮权重，请用--model指定')
        args.model=files[-1]
    rng=random.Random(42);torch.manual_seed(42);torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=ROOT/'endgame_course_data';train=read(data/'train.jsonl');val=read(data/'val.jsonl')
    if {r['group'] for r in train}&{r['group'] for r in val}:raise ValueError('训练验证泄漏')
    replay=read(ROOT/'value_train.jsonl')+read(ROOT/'value_expansion_labels'/'expanded_value_train.jsonl')
    replay=[r for r in replay if len(chess.Board(r['position']).piece_map())>3]
    rng.shuffle(replay);replay=replay[:50000]
    oldval=read(ROOT/'value_val.jsonl')
    if not all([train,val,replay,oldval]):raise ValueError('空数据集')
    folder=ROOT/'endgame_course_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');folder.mkdir(parents=True)
    write(folder/'data_report.json',{'train':len(train),'val':len(val),'replay':len(replay),'oldval':len(oldval),
        'model':str(args.model.resolve()),'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'epochs':args.epochs,'lr':1e-5,'batch_endgame':128,'batch_replay':128,'old_loss_weight':1.0,'device':str(device),
        'source_hashes':{n:hashlib.sha256((data/n).read_bytes()).hexdigest() for n in ['train.jsonl','val.jsonl']}})
    model=ResidualValueModel().to(device);model.load_state_dict(torch.load(args.model,map_location=device,weights_only=True))
    records=train+val+replay+oldval
    print('预编码并缓存：',len(records),'；设备',device,flush=True)
    xs=torch.empty(len(records),19,8,8,device=device);ms=torch.empty(len(records),device=device);ys=torch.empty_like(ms)
    values={1:100,2:320,3:330,4:500,5:900,6:0}
    for start in range(0,len(records),1024):
        rs=records[start:start+1024];bs=[chess.Board(r['position']) for r in rs]
        xs[start:start+len(rs)]=torch.stack([encode_board(b) for b in bs]).to(device)
        ms[start:start+len(rs)]=torch.tensor([sum(v*(len(b.pieces(p,b.turn))-len(b.pieces(p,not b.turn))) for p,v in values.items()) for b in bs],device=device)
        ys[start:start+len(rs)]=torch.tensor([r['value'] for r in rs],device=device)
        for j,r in enumerate(rs):r['_idx']=start+j
        if start%10240==0:print('预编码',start+len(rs),flush=True)
    def batch(rs):
        ids=torch.tensor([r['_idx'] for r in rs],device=device);return xs[ids],ms[ids],ys[ids]
    @torch.inference_mode()
    def evaluate(rs):
        model.eval();sq=ab=0.
        for i in range(0,len(rs),512):
            x,m,y=batch(rs[i:i+512]);p,_=model(x,m);sq+=(p-y).square().sum().item();ab+=(p-y).abs().sum().item()
        return {'mse':sq/len(rs),'mae':ab/len(rs)}
    report=[]
    def measure(epoch,loss=None):
        r={'epoch':epoch,'endgame_val':evaluate(val),'old_val':evaluate(oldval),'loss':loss};report.append(r)
        write(folder/'training_report.json',report);print(r,flush=True)
    measure(0);opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.001)
    try:
        for epoch in range(1,args.epochs+1):
            rng.shuffle(train);model.train();total=0.;steps=0
            for i in range(0,len(train),128):
                new=train[i:i+128];old=rng.sample(replay,min(128,len(replay)))
                x,m,y=batch(new+old);opt.zero_grad(set_to_none=True);p,_=model(x,m)
                loss=(p[:len(new)]-y[:len(new)]).square().mean()+(p[len(new):]-y[len(new):]).square().mean()
                if not torch.isfinite(loss):raise ValueError('损失非有限')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=loss.item();steps+=1
                if steps%100==0:print('轮次',epoch,'已处理',i+len(new),flush=True)
            p=folder/f'endgame_epoch{epoch}.pt';tmp=p.with_suffix('.tmp');torch.save(model.state_dict(),tmp);tmp.replace(p)
            measure(epoch,total/steps)
    except KeyboardInterrupt:print('保留完整轮次；重新运行会从基础权重新建实验，不续训。')
    print('报告：',folder,flush=True)

if __name__=='__main__':main()
