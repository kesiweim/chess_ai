import argparse,json,hashlib,time,random
from pathlib import Path
import chess
import torch
from model_residual_value import ResidualValueModel
from search_engine import SearchEngine,neural_evaluator
from neural_fast import FastEvaluator,FastSearchEngine

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path)
    ap.add_argument('--threads',type=int,default=1)
    args=ap.parse_args()
    if args.threads<1:ap.error('threads必须为正')
    if args.model is None:
        files=sorted((ROOT/'value_full_runs').glob('*/value_full_epoch2.pt'))
        if not files:ap.error('找不到全量第2轮模型，请用 --model 指定')
        args.model=files[-1]
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    model=ResidualValueModel().cpu().eval()
    model.load_state_dict(torch.load(args.model,map_location='cpu',weights_only=True))
    old=neural_evaluator(model,torch.device('cpu'));fast=FastEvaluator(model)
    cases=json.loads((ROOT/'neural_speed_cases.json').read_text(encoding='utf-8'))
    boards=[]
    for c in cases:
        b=chess.Board()
        for san in c['opening']:b.push_san(san)
        for uci in c['moves']:b.push_uci(uci)
        if b.fen()!=c['fen']:raise ValueError('测试局面重建失败')
        boards.append(b)
    # Exercise both turns, castling, en-passant, clocks, and a promotion board.
    probes=[chess.Board()]
    b=chess.Board()
    for san in ['e4','a6','e5','d5','exd6','exd6','Nf3','Nf6','Be2','Be7','O-O']:
        b.push_san(san);probes.append(b.copy())
    probes += [chess.Board('4k3/P7/8/8/8/8/8/4K3 w - - 100 60')]
    rng=random.Random(42);b=chess.Board()
    for _ in range(256):
        if b.is_game_over():b=chess.Board()
        b.push(rng.choice(list(b.legal_moves)));probes.append(b.copy())
    fast.validate(probes+boards)
    errors=[abs(old(b)-fast(b)) for b in probes+boards]
    report={'model':str(args.model.resolve()),'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'search_sha256':hashlib.sha256((ROOT/'search_engine.py').read_bytes()).hexdigest(),
            'threads':args.threads,'torch':torch.__version__,'max_evaluation_error':max(errors),'fixed_depth':[],'timed':[]}
    path=ROOT/'neural_speed_report.json'
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
    if max(errors)>1e-6:save();raise ValueError('评分一致性检查失败')
    print('编码与评分核验通过。开始同深度测试。',flush=True)
    def run(kind,b,depth,seconds):
        before=(b.fen(),list(b.move_stack))
        engine=(SearchEngine(old,seconds=seconds,max_depth=depth,claim_draw=True) if kind=='old'
                else FastSearchEngine(fast,seconds=seconds,max_depth=depth,claim_draw=True))
        start=time.perf_counter()
        with torch.inference_mode():move,mate,stats=engine.choose(b)
        elapsed=time.perf_counter()-start
        if (b.fen(),list(b.move_stack))!=before:raise ValueError('搜索破坏了棋盘历史')
        return {'move':move.uci() if move else None,'mate_one':mate,'wall_seconds':elapsed,**stats}
    for i,b in enumerate(boards[:12]):
        results={}
        for kind in (['old','fast'] if i%2==0 else ['fast','old']):results[kind]=run(kind,b,2,120)
        x,y=results['old'],results['fast']
        same=x['move']==y['move'] and x.get('score')==y.get('score') and x['nodes']==y['nodes']
        completed=all(r['mate_one'] or r['depth']==2 for r in results.values())
        report['fixed_depth'].append({'case':i,**results,'same':same,'completed':completed});save()
        print(f"固定深度 {i+1}/12：一致={same}；旧{x['wall_seconds']:.3f}s，新{y['wall_seconds']:.3f}s",flush=True)
        if not same or not completed:raise ValueError('固定深度检查未通过，报告已保存，请发报告排查')
    print('固定深度检查通过。开始5秒、深度上限5测试。',flush=True)
    for i,b in enumerate(boards[:8]):
        results={}
        for kind in (['old','fast'] if i%2==0 else ['fast','old']):results[kind]=run(kind,b,5,5)
        report['timed'].append({'case':i,**results});save()
        print(f"限时 {i+1}/8：旧深度{results['old']['depth']}，新深度{results['fast']['depth']}",flush=True)
    a=sum(r['old']['wall_seconds'] for r in report['fixed_depth']);b=sum(r['fast']['wall_seconds'] for r in report['fixed_depth'])
    report['fixed_depth_speedup']=a/b if b else None
    save();print('固定深度加速倍数：',report['fixed_depth_speedup'],flush=True)
    print('发我报告：',path,flush=True)

if __name__=='__main__':main()
