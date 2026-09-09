"""Validation-only mate benchmark: policy first move and v4 first move.
Different answers are verified against ALL defenses by verify_mate_course.
"""
import argparse,json,hashlib,time,os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
ROOT=Path(__file__).resolve().parent
STATE={}

def init_worker(model_path):
    import chess,torch
    from model_cnn import ChessCNN
    from model_residual_value import ResidualValueModel
    from neural_inference_v3 import TracedEvaluator
    from neural_search_v4 import NeuralSearchV4
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
    policy=ChessCNN().cpu().eval();policy.load_state_dict(torch.load(ROOT/'chess_model_balanced.pt',map_location='cpu',weights_only=True))
    model=ResidualValueModel().cpu().eval();model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True))
    evaluator=TracedEvaluator(model);evaluator.validate([chess.Board()])
    STATE.update(policy=policy,evaluator=evaluator,search=NeuralSearchV4(evaluator,seconds=5,max_depth=5,claim_draw=True))

def solve(row):
    import chess,torch
    from search_engine import policy_order
    from verify_mate_course import verify
    b=chess.Board(row['setup_position']);b.push_uci(row['setup_move'])
    if b.fen()!=row['position']:raise ValueError('FEN不一致')
    STATE['evaluator'].validate([b])
    order=policy_order(b,STATE['policy'],torch.device('cpu'))
    before=(b.fen(),list(b.move_stack))
    with torch.inference_mode():move,mate,stats=STATE['search'].choose(b,order)
    if before!=(b.fen(),list(b.move_stack)):raise ValueError('棋盘历史被改变')
    first=order[0].uci();chosen=move.uci() if move else None
    checked={}
    for u in set([first,chosen]):
        if u is None:checked[u]={'status':'not_solved_claim_draw'};continue
        if chess.Move.from_uci(u) not in b.legal_moves:raise ValueError('非法推荐走法')
        if u==row['verified_move']:
            checked[u]={'status':'verified','method':'previous_exact_proof'}
        else:
            trial={**row,'solution':[u]+row['solution'][1:]}
            r=verify((trial,1000000,60))
            checked[u]={'status':r['verification'],'method':'exact_and_or','nodes':r['nodes'],'seconds':r['seconds']}
    return {'puzzle_id':row['puzzle_id'],'theme':row['theme'],'position':row['position'],
            'reference_move':row['verified_move'],'policy_move':first,'v4_move':chosen,
            'policy_verdict':checked[first],'v4_verdict':checked[chosen],'search':stats,
            'scope':'First move preserves mate within tagged bound, not full-game execution or shortest-mate proof.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path);ap.add_argument('--model',type=Path)
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--per-theme',type=int,default=0)
    args=ap.parse_args()
    if not 1<=args.workers<=4 or args.per_theme<0:ap.error('workers 1至4；per-theme不小于0')
    if args.source is None:
        fs=sorted((ROOT/'mate_course_candidates').glob('*/forced_verification/overlap_report.json'))
        if not fs:ap.error('找不到重合报告')
        args.source=fs[-1].parent
    overlap=json.loads((args.source/'overlap_report.json').read_text(encoding='utf-8'))
    if overlap['errors']:ap.error('重合扫描存在错误，请先排查')
    for tag in ['mateIn2','mateIn3']:
        if overlap['counts']['val'][tag]['found_in_historical_jsonl']!=0:ap.error('验证题目有历史重合，需先筛选')
    if args.model is None:
        models=sorted((ROOT/'value_full_runs').glob('*/value_full_epoch2.pt'))
        if not models:ap.error('找不到基础模型')
        args.model=models[-1]
    model_sha=hashlib.sha256(args.model.read_bytes()).hexdigest()
    if model_sha!='4e44421b422c8467a32c06c85a31a5591b8397ae24c2f54fdd39c778fe7dd4a0':ap.error('不是稳定基础模型，请用--model指定原全量第2轮')
    rows=[];count={}
    with (args.source/'val_verified.jsonl').open(encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line)
            if r['verification']!='verified' or r['split']!='val':raise ValueError('未核验或非验证集')
            t=r['theme'];count[t]=count.get(t,0)
            if not args.per_theme or count[t]<args.per_theme:rows.append(r);count[t]+=1
    import random
    random.Random(42).shuffle(rows)
    folder=ROOT/'mate_v4_benchmarks'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');folder.mkdir(parents=True)
    config={'model':str(args.model.resolve()),'model_sha256':model_sha,'workers':args.workers,
            'threads_per_worker':4,'seconds':5,'depth':5,'selected_counts':count,
            'source':str(args.source.resolve()),'source_sha256':hashlib.sha256((args.source/'val_verified.jsonl').read_bytes()).hexdigest(),
            'code_hashes':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['search_engine.py','neural_search_v4.py','neural_search_v2.py','neural_inference_v3.py','neural_fast.py','verify_mate_course.py']},
            'policy_sha256':hashlib.sha256((ROOT/'chess_model_balanced.pt').read_bytes()).hexdigest(),
            'scope':'mateIn1 is historical-overlap regression only. mateIn2/3 have no root overlap in audited JSONL; not proof of zero prior exposure. Final test untouched.'}
    (folder/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    results=[];errors=[];it=iter(rows);active={}
    os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
    pool=ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(str(args.model.resolve()),))
    def submit():
        r=next(it,None)
        if r is None:return False
        active[pool.submit(solve,r)]=r;return True
    def save():
        groups={}
        for r in results:
            g=groups.setdefault(r['theme'],{'completed':0,'policy':{},'v4':{}});g['completed']+=1
            for mode in ['policy','v4']:
                status=r[mode+'_verdict']['status'];g[mode][status]=g[mode].get(status,0)+1
        for g in groups.values():
            for mode in ['policy','v4']:
                g[mode]['proven_success_rate']=g[mode].get('verified',0)/g['completed']
        report={'requested':len(rows),'completed':len(results),'errors':errors,'groups':groups,
                'note':'unresolved is unknown, not proven failure; success rate is a lower bound if any unresolved remain.'}
        tmp=folder/'report.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(folder/'benchmark_report.json')
    print('验证题：',count,'；4线程/进程；5秒、最高深度5。',flush=True)
    try:
        for _ in range(args.workers*2):
            if not submit():break
        with (folder/'details.jsonl').open('w',encoding='utf-8') as out:
            while active:
                finished,_=wait(active,timeout=.5,return_when=FIRST_COMPLETED)
                for f in finished:
                    r=active.pop(f)
                    try:
                        result=f.result();results.append(result);out.write(json.dumps(result)+'\n');out.flush()
                    except Exception as e:errors.append({'puzzle_id':r['puzzle_id'],'error':repr(e)})
                    save();print('完成',len(results),'/',len(rows),'；错误',len(errors),flush=True);submit()
    except KeyboardInterrupt:print('停止提交新题；等待在途任务结束。重新运行会新建测试，不续跑。',flush=True)
    finally:
        for f in active:f.cancel()
        pool.shutdown(wait=True,cancel_futures=True);save()
    print('发我：',folder/'benchmark_report.json',flush=True)

if __name__=='__main__':main()
