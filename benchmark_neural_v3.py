"""Sequential 1/2/4-thread tests. Does not change live arena/play settings."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parent

def compatible(a,b):
    if not a.get('eligible') or not b.get('eligible'):return False
    if a['evaluation_values']!=b['reference_values']:return False
    if len(a['fixed_depth'])!=len(b['fixed_depth']):return False
    for x,y in zip(a['fixed_depth'],b['fixed_depth']):
        x=x['fast'];y=y['old']
        if any(x.get(k)!=y.get(k) for k in ['move','score','nodes','depth','mate_one']):return False
    return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path)
    args=ap.parse_args()
    if args.model is None:
        files=sorted((ROOT/'value_full_runs').glob('*/value_full_epoch2.pt'))
        if not files:ap.error('找不到全量第2轮模型，请用 --model 指定')
        args.model=files[-1]
    args.model=args.model.resolve()
    if not args.model.is_file():ap.error('模型不存在')
    folder=ROOT/'neural_v3_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)
    reports=[];failures=[]
    for threads in [1,2,4]:
        print(f'\n===== 测试 {threads} 线程；其他测试尚未启动 =====',flush=True)
        target=ROOT/f'neural_v3_threads_{threads}.json'
        # Archive existing report before running, so a failure cannot reuse stale data.
        if target.exists():target.replace(folder/f'previous_{target.name}')
        r=subprocess.run([sys.executable,str(ROOT/'benchmark_neural_v3_worker.py'),
            '--threads',str(threads),'--model',str(args.model)],cwd=ROOT)
        if target.exists():
            data=json.loads(target.read_text(encoding='utf-8'))
            (folder/target.name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
            if r.returncode==0:reports.append(data)
            else:failures.append({'threads':threads,'exit':r.returncode})
        else:failures.append({'threads':threads,'exit':r.returncode})
    base=next((r for r in reports if r['threads']==1),None)
    candidates=[]
    if base and base.get('eligible'):
        baseline=sum(x['old']['wall_seconds'] for x in base['fixed_depth'])
        candidates.append({'backend':'v2_eager','threads':1,'fixed_seconds':baseline,'speedup_vs_v2_1thread':1.0})
        for r in reports:
            if compatible(r,base):
                elapsed=sum(x['fast']['wall_seconds'] for x in r['fixed_depth'])
                candidates.append({'backend':'v3_traced','threads':r['threads'],'fixed_seconds':elapsed,
                                   'speedup_vs_v2_1thread':baseline/elapsed})
    selected=min(candidates,key=lambda c:c['fixed_seconds']) if candidates else None
    report={'model':str(args.model),'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'runs':reports,'failures':failures,'eligible_candidates':candidates,'recommended':selected,
            'scope':'测试局面上的精确一致性与单进程测速；不是所有局面正确性或棋力保证；不自动替换实战。'}
    path=ROOT/'neural_speed_v3_report.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/path.name).write_bytes(path.read_bytes())
    print('建议配置：',selected,flush=True)
    print('请发报告：',path,flush=True)

if __name__=='__main__':main()
