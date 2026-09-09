"""Exact bounded AND/OR verification of the supplied first move.
No score engine, no selective pruning, no history-blind cache.
"""
import argparse,json,hashlib,sqlite3,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
ROOT=Path(__file__).resolve().parent
class BudgetExceeded(Exception):pass

def verify(task):
    import chess
    row,nodes_limit,seconds=task;start=time.perf_counter();nodes=0
    b=chess.Board(row['setup_position']);b.push_uci(row['setup_move'])
    if b.fen()!=row['position']:raise ValueError('题目起点重建不一致')
    attacker=b.turn;solution=row['solution'];bound=2*row['mate_moves']-1
    def solve(left,ply):
        nonlocal nodes
        nodes+=1
        if nodes>nodes_limit or time.perf_counter()-start>seconds:raise BudgetExceeded()
        if b.is_checkmate():return b.turn!=attacker
        if b.is_game_over() or left<=0:return False
        if b.turn!=attacker and b.can_claim_draw():return False
        moves=list(b.legal_moves)
        hint=solution[ply] if ply<len(solution) else None
        moves.sort(key=lambda m:(m.uci()==hint,b.gives_check(m),b.is_capture(m)),reverse=True)
        attacking=b.turn==attacker
        for m in moves:
            b.push(m)
            try:win=solve(left-1,ply+1)
            finally:b.pop()
            if attacking and win:return True
            if not attacking and not win:return False
        return not attacking
    first=chess.Move.from_uci(solution[0])
    if first not in b.legal_moves:raise ValueError('题库首选非法')
    b.push(first)
    try:
        passed=solve(bound-1,1)
        status='verified' if passed else 'refuted'
    except BudgetExceeded:status='unresolved'
    finally:b.pop()
    return {**row,'verification':status,'verified_move':first.uci() if status=='verified' else None,
            'verified_within_plies':bound if status=='verified' else None,'nodes':nodes,'seconds':time.perf_counter()-start}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path);ap.add_argument('--workers',type=int,default=16)
    ap.add_argument('--limit',type=int,default=300);ap.add_argument('--nodes',type=int,default=200000)
    ap.add_argument('--seconds',type=float,default=20);args=ap.parse_args()
    if not 1<=args.workers<=24 or args.limit<0 or args.nodes<1 or not 0<args.seconds<3600:ap.error('参数超出范围')
    if args.source is None:
        fs=sorted((ROOT/'mate_course_candidates').glob('*/preparation_report.json'))
        if not fs:ap.error('找不到候选数据')
        args.source=fs[-1].parent
    files=[args.source/f'{s}_candidates.jsonl' for s in ['train','val','test']]
    rows=[]
    for p in files:
        with p.open(encoding='utf-8-sig') as f:rows.extend(json.loads(x) for x in f if x.strip())
    # Interleave difficulty and split in the pilot rather than checking only mateIn1.
    import random
    random.Random(42).shuffle(rows)
    ids=[r['puzzle_id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('题目ID重复')
    folder=args.source/'forced_verification';folder.mkdir(exist_ok=True)
    cfg={'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'nodes':args.nodes,'seconds':args.seconds}
    db=sqlite3.connect(folder/'progress.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS config (payload TEXT)');db.execute('CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,payload TEXT)')
    old=db.execute('SELECT payload FROM config').fetchone()
    if old and json.loads(old[0])!=cfg:raise ValueError('数据或核验配置已变化；请保留原目录后另建核验目录')
    if not old:db.execute('INSERT INTO config VALUES(?)',(json.dumps(cfg),));db.commit()
    done={x[0] for x in db.execute('SELECT id FROM results')};pending=[r for r in rows if r['puzzle_id'] not in done]
    if args.limit:pending=pending[:args.limit]
    print('总候选',len(rows),'已核验',len(done),'本次',len(pending),'进程',args.workers,flush=True)
    pool=ProcessPoolExecutor(max_workers=args.workers);active={};it=iter(pending);added=0;interrupted=False
    def submit():
        r=next(it,None)
        if r is None:return False
        active[pool.submit(verify,(r,args.nodes,args.seconds))]=r;return True
    try:
        for _ in range(args.workers*2):
            if not submit():break
        while active:
            finished,_=wait(active,timeout=.5,return_when=FIRST_COMPLETED)
            for f in finished:
                row=active.pop(f)
                try:r=f.result()
                except Exception as e:r={**row,'verification':'error','error':repr(e)}
                db.execute('INSERT INTO results VALUES(?,?)',(row['puzzle_id'],json.dumps(r)));db.commit();added+=1
                if added%25==0:print('本次完成',added,'/',len(pending),flush=True)
                submit()
    except KeyboardInterrupt:
        interrupted=True;print('停止提交新题；已保存结果可续跑，等待在途任务结束。',flush=True)
    finally:
        for f in active:f.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
        all_results=[json.loads(x[0]) for x in db.execute('SELECT payload FROM results')]
        counts={}
        for r in all_results:
            k=r['theme'];counts.setdefault(k,{})
            status=r['verification'];counts[k][status]=counts[k].get(status,0)+1
        report={'total':len(rows),'completed':len(all_results),'new_this_run':added,'counts':counts,
                'remaining':len(rows)-len(all_results),'interrupted':interrupted,
                'scope':'Verified supplied FIRST MOVE forces mate within tagged bound; not a proof of shortest mate or a list of all winning first moves. History before puzzle setup unavailable.',
                'overlap_audit':'NOT DONE: do not treat pools as independent holdouts or start training yet.'}
        (folder/'verification_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        for split in ['train','val','test']:
            with (folder/f'{split}_verified.jsonl').open('w',encoding='utf-8') as f:
                for r in all_results:
                    if r['split']==split and r['verification']=='verified':f.write(json.dumps(r)+'\n')
        db.close();print(json.dumps(report,ensure_ascii=False,indent=2),flush=True);print('报告：',folder/'verification_report.json')

if __name__=='__main__':main()
