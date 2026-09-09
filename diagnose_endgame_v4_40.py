"""Read-only comparison of one-ply evaluation, v4 choice and exact DTM labels."""
import argparse,json,hashlib,os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor,as_completed
from zipfile import ZipFile,ZIP_DEFLATED
ROOT=Path(__file__).resolve().parent
STATE={}

def init_worker(model_path,tables):
    import chess,chess.gaviota,torch
    from model_cnn import ChessCNN
    from model_residual_value import ResidualValueModel
    from neural_inference_v3 import TracedEvaluator
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
    policy=ChessCNN().cpu().eval();policy.load_state_dict(torch.load(ROOT/'chess_model_balanced.pt',map_location='cpu',weights_only=True))
    model=ResidualValueModel().cpu().eval();model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True))
    ev=TracedEvaluator(model);ev.validate([chess.Board()])
    tb=chess.gaviota.PythonTablebase();tb.add_directory(tables)
    import atexit
    atexit.register(tb.close)
    STATE.update(policy=policy,evaluator=ev,tb=tb)

def child_label(b):
    """b is AFTER the root player's move. Result from ROOT player's view."""
    root=not b.turn;outcome=b.outcome()
    if outcome:
        return {'outcome':('win' if outcome.winner==root else 'draw' if outcome.winner is None else 'loss'),
                'dtm_total_plies':1 if outcome.winner==root else None,'terminal':outcome.termination.name}
    # Defender is entitled to any immediate/intended-move draw claim.
    claim=b.can_claim_draw()
    dtm=STATE['tb'].probe_dtm(b)
    root_wdl=-STATE['tb'].probe_wdl(b)
    distance=1+abs(dtm) if root_wdl else None
    clock_ok=bool(root_wdl and abs(dtm)<=100-b.halfmove_clock)
    if claim or not root_wdl or not clock_ok:category='draw'
    else:category='win' if root_wdl>0 else 'loss'
    return {'outcome':category,'dtm_total_plies':distance,'dtm_child_stm':dtm,
            'dtm_preserves_win_ignoring_claims':root_wdl>0,'defender_can_claim_draw':claim,
            'within_fifty_move_budget':clock_ok}

def diagnose(task):
    import chess,torch
    from neural_search_v4 import NeuralSearchV4
    from search_engine import policy_order
    b=chess.Board(task['start_fen'])
    for u in task['history']:b.push_uci(u)
    if b.fen()!=task['fen']:raise ValueError('历史重建错误')
    if len(b.piece_map())!=3:raise ValueError('仅支持三子残局')
    STATE['evaluator'].validate([b]);original_state=(b.fen(),list(b.move_stack))
    preferred=policy_order(b,STATE['policy'],torch.device('cpu'))
    engine=NeuralSearchV4(STATE['evaluator'],seconds=5,max_depth=5,claim_draw=True)
    with torch.inference_mode():chosen,_,stats=engine.choose(b,preferred)
    if original_state!=(b.fen(),list(b.move_stack)):raise ValueError('搜索损坏历史')
    candidates=[]
    for m in list(b.legal_moves):
        san=b.san(m);b.push(m)
        try:
            label=child_label(b)
            if b.is_checkmate():value=1000.0
            elif b.is_game_over():value=0.0
            else:value=-STATE['evaluator'](b)
        finally:b.pop()
        candidates.append({'move':m.uci(),'san':san,'one_ply_model_score':value,**label})
    wins=[r for r in candidates if r['outcome']=='win'];best_distance=min((r['dtm_total_plies'] for r in wins),default=None)
    for r in candidates:
        r['extra_dtm_plies']=r['dtm_total_plies']-best_distance if r['outcome']=='win' and best_distance is not None else None
    ranked=sorted(candidates,key=lambda r:r['one_ply_model_score'],reverse=True)
    bymove={r['move']:r for r in candidates}
    chosen_uci=chosen.uci() if chosen else None
    result={**task,'root_dtm':STATE['tb'].probe_dtm(b),'best_dtm_plies':best_distance,
            'dtm_best_moves':[r['move'] for r in wins if r['dtm_total_plies']==best_distance],
            'model_top':ranked[0],'v4_choice':bymove.get(chosen_uci),
            'v4_action':'move' if chosen else 'claim_draw','original_choice':bymove[task['original_move']],
            'search':stats,'candidates_model_order':ranked}
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if not 1<=args.workers<=4:ap.error('workers为1至4')
    if args.model is None:
        files=sorted((ROOT/'value_full_runs').glob('*/value_full_epoch2.pt'))
        if not files:ap.error('找不到稳定模型')
        args.model=files[-1]
    h=hashlib.sha256(args.model.read_bytes()).hexdigest()
    if h!='4e44421b422c8467a32c06c85a31a5591b8397ae24c2f54fdd39c778fe7dd4a0':ap.error('需要稳定全量第2轮模型，不能使用残局微调模型')
    tables=ROOT/'endgame_course_data'/'gaviota'
    manifest=json.loads((tables.parent/'download_report.json').read_text(encoding='utf-8'))
    for name in ['krk.gtb.cp4','kqk.gtb.cp4']:
        if hashlib.sha256((tables/name).read_bytes()).hexdigest()!=manifest['files'][name]:raise ValueError('残局库校验失败')
    tasks=json.loads((ROOT/'diagnosis_positions_40.json').read_text(encoding='utf-8'))
    folder=ROOT/'endgame_diagnosis_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');folder.mkdir(parents=True)
    config={'model':str(args.model.resolve()),'model_sha256':h,'workers':args.workers,'threads':4,'seconds':5,'depth':5,
            'code_hashes':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['neural_search_v4.py','neural_inference_v3.py','neural_search_v2.py','neural_fast.py','search_engine.py']},
            'scope':'16 correlated snapshots: 12 old failed KRK snapshots plus 4 successful KQK snapshots from game16 of the 40-game arena, for diagnosis only. DTM accounts for fifty-move budget and immediate defender claims, not all possible future repetition histories. One-ply model ranking is not a search evaluation.'}
    (folder/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
    results=[];errors=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(str(args.model.resolve()),str(tables))) as pool:
        futures={pool.submit(diagnose,t):t for t in tasks}
        for f in as_completed(futures):
            try:
                r=f.result();results.append(r)
                (folder/f"case_{r['id']}.json").write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
                print('完成',r['id'],'最佳DTM',r['best_dtm_plies'],'模型首选',r['model_top']['san'],
                      'v4',r['v4_choice']['san'] if r['v4_choice'] else '申领和棋',flush=True)
            except Exception as e:errors.append({'id':futures[f]['id'],'error':repr(e)});print(errors[-1],flush=True)
    summary={'completed':len(results),'errors':errors,'snapshots':[]}
    for r in sorted(results,key=lambda x:x['id']):
        summary['snapshots'].append({k:r[k] for k in ['id','case','ply_index','best_dtm_plies','model_top','v4_choice','original_choice','search']})
    (folder/'diagnosis_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with ZipFile(folder.with_suffix('.zip'),'w',ZIP_DEFLATED) as z:
        for p in folder.iterdir():z.write(p,p.name)
    print('请发整个ZIP：',folder.with_suffix('.zip'),flush=True)

if __name__=='__main__':main()
