"""v4 endgame baseline; Stockfish defends. Does not train or modify weights."""
import argparse,json,hashlib,time,os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor,as_completed
from zipfile import ZipFile,ZIP_DEFLATED
ROOT=Path(__file__).resolve().parent

def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def play(task):
    import chess,chess.engine,chess.pgn,torch
    from model_cnn import ChessCNN
    from model_residual_value import ResidualValueModel
    from neural_inference_v3 import TracedEvaluator
    from neural_search_v4 import NeuralSearchV4
    from search_engine import policy_order
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
    folder=Path(task['folder']);i=task['id']
    b=chess.Board(task['fen']);strong=task['strong']
    if not b.is_valid() or b.is_game_over():raise ValueError('测试局面无效')
    start=b.copy(stack=True)
    model=ResidualValueModel().cpu().eval()
    model.load_state_dict(torch.load(task['model'],map_location='cpu',weights_only=True))
    policy=ChessCNN().cpu().eval()
    policy.load_state_dict(torch.load(ROOT/'chess_model_balanced.pt',map_location='cpu',weights_only=True))
    evaluator=TracedEvaluator(model);evaluator.validate([b])
    search=NeuralSearchV4(evaluator,seconds=5,max_depth=5,claim_draw=True)
    records=[];reason='move_limit';result='*';strong_moves=0;lost_piece=False
    sf=chess.engine.SimpleEngine.popen_uci(task['engine'])
    try:
        sf.configure({'Threads':1,'Hash':64})
        teacher=sf.id
        for ply in range(160):
            outcome=b.outcome()
            if outcome:
                reason=outcome.termination.name;result=outcome.result();break
            fen=b.fen()
            if b.turn==strong:
                preferred=policy_order(b,policy,torch.device('cpu'))
                with torch.inference_mode():move,_,stats=search.choose(b,preferred)
                side='v4';strong_moves+=1
                if move is None:
                    if not b.can_claim_draw():raise ValueError('无效和棋申领')
                    reason='v4_claimed_draw';result='1/2-1/2';break
            else:
                # The defending side takes any available rule draw.
                if b.can_claim_draw():
                    reason='defender_claimed_draw';result='1/2-1/2';break
                t=time.perf_counter()
                answer=sf.play(b,chess.engine.Limit(nodes=100000))
                move=answer.move;stats={'wall_seconds':time.perf_counter()-t};side='Stockfish'
            if move not in b.legal_moves:raise ValueError('非法走法')
            san=b.san(move);b.push(move)
            if not (b.pieces(chess.ROOK,strong) or b.pieces(chess.QUEEN,strong)):lost_piece=True
            records.append({'position':fen,'move':move.uci(),'san':san,'side':side,'stats':stats})
            write(folder/f'case_{i}.json',{'task':task,'moves':records,'status':'running'})
        outcome=b.outcome()
        if outcome:reason=outcome.termination.name;result=outcome.result()
        success=bool(outcome and outcome.winner==strong and b.is_checkmate())
        g=chess.pgn.Game.from_board(b)
        g.headers['Event']='v4 endgame baseline vs Stockfish defender'
        g.headers['White']='v4' if strong else 'Stockfish'
        g.headers['Black']='Stockfish' if strong else 'v4'
        g.headers['Result']=result;g.headers['Termination']=reason
        g.headers['Case']=task['name']
        (folder/f'case_{i}.pgn').write_text(str(g)+'\n',encoding='utf-8')
        summary={'id':i,'name':task['name'],'success':success,'result':result,'reason':reason,
                 'strong_moves':strong_moves,'plies':len(records),'lost_major_piece':lost_piece,
                 'final_fen':b.fen(),'teacher':teacher}
        write(folder/f'case_{i}.json',{'task':task,'moves':records,'summary':summary})
        return summary
    finally:sf.quit()

def main():
    import chess
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path)
    ap.add_argument('--engine',type=Path)
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    if not 1<=args.workers<=4:ap.error('workers为1至4')
    if args.model is None:
        files=sorted((ROOT/'value_full_runs').glob('*/value_full_epoch2.pt'))
        if not files:ap.error('缺少模型，请用--model指定')
        args.model=files[-1]
    if args.engine is None:
        files=list(ROOT.glob('stockfish*/**/stockfish*.exe'))
        if len(files)!=1:ap.error('请用--engine指定Stockfish.exe完整路径')
        args.engine=files[0]
    if not args.engine.is_file() or not args.model.is_file():ap.error('引擎或模型路径不存在')
    bases=[('KR_far','7k/8/8/8/8/8/R7/4K3 w - - 0 1'),
           ('KR_edge','8/8/8/3k4/8/8/8/R3K3 w - - 0 1'),
           ('KQ_far','7k/8/8/8/8/8/Q7/4K3 w - - 0 1'),
           ('KQ_center','8/8/3k4/8/8/8/8/Q3K3 w - - 0 1')]
    folder=ROOT/'endgame_baseline_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');folder.mkdir(parents=True)
    tasks=[]
    for name,fen in bases:
        b=chess.Board(fen)
        for flipped in [False,True]:
            board=b.mirror() if flipped else b
            assert board.is_valid() and not board.is_game_over()
            tasks.append({'id':len(tasks)+1,'name':name+('_black' if flipped else '_white'),
                          'fen':board.fen(),'strong':not flipped,'folder':str(folder),'model':str(args.model.resolve()),'engine':str(args.engine.resolve())})
    config={'tasks':tasks,'seconds':5,'depth':5,'threads_per_worker':4,'workers':args.workers,
            'defender_nodes':100000,'max_plies':160,'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'engine_sha256':hashlib.sha256(args.engine.read_bytes()).hexdigest(),
            'scope':'8个固定基础残局，镜像并非独立样本；Stockfish有限节点防守，不保证最优防守。'}
    write(folder/'config.json',config)
    results=[];errors=[]
    print('8个残局；v4执优势方；Stockfish防守。结果：',folder,flush=True)
    os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(play,t):t for t in tasks}
        for f in as_completed(futures):
            try:
                r=f.result();results.append(r);print(r,flush=True)
            except Exception as e:errors.append({'id':futures[f]['id'],'error':repr(e)});print(errors[-1],flush=True)
            write(folder/'summary.json',{'completed':len(results),'wins':sum(r['success'] for r in results),'results':results,'errors':errors})
    with ZipFile(folder.with_suffix('.zip'),'w',ZIP_DEFLATED) as z:
        for p in folder.iterdir():
            if p.is_file():z.write(p,p.name)
    print('请发我：',folder.with_suffix('.zip'),flush=True)

if __name__=='__main__':main()
