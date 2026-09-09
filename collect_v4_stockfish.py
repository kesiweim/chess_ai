"""Local opponent calibration. Opponent moves are NOT teacher labels."""
import argparse, hashlib, json, os, subprocess, sys, time, random
from pathlib import Path
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED
import chess, chess.pgn, chess.engine
ROOT=Path(__file__).resolve().parent
BASE_SHA='4e44421b422c8467a32c06c85a31a5591b8397ae24c2f54fdd39c778fe7dd4a0'
OPENINGS=[('Italian',['e4','e5','Nf3','Nc6','Bc4','Bc5']),
 ('Queens_Gambit',['d4','d5','c4','e6','Nc3','Nf6']),
 ('English',['c4','c5','Nc3','Nc6','Nf3','Nf6'])]
OPENINGS += [('Spanish',['e4','e5','Nf3','Nc6','Bb5','a6']),
 ('Sicilian',['e4','c5','Nf3','d6','d4','cxd4']),
 ('French',['e4','e6','d4','d5','Nc3','Nf6']),
 ('CaroKann',['e4','c6','d4','d5','Nc3','dxe4']),
 ('Slav',['d4','d5','c4','c6','Nf3','Nf6']),
 ('KingsIndian',['d4','Nf6','c4','g6','Nc3','Bg7']),
 ('Scotch',['e4','e5','Nf3','Nc6','d4','exd4'])]

class SampledEvaluator:
 """Reservoir over evaluator calls (cache misses), not all search nodes."""
 def __init__(self,base,capacity,seed):
  self.base=base;self.capacity=capacity;self.rng=random.Random(seed);self.samples=[];self.n=0;self.root_ply=0
 def reset(self,board):
  self.samples=[];self.n=0;self.root_ply=len(board.move_stack)
 def __call__(self,board):
  value=self.base(board)
  self.n+=1
  if not self.capacity:return value
  j=self.n-1 if self.n<=self.capacity else self.rng.randrange(self.n)
  if j<self.capacity:
   item=(board.copy(stack=False),list(board.move_stack[self.root_ply:]),float(value))
   if j==len(self.samples):self.samples.append(item)
   else:self.samples[j]=item
  return value
 def export(self,board,metadata):
  rows=[];seen=set()
  for b,path,value in self.samples:
   fen=b.fen();moves=[m.uci() for m in path];key=(fen,tuple(moves))
   if key in seen:continue
   seen.add(key)
   rows.append({**metadata,'source':'search_evaluation','position':fen,
     'root_position':board.fen(),'root_ply':len(board.move_stack),'path_from_root':moves,
     'candidate_static_value':value,'value_pov':'side_to_move','teacher_label':None})
  return rows

def append_rows(path,rows):
 with path.open('a',encoding='utf-8') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)
def claim(b):
 if b.is_repetition(3):return {'reason':'threefold','intended_move':None}
 if b.is_fifty_moves():return {'reason':'fifty_moves','intended_move':None}
 for m in list(b.legal_moves):
  b.push(m)
  try:
   if b.is_repetition(3):return {'reason':'threefold','intended_move':m.uci()}
   if b.is_fifty_moves():return {'reason':'fifty_moves','intended_move':m.uci()}
  finally:b.pop()
 return None

def worker(a):
 import torch
 from model_cnn import ChessCNN
 from model_residual_value import ResidualValueModel
 from search_engine import policy_order
 from neural_search_v4 import NeuralSearchV4
 from neural_inference_v3 import TracedEvaluator
 cfg=json.loads((a.output/'config.json').read_text(encoding='utf-8'));job=cfg['jobs'][a.job]
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
 device=torch.device('cpu');policy=ChessCNN().to(device)
 policy.load_state_dict(torch.load(cfg['policy'],map_location=device,weights_only=True));policy.eval()
 model=ResidualValueModel().to(device)
 model.load_state_dict(torch.load(cfg['model'],map_location=device,weights_only=True));model.eval()
 ev=TracedEvaluator(model);probe=chess.Board()
 for s in job['opening_moves']:probe.push_san(s)
 ev.validate([chess.Board(),probe]);policy_order(probe,policy,device)
 sampled=SampledEvaluator(ev,cfg['samples_per_move'],8300+a.job)
 search=NeuralSearchV4(sampled,seconds=cfg['seconds'],max_depth=cfg['depth'],claim_draw=True)
 sf=chess.engine.SimpleEngine.popen_uci(cfg['engine'],timeout=60)
 try:
  sf.configure({'Threads':1,'Hash':128,'UCI_LimitStrength':False,'Skill Level':job['skill']})
  for color in [True,False]:
   ident=a.job*2+(1 if color else 2);stem=a.output/f'game_{ident:02d}'
   board=probe.copy(stack=True);logs=[];result='*';reason='running';error=None
   game_token=object();opp=f'Stockfish_skill_{job["skill"]}'
   def save():
    g=chess.pgn.Game.from_board(board)
    g.headers.update({'Event':'Local v4 vs Stockfish calibration','Date':datetime.now().strftime('%Y.%m.%d'),
      'White':'v4' if color else opp,'Black':opp if color else 'v4','Result':result,
      'Termination':reason,'OpeningLabel':job['opening'],'PairGroup':str(a.job),
      'ValueModelSHA256':cfg['model_sha256']})
    t=stem.with_suffix('.pgn.tmp');t.write_text(str(g)+'\n',encoding='utf-8');t.replace(stem.with_suffix('.pgn'))
    write(stem.with_suffix('.json'),{'game_id':ident,'pair_group':a.job,'skill':job['skill'],'v4_white':color,
       'opening':job['opening'],'opening_moves':job['opening_moves'],'result':result,'reason':reason,
       'error':error,'moves':logs,'final_fen':board.fen(),'teacher_labels':False})
   save()
   try:
    while True:
     out=board.outcome()
     if out:result=out.result();reason=out.termination.name;break
     if len(board.move_stack)>=cfg['max_plies']:reason='move_limit_unfinished';break
     isv4=board.turn==color;side='v4' if isv4 else opp;start=time.perf_counter()
     meta={'game_id':ident,'pair_group':a.job,'opening_group':job['opening'],
           'skill':job['skill'],'v4_white':color,'root_ply':len(board.move_stack)}
     append_rows(stem.with_suffix('.jsonl'),[{**meta,'source':'played_root','position':board.fen(),
       'teacher_label':None,'history_uci':[m.uci() for m in board.move_stack]}])
     if isv4:
      preferred=policy_order(board,policy,device)
      sampled.reset(board)
      move,_,stats=search.choose(board,preferred)
      rows=sampled.export(board,meta)
      append_rows(stem.with_suffix('.jsonl'),rows)
      stats['evaluated_calls']=sampled.n;stats['saved_search_samples']=len(rows)
     else:
      # Explicit match convention: Stockfish claims any available legal draw.
      c=claim(board) if board.can_claim_draw() else None
      if c:
       logs.append({'position':board.fen(),'side':side,'move':None,'claim':c});result='1/2-1/2';reason='claimed_'+c['reason'];break
      answer=sf.play(board,chess.engine.Limit(time=cfg['sf_seconds']),game=game_token,info=chess.engine.INFO_BASIC)
      move=answer.move;stats={k:answer.info[k] for k in ['depth','nodes','time','nps'] if k in answer.info}
     stats['wall_seconds']=time.perf_counter()-start
     if move is None:
      c=claim(board)
      if not c:raise ValueError('引擎返回空走法，但没有合法申和')
      logs.append({'position':board.fen(),'side':side,'move':None,'claim':c,'stats':stats})
      result='1/2-1/2';reason='claimed_'+c['reason'];break
     if move not in board.legal_moves:raise ValueError('非法走法 '+str(move))
     san=board.san(move)
     logs.append({'ply':len(board.move_stack),'position':board.fen(),'side':side,'move':move.uci(),'san':san,'stats':stats})
     board.push(move);save()
     print(f'局{ident} {side} {san} {stats["wall_seconds"]:.2f}s',flush=True)
   except BaseException as e:
    reason='interrupted' if isinstance(e,KeyboardInterrupt) else 'error';error=repr(e);save();raise
   save();print(f'局{ident}结束 {result} {reason}',flush=True)
 finally:
  try:sf.quit()
  except Exception:sf.close()

def summarize(folder,cfg):
 groups={str(s):{'v4_wins':0,'stockfish_wins':0,'draws':0,'unfinished':0,'errors':0} for s in cfg['skills']}
 for p in folder.glob('game_*.json'):
  r=json.loads(p.read_text(encoding='utf-8'));g=groups[str(r['skill'])]
  if r['error']:g['errors']+=1
  elif r['result']=='*':g['unfinished']+=1
  elif r['result']=='1/2-1/2':g['draws']+=1
  else:
   vwin=(r['result']=='1-0')==r['v4_white'];g['v4_wins' if vwin else 'stockfish_wins']+=1
 for g in groups.values():
  n=g['v4_wins']+g['stockfish_wins']+g['draws'];g['finished']=n
  g['v4_score_finished_only']=(g['v4_wins']+.5*g['draws'])/n if n else None
 return groups

def main():
 ap=argparse.ArgumentParser(description='v4 对 Stockfish 多档位，本地自动交换颜色')
 ap.add_argument('--skills',type=int,nargs='+',default=[0,3,6]);ap.add_argument('--pairs',type=int,default=None,help='指定后每档使用相同数量开局；默认0/3/6档使用2/10/6个开局')
 ap.add_argument('--samples-per-move',type=int,default=32)
 ap.add_argument('--workers',type=int,default=4);ap.add_argument('--seconds',type=float,default=5)
 ap.add_argument('--depth',type=int,default=5);ap.add_argument('--sf-seconds',type=float,default=1)
 ap.add_argument('--max-plies',type=int,default=240);ap.add_argument('--model',type=Path);ap.add_argument('--engine',type=Path)
 ap.add_argument('--job',type=int,default=None,help=argparse.SUPPRESS);ap.add_argument('--output',type=Path,help=argparse.SUPPRESS)
 a=ap.parse_args()
 if a.job is not None:worker(a);return
 if (a.pairs is not None and not 1<=a.pairs<=len(OPENINGS)) or not 1<=a.workers<=4:ap.error('pairs须1至10；workers须1至4')
 if not 0<=a.samples_per_move<=256:ap.error('samples-per-move须0至256')
 if any(s<0 or s>20 for s in a.skills) or len(set(a.skills))!=len(a.skills):ap.error('skills须为不重复的0至20整数')
 if a.seconds<=0 or a.sf_seconds<=0 or a.depth<1 or a.max_plies<10:ap.error('时间、深度、步数参数无效')
 required=['model_cnn.py','model_residual_value.py','search_engine.py','neural_fast.py','neural_search_v2.py','neural_search_v4.py','neural_inference_v3.py','chess_model_balanced.pt']
 for n in required:
  if not (ROOT/n).is_file():ap.error('项目目录缺少 '+n)
 if a.model is None:
  found=[p for p in sorted((ROOT/'value_full_runs').glob('*/value_full_epoch2.pt')) if sha(p)==BASE_SHA]
  if not found:ap.error('找不到原版v4全量第2轮模型，请用 --model 指定')
  a.model=found[-1]
 if not a.model.is_file() or sha(a.model)!=BASE_SHA:ap.error('必须使用原版v4全量第2轮模型，不能使用残局微调模型')
 if a.engine is None:a.engine=ROOT/'stockfish-windows-x86-64-universal'/'stockfish'/'stockfish-windows-x86-64-universal.exe'
 if not a.engine.is_file():ap.error('找不到Stockfish，请用 --engine 指定exe路径')
 with chess.engine.SimpleEngine.popen_uci(str(a.engine.resolve())) as sf:
  for n in ['Skill Level','UCI_LimitStrength','Threads','Hash']:
   if n not in sf.options:ap.error('引擎缺少选项 '+n)
  for s in a.skills:sf.options['Skill Level'].parse(s)
  sf.configure({'Threads':1,'Hash':128,'UCI_LimitStrength':False,'Skill Level':a.skills[0]});engine_id=sf.id
 for _,moves in OPENINGS:
  b=chess.Board()
  for san in moves:b.push_san(san)
 folder=ROOT/'v4_stockfish_collection_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');folder.mkdir(parents=True)
 cfg={'model':str(a.model.resolve()),'model_sha256':sha(a.model),'policy':str(ROOT/'chess_model_balanced.pt'),
 'policy_sha256':sha(ROOT/'chess_model_balanced.pt'),'engine':str(a.engine.resolve()),'engine_sha256':sha(a.engine),'engine_id':engine_id,
 'skills':a.skills,'seconds':a.seconds,'depth':a.depth,'sf_seconds':a.sf_seconds,'max_plies':a.max_plies,'workers':a.workers,
 'v4_threads':4,'sf_threads':1,'sf_hash_mb':128,'UCI_LimitStrength':False,
 'samples_per_move':a.samples_per_move,
 'jobs':[{'skill':s,'opening':n,'opening_moves':moves} for s in a.skills for n,moves in OPENINGS[:(a.pairs if a.pairs is not None else {0:2,3:10,6:6}.get(s,6))]],
 'implementation_sha256':{n:sha(ROOT/n) for n in required if n.endswith('.py')},
 'scope':'Collection, not Elo. Reservoir of neural evaluator cache-miss calls across all iterations, including unfinished iteration; terminal nodes excluded. Sampling adds overhead and can change timed search depth. Candidate values and opponent moves are NOT teacher labels. Full real-game history plus root-relative search paths retained. Future split must group same opening across skills/colors, then purge overlapping positions and prior holdouts before labeling. No automatic training.',
 'collector_sha256':sha(Path(__file__))}
 write(folder/'config.json',cfg)
 print(f'共{len(cfg["jobs"])*2}局；档位{a.skills}；同时最多{a.workers}局。',flush=True)
 print('日志目录：',folder,flush=True)
 env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
 active={};nextjob=0;failed=[];interrupted=False
 try:
  while nextjob<len(cfg['jobs']) or active:
   while nextjob<len(cfg['jobs']) and len(active)<a.workers:
    i=nextjob;nextjob+=1;log=(folder/f'pair_{i:02d}.log').open('w',encoding='utf-8')
    try:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--job',str(i),'--output',str(folder)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    except BaseException:log.close();raise
    active[i]=(p,log);print('启动组',i+1,cfg['jobs'][i],flush=True)
   for i,(p,log) in list(active.items()):
    code=p.poll()
    if code is not None:
     log.close();del active[i]
     if code:failed.append(i)
     print('完成组',i+1,'退出码',code,flush=True)
     write(folder/'summary.json',{'by_skill':summarize(folder,cfg),'failed_jobs':failed})
   time.sleep(.25)
 except KeyboardInterrupt:
  interrupted=True;print('正在停止；保留已保存棋谱，不支持续下本盘。',flush=True)
 finally:
  for p,log in active.values():
   if p.poll() is None:
    if os.name=='nt':subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    else:p.terminate()
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   log.close()
  summary={'by_skill':summarize(folder,cfg),'failed_jobs':failed,'interrupted':interrupted,'planned_games':2*len(cfg['jobs'])}
  summary['saved_games']=len(list(folder.glob('game_*.json')));write(folder/'summary.json',summary)
  counts={};total=0
  for p in folder.glob('game_*.jsonl'):
   with p.open(encoding='utf-8') as f:
    for line in f:
     try:r=json.loads(line)
     except json.JSONDecodeError:counts['invalid_lines']=counts.get('invalid_lines',0)+1;continue
     k=r['source'];counts[k]=counts.get(k,0)+1;total+=1
  write(folder/'collection_report.json',{'records':total,'by_source':counts,'teacher_labeled':0,
    'samples_per_move':a.samples_per_move,'planned_games':summary['planned_games'],
    'note':'Raw records, not deduplicated training samples; exclude error/interrupted games before training.'})
  with ZipFile(folder.with_suffix('.zip'),'w',ZIP_DEFLATED) as z:
   for p in sorted(folder.iterdir()):
    if p.suffix in ['.json','.pgn','.log','.jsonl']:z.write(p,p.name)
  print(json.dumps(summary,ensure_ascii=False,indent=2));print('发我这个压缩包：',folder.with_suffix('.zip'))
 if failed:raise SystemExit('有任务出错，不计作输棋，请发结果ZIP排查')
if __name__=='__main__':main()
