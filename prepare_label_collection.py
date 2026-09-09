"""Verify collected games, partition by opening, then resumable teacher labeling."""
import argparse, collections, hashlib, io, json, math, multiprocessing, os
import signal, sqlite3, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import chess, chess.engine, chess.pgn

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = '20260909_122618_199634'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

def key(fen):
    # Conservative overlap grouping: ignore clocks AND en-passant availability.
    return ' '.join(fen.split()[:3])

def fen_strings(x):
    if isinstance(x, str):
        p = x.split()
        if len(p) >= 4 and p[0].count('/') == 7 and p[1] in ('w', 'b'): yield x
    elif isinstance(x, dict):
        for v in x.values(): yield from fen_strings(v)
    elif isinstance(x, list):
        for v in x: yield from fen_strings(v)

class Source:
    def __init__(self, path):
        self.path = path
        self.z = zipfile.ZipFile(path) if path.is_file() else None
        self.names = self.z.namelist() if self.z else [p.name for p in path.iterdir() if p.is_file()]
    def text(self, name):
        return self.z.read(name).decode('utf-8-sig') if self.z else (self.path/name).read_text(encoding='utf-8-sig')
    def close(self):
        if self.z: self.z.close()

def prepare(a):
    if a.out.exists(): raise ValueError('输出目录已存在。准备无需重复；请直接运行 label 子命令。')
    for n in ['train.jsonl', 'val.jsonl']:
        if not (a.history_root/n).is_file(): raise ValueError('历史数据目录缺少 '+n+'；请用 --history-root 指向完整 chess_ai 项目')
    source = a.source
    if source is None:
        options = [ROOT/'v4_stockfish_collection_runs'/DEFAULT_SOURCE, ROOT/(DEFAULT_SOURCE+'.zip')]
        source = next((p for p in options if p.exists()), None)
        if source is None: raise ValueError('找不到本次数据，请用 --source 指定结果目录或 ZIP')
    src = Source(source); records = []; counts = collections.Counter(); games = []
    try:
        for name in sorted(src.names):
            if not name.startswith('game_') or not name.endswith('.json'): continue
            meta = json.loads(src.text(name))
            if meta.get('error') or meta['result'] == '*': counts['excluded_incomplete_games'] += 1; continue
            stem = name[:-5]
            game = chess.pgn.read_game(io.StringIO(src.text(stem+'.pgn')))
            if game is None or game.errors: raise ValueError('棋谱解析失败 '+name)
            b = game.board(); boards = [b.copy(stack=True)]
            for m in game.mainline_moves():
                if m not in b.legal_moves: raise ValueError('非法棋谱 '+name)
                b.push(m); boards.append(b.copy(stack=True))
            if b.fen() != meta['final_fen'] or game.headers['Result'] != meta['result']: raise ValueError('棋谱与记录不一致 '+name)
            if meta['reason'] == 'CHECKMATE':
                if not b.is_checkmate() or b.result() != meta['result']: raise ValueError('将死结果不符 '+name)
            elif meta['result'] == '1/2-1/2':
                if not (b.is_game_over(claim_draw=True)): raise ValueError('无法核验和棋 '+name)
            else: raise ValueError('不支持的结束原因 '+name)
            for line_no, line in enumerate(src.text(stem+'.jsonl').splitlines(), 1):
                if not line.strip(): continue
                r = json.loads(line); ply = r['root_ply']
                if not 0 <= ply < len(boards): raise ValueError('根节点索引错误 '+name)
                root = boards[ply]; c = root.copy(stack=True)
                if r['game_id'] != meta['game_id'] or r['opening_group'] != meta['opening']: raise ValueError('来源不一致')
                if r['source'] == 'played_root':
                    if r['history_uci'] != [m.uci() for m in root.move_stack]: raise ValueError('历史走法不一致')
                elif r['source'] == 'search_evaluation':
                    if r['root_position'] != root.fen(): raise ValueError('搜索起点不一致')
                    for uci in r['path_from_root']:
                        if c.is_game_over(): raise ValueError('搜索路径越过自动终局')
                        m = chess.Move.from_uci(uci)
                        if m not in c.legal_moves: raise ValueError('搜索路径非法')
                        c.push(m)
                else: raise ValueError('未知记录类型')
                if c.fen() != r['position'] or not c.is_valid(): raise ValueError(f'局面核验失败 {name}:{line_no}')
                counts['verified_records'] += 1
                if c.is_game_over(claim_draw=True): counts['excluded_terminal_or_claimable'] += 1; continue
                records.append({'position':c.fen(),'history_uci':[m.uci() for m in c.move_stack],
                    'opening_group':meta['opening'],'game_id':meta['game_id'],'source':r['source'],
                    'skill':meta['skill'],'v4_white':meta['v4_white']})
            games.append(meta['game_id']); print('已核验对局',meta['game_id'],flush=True)
    finally: src.close()
    groups = sorted({r['opening_group'] for r in records}, key=lambda s: hashlib.sha256(('collection-v1:'+s).encode()).hexdigest())
    if len(groups) < 4: raise ValueError('完整开局组不足4个，暂不划分')
    val_groups = set(groups[:max(1, round(len(groups)*.2))])
    group_keys = {'train':set(), 'val':set()}
    for r in records:
        r['split'] = 'val' if r['opening_group'] in val_groups else 'train'
        group_keys[r['split']].add(key(r['position']))
    cross = group_keys['train'] & group_keys['val']
    wanted = group_keys['train'] | group_keys['val']; historical = set(); inventory = []
    # Inventory all historical JSONL except raw arenas/collections and this output.
    for base, dirs, files in os.walk(a.history_root):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and 'collection' not in d.lower() and 'arena' not in d.lower() and Path(base,d).resolve()!=a.out.resolve())
        for n in sorted(files):
            if not n.endswith('.jsonl'): continue
            p = Path(base)/n; h = hashlib.sha256(); lines = 0; hits = 0
            with p.open('rb') as f:
                for raw in f:
                    h.update(raw)
                    if not raw.strip(): continue
                    lines += 1
                    try: obj = json.loads(raw.decode('utf-8-sig'))
                    except Exception as e: raise ValueError(f'历史文件格式错误 {p}:{lines}') from e
                    for fen in fen_strings(obj):
                        k = key(fen)
                        if k in wanted: historical.add(k); hits += 1
            inventory.append({'path':str(p.resolve()),'sha256':h.hexdigest(),'lines':lines,'matching_occurrences':hits})
            print('扫描历史数据',p.name,'命中',hits,flush=True)
    output = {'train':[], 'val':[]}; seen = set()
    # Prefer real-game roots when the same board occurs in sampled search paths.
    records.sort(key=lambda r:(r['source']!='played_root',r['game_id']))
    for r in records:
        k = key(r['position'])
        if k in cross: counts['excluded_cross_split'] += 1; continue
        if k in historical: counts['excluded_historical'] += 1; continue
        if k in seen: counts['excluded_duplicate'] += 1; continue
        seen.add(k); r['id'] = hashlib.sha256(k.encode()).hexdigest(); output[r['split']].append(r)
    if min(map(len,output.values())) < 100: raise ValueError('过滤后某组不足100条，请发终端输出；不会降低排重标准')
    a.out.mkdir(parents=True)
    for split, rows in output.items():
        p = a.out/(split+'_unlabeled.jsonl')
        with p.open('w',encoding='utf-8') as f:
            for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report = {'source':str(source.resolve()),'source_sha256':digest(source) if source.is_file() else None,
        'completed_game_ids':games,'validation_openings':sorted(val_groups),'training_openings':sorted(set(groups)-val_groups),
        'counts':dict(counts),'samples':{s:len(rs) for s,rs in output.items()},'historical_inventory':inventory,
        'overlap_key':'board, turn, castling; ignores ep and clocks conservatively',
        'limits':'Historical JSONL only; raw arena/collection folders excluded. Does not prove absence from PGN/CSV/SQLite or unlisted files. Opening-group validation; no independent final test created.'}
    write(a.out/'preparation_report.json',report); print('准备完成：',report['samples'],a.out,flush=True)

def teacher(task, engine):
    r,nodes = task; b = chess.Board()
    for u in r['history_uci']: b.push_uci(u)
    if b.fen()!=r['position']: raise ValueError('标注历史还原不一致')
    engine.configure({'Clear Hash':None})
    info = engine.analyse(b,chess.engine.Limit(nodes=nodes),game=object())
    score = info['score'].pov(b.turn); cp=score.score(); mate=score.mate()
    if cp is None and mate is None: raise ValueError('老师没有返回评分')
    value = (1.0 if mate>0 else -1.0) if mate is not None else math.tanh(cp/300.0)
    return {**r,'value':value,'score_pov':'side_to_move','teacher_cp':cp,'teacher_mate':mate,
        'teacher_best_move':info['pv'][0].uci() if info.get('pv') else None,
        'teacher_depth':info.get('depth'),'teacher_nodes':info.get('nodes'),
        'teacher_pv':[m.uci() for m in info.get('pv',[])],
        'target_mapping':'tanh(cp/300); signed 1 for mate; finite-node teacher, not ground truth'}

def label(a):
    if not 1<=a.workers<=32 or a.nodes<1 or a.limit<0: raise ValueError('workers须1至32，nodes正数，limit非负')
    engine = a.engine or ROOT/'stockfish-windows-x86-64-universal'/'stockfish'/'stockfish-windows-x86-64-universal.exe'
    if not engine.is_file(): raise ValueError('找不到Stockfish，请用 --engine 指定可执行文件')
    paths = [a.out/(s+'_unlabeled.jsonl') for s in ['train','val']]
    manifest = {'engine_sha256':digest(engine),'nodes':a.nodes,'data_sha256':[digest(p) for p in paths],
        'mapping':'tanh(cp/300);mate=sign','version':1}
    with chess.engine.SimpleEngine.popen_uci(str(engine.resolve())) as probe:
        probe.configure({'Threads':1,'Hash':128,'UCI_LimitStrength':False,'Skill Level':20,'Clear Hash':None})
        teacher_name=probe.id
    db = sqlite3.connect(a.out/'labels.sqlite3');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS meta (id INTEGER PRIMARY KEY, body TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS labels (id TEXT PRIMARY KEY, split TEXT, body TEXT)')
    old = db.execute('SELECT body FROM meta WHERE id=1').fetchone()
    if old and json.loads(old[0])!=manifest: db.close();raise ValueError('节点预算、引擎或输入数据改变，拒绝混用已有标注')
    if not old: db.execute('INSERT INTO meta VALUES (1,?)',(json.dumps(manifest),));db.commit()
    done={row[0] for row in db.execute('SELECT id FROM labels')};pending=[];total=0
    for p in paths:
        with p.open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);total+=1
                if r['id'] not in done: pending.append(r)
    if a.limit: pending=pending[:a.limit]
    print('总样本',total,'已标注',len(done),'本次最多',len(pending),flush=True)
    added=0;errors=[];stopped=False;start=time.monotonic()
    # Main process owns all UCI engines. Python threads only wait for independent
    # Stockfish processes; heavy chess search still runs in separate processes.
    print('标注器 v3：主程序管理独立 Stockfish 进程',flush=True)
    engines=[];pool=None
    try:
        for _ in range(min(a.workers,len(pending))):
            e=chess.engine.SimpleEngine.popen_uci(str(engine.resolve()),timeout=120)
            engines.append(e)
            e.configure({'Threads':1,'Hash':128,'UCI_LimitStrength':False,'Skill Level':20})
        if pending: pool=ThreadPoolExecutor(max_workers=len(engines))
        for offset in range(0,len(pending),a.workers):
            futures={pool.submit(teacher,(r,a.nodes),engines[i]):r['id'] for i,r in enumerate(pending[offset:offset+a.workers])}
            for f in as_completed(futures):
                try: r=f.result()
                except Exception as e: errors.append({'id':futures[f],'error':repr(e)});continue
                db.execute('INSERT INTO labels VALUES (?,?,?)',(r['id'],r['split'],json.dumps(r,ensure_ascii=False)));db.commit();added+=1
                if added%100==0: print('累计',len(done)+added,'/',total,'本次',round(added/(time.monotonic()-start),2),'条/秒',flush=True)
            if errors: print('出现标注错误，停止本批，成功结果已保存。',flush=True);break
    except KeyboardInterrupt:
        stopped=True; print('停止派发任务；等待当前最多一批任务退出，已提交结果保留。',flush=True)
    finally:
        print('标注结果已提交数据库，正在关闭引擎进程……',flush=True)
        # Close UCI transports BEFORE joining Python threads. This also cancels
        # active analyses after Ctrl+C. No process-name-wide termination is used.
        for e in engines: e.close()
        if pool is not None: pool.shutdown(wait=True,cancel_futures=True)
        for e in engines:
            try: e.returncode.result(timeout=5)
            except Exception as exc: errors.append({'stage':'engine_shutdown','error':repr(exc)})
        print('关闭阶段完成，正在导出数据和报告……',flush=True)
        counts={}
        for s in ['train','val']:
            target=a.out/('teacher_'+s+'.jsonl');tmp=target.with_suffix('.jsonl.tmp');n=0
            with tmp.open('w',encoding='utf-8') as f:
                for (body,) in db.execute('SELECT body FROM labels WHERE split=? ORDER BY id',(s,)): f.write(body+'\n');n+=1
            tmp.replace(target);counts[s]=n
        write(a.out/'label_report.json',{'teacher':teacher_name,**manifest,'workers':a.workers,'threads_each':1,
            'hash_mb_each':128,'added':added,'labeled':counts,'total_expected':total,
            'complete':sum(counts.values())==total,'interrupted':stopped,'errors':errors,'seconds':time.monotonic()-start,
            'controller_version':3,'parallelism':'independent Stockfish processes, main-process thread orchestration'})
        db.close()
    print('已保存标注与报告：',a.out,'；重新运行同一命令会跳过已完成样本。')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','label'])
    ap.add_argument('--source',type=Path);ap.add_argument('--out',type=Path,default=ROOT/'collection_teacher_data')
    ap.add_argument('--history-root',type=Path,default=ROOT)
    ap.add_argument('--engine',type=Path);ap.add_argument('--workers',type=int,default=16)
    ap.add_argument('--nodes',type=int,default=300000);ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args();(prepare if a.action=='prepare' else label)(a)

if __name__=='__main__': main()
