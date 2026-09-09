"""Download three-piece DTM teachers and prepare symmetry-separated examples."""
import argparse,json,hashlib,random,math,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'endgame_course_data'

def write(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)

def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'chess-endgame-course'}),timeout=60).read()

def download():
    OUT.mkdir(exist_ok=True)
    # Resolve and record a repository commit, then fetch immutable file versions.
    commit=json.loads(fetch('https://api.github.com/repos/niklasf/python-chess/commits/master'))['sha']
    base=f'https://raw.githubusercontent.com/niklasf/python-chess/{commit}/data/gaviota/'
    meta=json.loads(fetch(f'https://api.github.com/repos/niklasf/python-chess/contents/data/gaviota?ref={commit}'))
    manifest={'commit':commit,'files':{}}
    folder=OUT/'gaviota';folder.mkdir(exist_ok=True)
    for name in ['krk.gtb.cp4','kqk.gtb.cp4','SOURCE.txt','MD5SUMS.txt']:
        entry=next(x for x in meta if x['name']==name)
        p=folder/name;data=p.read_bytes() if p.exists() else fetch(base+name)
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if blob!=entry['sha']:
            data=fetch(base+name)
            blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if blob!=entry['sha']:raise ValueError('下载校验失败：'+name)
        p.write_bytes(data);manifest['files'][name]=hashlib.sha256(data).hexdigest()
        print('已校验：',name,len(data),'字节',flush=True)
    write(OUT/'download_report.json',manifest)

def canonical(b):
    import chess
    keys=[]
    for base in [b,b.mirror()]:
        for diagonal in [False,True]:
            a=base.transform(chess.flip_diagonal) if diagonal else base
            for horizontal in [False,True]:
                c=a.transform(chess.flip_horizontal) if horizontal else a
                for vertical in [False,True]:
                    d=c.transform(chess.flip_vertical) if vertical else c
                    keys.append(d.board_fen()+' '+('w' if d.turn else 'b'))
    return min(keys)

def target(dtm,clock,terminal=False):
    # Applies only to pawnless KRK/KQK. Capturing the major piece draws.
    if terminal:return -1.0
    if not dtm or abs(dtm)>100-clock:return 0.0
    return (1 if dtm>0 else -1)*max(.35,.98-.008*abs(dtm))

def prepare(count):
    import chess,chess.gaviota
    if (OUT/'train.jsonl').exists() or (OUT/'val.jsonl').exists():
        raise ValueError('数据已存在，避免意外重划分；请先保留现有数据目录后另建实验')
    manifest=json.loads((OUT/'download_report.json').read_text())
    for n,h in manifest['files'].items():
        if hashlib.sha256((OUT/'gaviota'/n).read_bytes()).hexdigest()!=h:raise ValueError('残局库文件变化')
    blocked=set()
    for fen in json.loads((ROOT/'endgame_holdout_fens.json').read_text()):
        b=chess.Board(fen)
        if len(b.piece_map())==3:blocked.add(canonical(b))
    # Old validation remains a regression holdout.
    for name in ['value_val.jsonl','value_train.jsonl','value_expansion_labels/expanded_value_train.jsonl','value_expansion_labels/expanded_value_val.jsonl','move_choices_train.jsonl']:
        with (ROOT/name).open(encoding='utf-8-sig') as f:
            for line in f:
                if not line.strip():continue
                b=chess.Board(json.loads(line)['position'])
                if len(b.piece_map())==3:blocked.add(canonical(b))
    rng=random.Random(20260908);rows={'train':[],'val':[],'test':[]};seen=set();groups={};attempts=0
    with chess.gaviota.open_tablebase(str(OUT/'gaviota')) as tb:
        while sum(map(len,rows.values()))<count:
            attempts+=1
            if attempts>count*100:raise RuntimeError('可用样本不足')
            wk,bk,major=rng.sample(range(64),3);color=rng.choice([True,False]);piece=rng.choice([chess.ROOK,chess.QUEEN])
            b=chess.Board(None);b.set_piece_at(wk,chess.Piece(chess.KING,True));b.set_piece_at(bk,chess.Piece(chess.KING,False));b.set_piece_at(major,chess.Piece(piece,color));b.turn=rng.choice([True,False])
            if not b.is_valid() or b.is_game_over():continue
            key=canonical(b)
            if key in blocked:continue
            clock=rng.choice([0,0,0,10,20,40,60,80,90,95,98,99]);b.halfmove_clock=clock
            ident=(key,clock)
            if ident in seen:continue
            dtm=tb.probe_dtm(b)
            splitnum=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%100
            split='test' if splitnum<10 else 'val' if splitnum<20 else 'train'
            seen.add(ident);groups[key]=split
            rows[split].append({'position':b.fen(),'value':target(dtm,clock),'dtm':dtm,
                                'group':key,'score_pov':'side_to_move','piece':chess.piece_symbol(piece)})
            n=sum(map(len,rows.values()))
            if n%2000==0:print('已生成',n,'/',count,flush=True)
    for split,rs in rows.items():
        tmp=OUT/(split+'.jsonl.tmp')
        with tmp.open('w',encoding='utf-8') as f:
            for r in rs:f.write(json.dumps(r)+'\n')
        tmp.replace(OUT/(split+'.jsonl'))
    report={'counts':{s:len(r) for s,r in rows.items()},'groups':len(groups),'excluded_groups':len(blocked),
            'target':'sign(DTM)*max(.35,.98-.008*abs(DTM)); draw or mate beyond 50-move budget -> 0',
            'scope':'KRK/KQK fresh FEN only; no repetition history; symmetry and clock variants grouped; final test reserved',
            'distribution':{s:{'draw':sum(r['value']==0 for r in rs),'positive':sum(r['value']>0 for r in rs),'negative':sum(r['value']<0 for r in rs)} for s,rs in rows.items()}}
    write(OUT/'preparation_report.json',report);print(report,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['download','prepare']);ap.add_argument('--samples',type=int,default=40000);args=ap.parse_args()
    if args.action=='download':download()
    else:
        if args.samples<1000:ap.error('至少1000样本')
        prepare(args.samples)
