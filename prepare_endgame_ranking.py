import json,random,hashlib,argparse
from pathlib import Path
import chess,chess.gaviota
from endgame_course import canonical
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'endgame_ranking_data'

def read(p):
    with p.open(encoding='utf-8-sig') as f:return [json.loads(x) for x in f if x.strip()]

def choices(b,tb):
    rows=[]
    for m in list(b.legal_moves):
        b.push(m)
        try:
            out=b.outcome();dtm=tb.probe_dtm(b) if not out else 0
            if out:
                win=out.winner== (not b.turn);distance=1 if win else None
                terminal_score=1.0 if win else 0.0
            else:
                win=dtm<0 and abs(dtm)<=100-b.halfmove_clock and not b.can_claim_draw()
                distance=1+abs(dtm) if win else None;terminal_score=None
            rows.append({'move':m.uci(),'position':b.fen(),'group':canonical(b),
                         'win':win,'distance':distance,'terminal_score':terminal_score})
        finally:b.pop()
    return rows

def main():
    from collections import Counter
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',type=int,default=1200)
    ap.add_argument('--val',type=int,default=200)
    a=ap.parse_args()
    if a.train<100 or a.val<20:ap.error('至少100训练父局面、20验证父局面')
    if OUT.exists():ap.error('endgame_ranking_data已存在，请先将其重命名保留，再运行')
    tbdir=ROOT/'endgame_course_data'/'gaviota'
    manifest=json.loads((tbdir.parent/'download_report.json').read_text(encoding='utf-8-sig'))
    for name in ['krk.gtb.cp4','kqk.gtb.cp4']:
        if hashlib.sha256((tbdir/name).read_bytes()).hexdigest()!=manifest['files'][name]:
            raise ValueError('残局库校验失败：'+name)
    blocked=set()
    for fen in json.loads((ROOT/'ranking_holdout_fens.json').read_text(encoding='utf-8-sig')):
        b=chess.Board(fen)
        if len(b.piece_map())<=3:blocked.add(canonical(b))
    # Only read test positions to exclude overlap, never use test labels.
    for r in read(ROOT/'endgame_course_data'/'test.jsonl'):
        blocked.add(canonical(chess.Board(r['position'])))
    rng=random.Random(83);output={};val_keys=set();discard={}
    with chess.gaviota.open_tablebase(str(tbdir)) as tb:
        # Reserve validation neighborhoods BEFORE selecting training examples.
        for split,limit in [('val',a.val),('train',a.train)]:
            selected=[];seen=set();counts=Counter()
            source=read(ROOT/'endgame_course_data'/f'{split}.jsonl');rng.shuffle(source)
            attempts=0
            while len(selected)<limit and attempts<max(200000,limit*1000):
                attempts+=1
                if source:
                    r=source.pop();b=chess.Board(r['position'])
                    counts['existing_candidates']+=1
                else:
                    # Supplement the finite old pool with legal KRK positions.
                    wk,bk,rook=rng.sample(range(64),3);color=rng.choice([True,False])
                    b=chess.Board(None)
                    b.set_piece_at(wk,chess.Piece(chess.KING,True))
                    b.set_piece_at(bk,chess.Piece(chess.KING,False))
                    b.set_piece_at(rook,chess.Piece(chess.ROOK,color));b.turn=color
                    b.halfmove_clock=rng.choice([0,10,20,40,60])
                    counts['generated_candidates']+=1
                if not b.is_valid() or b.is_game_over() or len(b.piece_map())!=3 or b.halfmove_clock>60 or not b.pieces(chess.ROOK,b.turn):
                    counts['ineligible']+=1;continue
                g=canonical(b)
                # Same parent-group partition as the original endgame course.
                bucket=int(hashlib.sha256(g.encode()).hexdigest()[:8],16)%100
                if not (10<=bucket<20 if split=='val' else bucket>=20):
                    counts['other_parent_split']+=1;continue
                if g in seen:counts['duplicate_parent']+=1;continue
                seen.add(g)
                if g in blocked or (split=='train' and g in val_keys):
                    counts['parent_overlap']+=1;continue
                cs=choices(b,tb);wins=[c for c in cs if c['win']]
                if not wins:counts['no_win']+=1;continue
                best=min(c['distance'] for c in wins)
                good=[i for i,c in enumerate(cs) if c['win'] and c['distance']==best and c['terminal_score'] is None]
                bad=[i for i,c in enumerate(cs) if c['terminal_score'] is None and (not c['win'] or c['distance']>best)]
                if not good or not bad:counts['no_comparison']+=1;continue
                keys={g}|{c['group'] for c in cs}
                if keys&blocked or (split=='train' and keys&val_keys):
                    counts['child_overlap']+=1;continue
                selected.append({'position':b.fen(),'group':g,'choices':cs,'good':good,'bad':bad,'best_distance':best})
                if split=='val':val_keys.update(keys)
                if len(selected)%50==0:print(split,len(selected),'/',limit,flush=True)
            counts['attempts']=attempts;discard[split]=dict(counts)
            if len(selected)!=limit:
                report={'failed_split':split,'requested':limit,'selected':len(selected),'filters':discard}
                (ROOT/'endgame_ranking_preparation_failed.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                raise ValueError(f'{split}: 需要{limit}个，实际{len(selected)}个；详细原因见endgame_ranking_preparation_failed.json')
            output[split]=selected
    keys_by_split={s:{k for r in rs for k in [r['group']]+[c['group'] for c in r['choices']]} for s,rs in output.items()}
    if keys_by_split['train']&keys_by_split['val']:raise AssertionError('跨组重合')
    if (keys_by_split['train']|keys_by_split['val'])&blocked:raise AssertionError('保留局面重合')
    OUT.mkdir()
    for split,rs in output.items():
        with (OUT/f'{split}.jsonl').open('w',encoding='utf-8') as f:
            for r in rs:f.write(json.dumps(r)+'\n')
    report={'version':2,'parents':{s:len(rs) for s,rs in output.items()},
        'children':{s:sum(len(r['choices']) for r in rs) for s,rs in output.items()},
        'filters':discard,'cross_split_overlap':0,'holdout_overlap':0,
        'scope':'Validation neighborhoods reserved first; random KRK supplementation; same original parent hash split. New train/val parent and child canonical groups disjoint. Historical test roots and diagnosis holdouts excluded. Prior full-model exposure not exhaustively audited. No repetition history.'}
    (OUT/'preparation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
