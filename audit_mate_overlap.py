"""Find exact FEN overlaps in local historical JSONL files.
This is an inventory, not proof of zero prior exposure or game-level separation.
"""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def key(fen):
    if not isinstance(fen,str):return None
    p=fen.split()
    return ' '.join(p[:4]) if len(p)>=4 and p[0].count('/')==7 else None

def fens(obj):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if isinstance(v,str):
                x=key(v)
                if x:yield x
            elif isinstance(v,(dict,list)):yield from fens(v)
    elif isinstance(obj,list):
        for v in obj:
            if isinstance(v,str):
                x=key(v)
                if x:yield x
            else:yield from fens(v)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path);args=ap.parse_args()
    if args.source is None:
        fs=sorted((ROOT/'mate_course_candidates').glob('*/forced_verification/verification_report.json'))
        if not fs:ap.error('没有找到核验结果')
        args.source=fs[-1].parent
    candidates={}
    for split in ['train','val','test']:
        with (args.source/f'{split}_verified.jsonl').open(encoding='utf-8-sig') as f:
            for line in f:
                if line.strip():
                    r=json.loads(line);candidates.setdefault(key(r['position']),[]).append(r)
    hits=set();files=[];errors=[]
    paths=sorted(p for p in ROOT.rglob('*.jsonl') if not any(x in p.relative_to(ROOT).parts for x in
        ['.venv','.git','mate_course_candidates']))
    for i,p in enumerate(paths):
        matched=set();rows=parsed=bad=0
        try:
            with p.open(encoding='utf-8-sig') as f:
                for line in f:
                    if not line.strip():continue
                    rows+=1
                    try:obj=json.loads(line)
                    except ValueError:bad+=1;continue
                    ks=set(fens(obj))
                    if ks:parsed+=1
                    matched.update(ks & candidates.keys())
        except (OSError,UnicodeError) as e:errors.append({'file':str(p.relative_to(ROOT)),'error':str(e)})
        hits.update(matched)
        entry={'file':str(p.relative_to(ROOT)),'rows':rows,'rows_with_fen':parsed,'invalid_json':bad,'matched_roots':len(matched)}
        files.append(entry);print(i+1,'/',len(paths),entry,flush=True)
    counts={s:{t:{'total':0,'found_in_historical_jsonl':0,'not_found':0} for t in ['mateIn1','mateIn2','mateIn3']} for s in ['train','val','test']}
    for k,rs in candidates.items():
        for r in rs:
            c=counts[r['split']][r['theme']];c['total']+=1;c['found_in_historical_jsonl' if k in hits else 'not_found']+=1
    report={'counts':counts,'files':files,'errors':errors,'source':str(args.source.resolve()),
            'scope':'Conservative overlap with ALL scanned historical JSONL, including past validation and tests; not all matches imply training exposure. Only explicit FEN fields; no PGN replay, CSV, SQLite or missing/deleted data. Not-found is not proof of clean holdout.'}
    out=args.source/'overlap_report.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('发我：',out,flush=True)

if __name__=='__main__':main()
