"""Extract candidate mate puzzles. Line legality is NOT a forced-mate proof."""
import argparse,csv,io,json,random,hashlib
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parent

def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    import chess,zstandard
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=ROOT/'lichess_db_puzzle.csv.zst');ap.add_argument('--per-theme',type=int,default=3000);args=ap.parse_args()
    if not args.source.is_file():ap.error('找不到题库，请用 --source 指定 csv.zst 路径')
    if args.per_theme<1:ap.error('per-theme须为正数')
    tags=['mateIn1','mateIn2','mateIn3'];rng=random.Random(42)
    pools={k:[] for k in tags};seen={k:0 for k in tags};total=0
    print('扫描题库；先均匀抽样，再核验所选棋谱。',flush=True)
    with args.source.open('rb') as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw) as reader:
            with io.TextIOWrapper(reader,encoding='utf-8-sig',newline='') as text:
                for row in csv.DictReader(text):
                    total+=1
                    theme=next((k for k in tags if k in row['Themes'].split()),None)
                    if theme:
                        seen[theme]+=1
                        if len(pools[theme])<args.per_theme:pools[theme].append(row)
                        else:
                            j=rng.randrange(seen[theme])
                            if j<args.per_theme:pools[theme][j]=row
                    if total%500000==0:print('读取',total,'；命中',seen,flush=True)
    folder=ROOT/'mate_course_candidates'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');folder.mkdir(parents=True)
    good=[];bad=[]
    for tag,rows in pools.items():
        for r in rows:
            try:
                b=chess.Board(r['FEN']);moves=r['Moves'].split()
                if not b.is_valid() or len(moves)<2:raise ValueError('无效棋盘或缺少走法')
                first=chess.Move.from_uci(moves[0])
                if first not in b.legal_moves:raise ValueError('题目前一手非法')
                b.push(first) # Lichess FEN precedes the opponent's setup move.
                if b.is_game_over():raise ValueError('题目起点已结束')
                fen=b.fen();attacker=b.turn
                for u in moves[1:]:
                    m=chess.Move.from_uci(u)
                    if m not in b.legal_moves:raise ValueError('答案变化包含非法走法')
                    b.push(m)
                n=int(tag[-1])
                if not b.is_checkmate() or b.turn==attacker:raise ValueError('答案未由解题方将死')
                if len(moves[1:])!=2*n-1:raise ValueError('答案长度与主题不符')
                game=r.get('GameUrl','').split('#')[0].rstrip('/')
                group=game or r['PuzzleId']
                v=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%100
                split='test' if v<10 else 'val' if v<20 else 'train'
                good.append({'puzzle_id':r['PuzzleId'],'theme':tag,'mate_moves':n,'position':fen,
                             'solution':moves[1:],'setup_position':r['FEN'],'setup_move':moves[0],
                             'game_url':game,'split':split,'verification':'legal_mating_line_only',
                             'rating':r.get('Rating'),'themes':r['Themes']})
            except (ValueError,KeyError,IndexError) as e:bad.append({'id':r.get('PuzzleId'),'error':str(e)})
    # Remove all instances when the same root appears in different game splits.
    bykey={}
    for r in good:
        k=' '.join(r['position'].split()[:4]);bykey.setdefault(k,[]).append(r)
    unique=[];conflicts=duplicates=0
    for rows in bykey.values():
        if len({r['split'] for r in rows})>1:conflicts+=len(rows);continue
        unique.append(rows[0]);duplicates+=len(rows)-1
    counts={s:{k:0 for k in tags} for s in ['train','val','test']}
    for split in counts:
        with (folder/f'{split}_candidates.jsonl').open('w',encoding='utf-8') as f:
            for r in unique:
                if r['split']==split:f.write(json.dumps(r,ensure_ascii=False)+'\n');counts[split][r['theme']]+=1
    report={'source':str(args.source.resolve()),'source_bytes':args.source.stat().st_size,
            'rows_scanned':total,'theme_hits':seen,'sampled':{k:len(v) for k,v in pools.items()},
            'invalid_lines':len(bad),'duplicates_removed':duplicates,'cross_split_rows_removed':conflicts,
            'candidate_counts':counts,'output':str(folder),'verification':'Only legal line ending in mate verified; forced mate against all defenses NOT yet verified.',
            'holdout_warning':'Prior project data overlap has NOT been audited. These are candidate pools, not clean independent evaluation sets. Do not train yet.'}
    write(folder/'preparation_report.json',report);write(folder/'rejected_lines.json',bad)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('下一步先核验强制将死及历史数据重合；现在不要直接训练。',flush=True)

if __name__=='__main__':main()
