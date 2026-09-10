"""Conservative root DTZ selection, plus WDL at zero-clock search nodes.

Rounded DTZ values at the fifty-move boundary are not treated as proofs.
Missing tables and ambiguous boundary cases explicitly fall back to v4.
This is DTZ, NOT a shortest-mate oracle. No online requests in play.
"""
import time
from pathlib import Path
import chess
import chess.syzygy


class LocalSyzygy:
    def __init__(self, directory):
        self.table = chess.syzygy.open_tablebase(str(directory), max_fds=64)
        self.reason = ''

    def close(self):
        self.table.close()

    def select(self, board, preferred=None):
        self.reason = ''
        if board.chess960 or type(board) is not chess.Board or not board.is_valid():
            self.reason = '非标准或无效局面'
            return None
        if len(board.piece_map()) > 6 or board.castling_rights:
            self.reason = '子数超过6或仍有易位权'
            return None
        if board.is_game_over():
            self.reason = '对局已结束'
            return None
        # Work on a private full-history board, including repetition context.
        b = board.copy(stack=True)
        start = time.perf_counter()
        moves = list(b.legal_moves)
        order = {m: i for i,m in enumerate(preferred or moves)}
        rows = []
        try:
            root_wdl = self.table.probe_wdl(b)
            root_dtz = self.table.probe_dtz(b)
            for move in moves:
                zeroing = b.is_zeroing(move)
                b.push(move)
                try:
                    # Mate takes precedence over any fifty-move counter.
                    if b.is_checkmate():
                        return move, True, {'depth':0,'nodes':0,'seconds':time.perf_counter()-start,
                            'source':'syzygy','tb_wdl':2,'tb_dtz':1,'action':'move','score':999.0}
                    if b.is_game_over():
                        rows.append((0,0,move,0))
                        continue
                    wdl = -self.table.probe_wdl(b)
                    if zeroing:
                        # Capture/pawn move resets clock: WDL is sufficient here.
                        dtz = {2:1,1:101,0:0,-1:-101,-2:-1}[wdl]
                    else:
                        d = -self.table.probe_dtz(b)
                        dtz = d+1 if d>0 else d-1 if d<0 else 0
                    if abs(wdl)==1 or wdl==0:
                        rank = 0  # Draw with the fifty-move rule and proper claims.
                    elif zeroing:
                        rank = 1 if wdl>0 else -1
                    elif abs(dtz)+board.halfmove_clock <= 99:
                        rank = 1 if wdl>0 else -1
                    else:
                        # Rounding near the boundary can change result. Never infer a draw.
                        rank = None
                    if b.can_claim_draw():
                        # The opponent may claim only when it benefits them.
                        if rank is not None: rank = min(rank,0)
                    rows.append((rank,dtz,move,wdl))
                finally:
                    b.pop()
        except KeyError as e:
            self.reason = '缺表：'+str(e)
            return None
        wins = [r for r in rows if r[0]==1]
        if wins:
            chosen = min(wins,key=lambda r:(r[1],order.get(r[2],9999),r[2].uci()))
        else:
            if any(r[0] is None for r in rows):
                self.reason = '接近五十步边界，DTZ取整不能安全裁决'
                return None
            if b.can_claim_draw():
                return None, False, {'depth':0,'nodes':0,'seconds':time.perf_counter()-start,
                    'source':'syzygy','tb_wdl':root_wdl,'tb_dtz':root_dtz,
                    'action':'claim_draw','score':0.0}
            draws = [r for r in rows if r[0]==0]
            if draws:
                chosen = min(draws,key=lambda r:(order.get(r[2],9999),r[2].uci()))
            else:
                # Lose as slowly as possible by DTZ, without pretending this forces a draw.
                chosen = min(rows,key=lambda r:(r[1],order.get(r[2],9999),r[2].uci()))
        rank,dtz,move,wdl = chosen
        return move, False, {'depth':0,'nodes':0,'seconds':time.perf_counter()-start,
            'source':'syzygy','tb_wdl':root_wdl,'tb_dtz':root_dtz,
            'selected_dtz':dtz,'selected_wdl':wdl,'action':'move','score':float(rank)*2}


def make_engine(evaluator, directory, seconds=5, depth=5):
    from neural_search_v4 import NeuralSearchV4

    class SyzygySearchV4(NeuralSearchV4):
        def __init__(self):
            super().__init__(evaluator, seconds, depth, claim_draw=True)
            self.syzygy = LocalSyzygy(directory)
            self.tb_hits = 0

        def terminal(self, board, ply):
            result = super().terminal(board, ply)
            if result is not None: return result
            # Only after a zeroing move: raw WDL otherwise ignores elapsed fifty-move clock.
            if board.halfmove_clock==0 and not board.castling_rights and chess.popcount(board.occupied)<=6:
                try:
                    wdl = self.syzygy.table.probe_wdl(board)
                except KeyError:
                    return None
                self.tb_hits += 1
                return 2.0 if wdl==2 else -2.0 if wdl==-2 else 0.0
            return None

        def choose(self, board, preferred=None):
            self.tb_hits = 0
            selected = self.syzygy.select(board,preferred)
            if selected is not None: return selected
            result = super().choose(board,preferred)
            result[2].update(source='v4_search',tb_search_hits=self.tb_hits,tb_fallback=self.syzygy.reason)
            return result

        def close(self): self.syzygy.close()

    return SyzygySearchV4()


def verify_install(directory):
    """Require the downloader's verified manifest; detect later size/mtime changes."""
    import json
    from download_syzygy import manifest, digest, ROOT
    data = manifest()
    directory = Path(directory)
    report_path = directory/'download_report.json'
    if not report_path.is_file(): raise ValueError('请先运行 download_syzygy.py 完成下载校验')
    report = json.loads(report_path.read_text(encoding='utf-8-sig'))
    if not report.get('complete') or report.get('manifest_sha256')!=digest(ROOT/'syzygy_manifest.json'):
        raise ValueError('下载未完成或清单变化，请运行 download_syzygy.py --verify-only')
    for row in data['files']:
        p = directory/row['name']
        old = report['files'].get(row['name'],{})
        if (not p.is_file() or p.stat().st_size!=row['bytes'] or old.get('sha256')!=row['sha256']
                or p.stat().st_mtime_ns!=old.get('mtime_ns')):
            raise ValueError('文件变化，请重新校验：'+str(p))
    return data
