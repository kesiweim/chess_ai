"""Exact bounded AND/OR mate proof on top of the unchanged Syzygy wrapper.

No heuristic evaluation, transposition score cache, or selective pruning is
used in the proof. A timeout is UNKNOWN and retains the original TB choice.
"""
import time
import chess


class ProofTimeout(Exception):
    pass


def prove_mate(board, seconds=0.25, max_plies=5, preferred=None, max_nodes=50000):
    start = time.perf_counter()
    deadline = start + max(0.0, seconds)
    attacker = board.turn
    b = board.copy(stack=True)
    nodes = 0
    completed = 0
    order = {m: i for i, m in enumerate(preferred or [])}

    def tick():
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes or time.perf_counter() >= deadline:
            raise ProofTimeout()

    def moves():
        # Ordering only: all legal defender moves are still considered.
        legal = list(b.legal_moves)
        def key(m):
            tick()
            return (b.gives_check(m), bool(m.promotion), b.is_capture(m), -order.get(m, 9999))
        return sorted(legal, key=key, reverse=True)

    def solve(left):
        tick()
        # Checkmate takes precedence over automatic move-count draws.
        if b.is_checkmate():
            return 0 if b.turn != attacker else None
        if b.is_game_over():
            return None
        if b.turn != attacker and b.can_claim_draw():
            return None
        if left == 0:
            return None
        attack_turn = b.turn == attacker
        worst = 0
        for m in moves():
            b.push(m)
            try:
                distance = solve(left - 1)
            finally:
                b.pop()
            if attack_turn and distance is not None:
                return distance + 1
            if not attack_turn:
                if distance is None:
                    return None
                worst = max(worst, distance + 1)
        return None if attack_turn else worst

    status = 'not_found_within_limit'
    chosen = None
    bound = None
    try:
        if b.is_game_over():
            status = 'terminal'
        else:
            for depth in range(1, max_plies + 1, 2):
                for m in moves():
                    b.push(m)
                    try:
                        distance = solve(depth - 1)
                    finally:
                        b.pop()
                    if distance is not None:
                        tick()  # Never accept a proof completed after the deadline.
                        chosen, bound = m, distance + 1
                        break
                if chosen is not None:
                    status = 'proven'
                    break
                completed = depth
    except ProofTimeout:
        status = 'budget_exhausted'
        chosen, bound = None, None
    return chosen, {'mate_status': status, 'mate_bound_plies': bound,
                    'mate_completed_depth': completed, 'mate_work_units': nodes,
                    'mate_seconds': time.perf_counter() - start}


def make_engine(evaluator, directory, seconds=5, depth=5, mate_seconds=0.25, mate_plies=5):
    from syzygy_v4 import make_engine as original_make
    base = original_make(evaluator, directory, seconds, depth)

    class Candidate:
        def __getattr__(self, name):
            return getattr(base, name)

        def choose(self, board, preferred=None):
            start = time.perf_counter()
            result = base.choose(board, preferred)
            move, immediate, original_stats = result
            stats = dict(original_stats)
            stats['variant'] = 'tb_mate_candidate'
            # Root tablebase must explicitly certify a rule-aware win.
            if stats.get('source') == 'syzygy' and stats.get('score') == 2.0 and move is not None and not immediate:
                budget = min(mate_seconds, max(0.0, seconds - (time.perf_counter() - start)))
                proven, details = prove_mate(board, budget, mate_plies, preferred)
                stats.update(details)
                stats['original_tb_move'] = move.uci()
                if proven is not None:
                    stats.update(source='syzygy_mate', action='move', selected_by='forced_mate_proof')
                    # These belong to the original selected move, not the proof move.
                    stats.pop('selected_dtz', None)
                    stats.pop('selected_wdl', None)
                    move = proven
                    immediate = details['mate_bound_plies'] == 1
            stats['seconds'] = time.perf_counter() - start
            return move, immediate, stats

        def close(self):
            base.close()

    return Candidate()
