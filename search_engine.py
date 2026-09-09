"""Negamax with bounded quiescence and optional neural evaluation."""
import math
import time

import chess

VALUES = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 0}
MATE = 1000.0


def material(board):
    return sum(v * (len(board.pieces(p, board.turn)) -
                    len(board.pieces(p, not board.turn)))
               for p, v in VALUES.items())


class SearchTimeout(Exception):
    pass


class SearchEngine:
    def __init__(self, evaluator=None, seconds=5.0, max_depth=3, qdepth=6, claim_draw=False):
        self.evaluator = evaluator or (lambda b: math.tanh(material(b) / 300))
        self.seconds = seconds
        self.max_depth = max_depth
        self.qdepth = qdepth
        # Opt-in: choose() may return None to claim a draw.
        # Legacy play scripts keep receiving a legal move.
        self.claim_draw = claim_draw

    def claim_available(self, board):
        # History-dependent: never cache this by FEN.
        return self.claim_draw and board.can_claim_draw()


    def tick(self):
        self.nodes += 1
        if time.perf_counter() >= self.deadline:
            raise SearchTimeout()

    def terminal(self, board, ply):
        outcome = board.outcome()
        if outcome is None:
            return None
        if outcome.winner is None:
            return 0.0
        return -MATE + ply * 0.01

    def static(self, board):
        key = board.fen()
        if key not in self.cache:
            value = float(self.evaluator(board))
            if not math.isfinite(value):
                raise ValueError("局面评价不是有限数值")
            self.cache[key] = max(-1.0, min(1.0, value))
        return self.cache[key]

    def ordered(self, board, moves):
        def priority(move):
            victim = board.piece_at(move.to_square)
            capture = (10 * (VALUES[victim.piece_type] if victim else 100)
                       - VALUES[board.piece_type_at(move.from_square)]) if board.is_capture(move) else 0
            return capture + (VALUES[move.promotion] if move.promotion else 0)
        return sorted(moves, key=priority, reverse=True)

    def quiesce(self, board, alpha, beta, remaining, ply):
        self.tick()
        terminal = self.terminal(board, ply)
        if terminal is not None:
            return terminal
        # At the absolute safety cap use an estimate, even in check.
        # Otherwise checks always search every legal evasion; no stand-pat.
        floor = 0.0 if self.claim_available(board) else -float('inf')
        if floor >= beta:
            return floor
        alpha = max(alpha, floor)
        if ply >= 32:
            return max(floor, self.static(board))
        in_check = board.is_check()
        if not in_check:
            best = max(floor, self.static(board))
            if remaining <= 0:
                return best
            if best >= beta:
                return best
            alpha = max(alpha, best)
            moves = [m for m in board.legal_moves
                     if board.is_capture(m) or m.promotion]
        else:
            best = floor
            moves = list(board.legal_moves)
        for move in self.ordered(board, moves):
            board.push(move)
            try:
                score = -self.quiesce(board, -beta, -alpha, remaining - 1, ply + 1)
            finally:
                board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def negamax(self, board, depth, alpha, beta, ply):
        self.tick()
        terminal = self.terminal(board, ply)
        if terminal is not None:
            return terminal
        if depth <= 0:
            return self.quiesce(board, alpha, beta, self.qdepth, ply)
        best = 0.0 if self.claim_available(board) else -float('inf')
        if best >= beta:
            return best
        alpha = max(alpha, best)
        for move in self.ordered(board, list(board.legal_moves)):
            board.push(move)
            try:
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def choose(self, board, preferred=None):
        if board.is_game_over():
            raise ValueError('对局已结束')
        moves = list(board.legal_moves)
        if preferred:
            order = {m: i for i, m in enumerate(preferred)}
            moves.sort(key=lambda m: order.get(m, len(order)))
        # Exhaustive mate-in-one check does not depend on policy ranking.
        for move in moves:
            board.push(move)
            try:
                mate = board.is_checkmate()
            finally:
                board.pop()
            if mate:
                return move, True, {'depth': 0, 'nodes': 0, 'seconds': 0.0}
        start = time.perf_counter()
        self.deadline = start + self.seconds
        self.nodes = 0
        self.cache = {}
        can_claim = self.claim_available(board)
        best_move = None if can_claim else moves[0]
        committed_score = 0.0 if can_claim else None
        completed = 0
        try:
            for depth in range(1, self.max_depth + 1):
                iteration_best = None if can_claim else moves[0]
                best_score = 0.0 if can_claim else -float('inf')
                alpha = best_score
                for move in moves:
                    self.tick()
                    board.push(move)
                    try:
                        score = -self.negamax(board, depth - 1, -float('inf'), -alpha, 1)
                    finally:
                        board.pop()
                    if score > best_score:
                        best_score, iteration_best = score, move
                    alpha = max(alpha, score)
                # Commit only after every root move has been considered.
                best_move, completed = iteration_best, depth
                committed_score = best_score
                if best_move is not None:
                    moves.remove(best_move)
                    moves.insert(0, best_move)
        except SearchTimeout:
            pass
        return best_move, False, {
            'depth': completed, 'nodes': self.nodes,
            'action': 'claim_draw' if best_move is None else 'move',
            'score': committed_score,
            'seconds': time.perf_counter() - start,
        }


def neural_evaluator(model, device):
    import torch
    from board_encoder import encode_board
    def evaluate(board):
        with torch.inference_mode():
            x = encode_board(board).unsqueeze(0).to(device)
            m = torch.tensor([material(board)], dtype=torch.float32, device=device)
            value, _ = model(x, m)
            return value.item()
    return evaluate


def policy_order(board, model, device):
    import torch
    from board_encoder import encode_board
    from move_encoder import encode_move
    with torch.inference_mode():
        scores = model(encode_board(board).unsqueeze(0).to(device))[0].cpu()
    return sorted(board.legal_moves, key=lambda m: scores[encode_move(m)].item(), reverse=True)
