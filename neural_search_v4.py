"""Experimental move ordering. No reduced-depth or forward-pruned moves.
Position table stores ONLY move hints, never history-dependent scores.
"""
from collections import defaultdict
from neural_search_v2 import NeuralSearchV2,repetition_key

class NeuralSearchV4(NeuralSearchV2):
    def choose(self,board,preferred=None):
        self.move_hints={}
        self.killers={}
        self.history=defaultdict(int)
        self.ordering_hits=0
        result=super().choose(board,preferred)
        result[2]['ordering_hits']=self.ordering_hits
        return result

    def main_order(self,board,moves,ply):
        hint=self.move_hints.get(repetition_key(board))
        killers=self.killers.get(ply,[])
        original=super().ordered(board,moves)
        rank={m:i for i,m in enumerate(original)}
        def priority(m):
            if m==hint:
                return (4,0)
            if board.is_capture(m) or m.promotion:return (3,-rank[m])
            if m in killers:return (2,-killers.index(m))
            return (1,self.history[(board.turn,m.from_square,m.to_square)])
        if hint in moves:self.ordering_hits+=1
        return sorted(original,key=priority,reverse=True)

    def negamax(self,board,depth,alpha,beta,ply):
        with self.path(board):
            self.tick()
            terminal=self.terminal(board,ply)
            if terminal is not None:return terminal
            if depth<=0:
                # Quiescence and its move order remain exactly as v3.
                return self.quiesce(board,alpha,beta,self.qdepth,ply)
            best=0.0 if self.claim_available(board) else -float('inf')
            if best>=beta:return best
            alpha=max(alpha,best)
            best_move=None
            for move in self.main_order(board,list(board.legal_moves),ply):
                quiet=not board.is_capture(move) and not move.promotion
                color=board.turn
                board.push(move)
                try:score=-self.negamax(board,depth-1,-beta,-alpha,ply+1)
                finally:board.pop()
                if score>best:best,best_move=score,move
                alpha=max(alpha,score)
                if alpha>=beta:
                    if quiet:
                        killers=self.killers.setdefault(ply,[])
                        if move in killers:killers.remove(move)
                        killers.insert(0,move);del killers[2:]
                        k=(color,move.from_square,move.to_square)
                        self.history[k]=min(1000000,self.history[k]+depth*depth)
                    break
            if best_move is not None:
                # Reset each choose() and bounded; stale hints only change order.
                if len(self.move_hints)<100000:
                    self.move_hints[repetition_key(board)]=best_move
            return best
