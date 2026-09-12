"""Incremental repetition counts scoped to the current search path.
No history-dependent transposition table and no change in search ordering.
"""
from collections import Counter
from contextlib import contextmanager
from neural_fast import FastSearchEngine

def repetition_key(b):
    return (b.pawns,b.knights,b.bishops,b.rooks,b.queens,b.kings,
            b.occupied_co[0],b.occupied_co[1],b.turn,b.clean_castling_rights(),
            b.ep_square if b.has_legal_en_passant() else None)

class NeuralSearchV2(FastSearchEngine):
    def __init__(self,*args,verify_claims=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.verify_claims=verify_claims

    def choose(self,board,preferred=None):
        if board.chess960 or type(board).__name__!='Board':
            raise ValueError('当前加速版只支持标准国际象棋Board')
        history=board.copy(stack=True)
        self.repetitions=Counter()
        while True:
            self.repetitions[repetition_key(history)]+=1
            if not history.move_stack:break
            history.pop()
        self.twice=sum(v>=2 for v in self.repetitions.values())
        self.active_length=len(board.move_stack)
        return super().choose(board,preferred)

    @contextmanager
    def path(self,board):
        length=len(board.move_stack)
        added=length!=self.active_length
        previous=self.active_length
        if added:
            if length!=previous+1:raise RuntimeError('搜索路径深度不同步')
            k=repetition_key(board)
            self.repetitions[k]+=1
            if self.repetitions[k]==2:self.twice+=1
            self.active_length=length
        try:yield
        finally:
            if added:
                if self.repetitions[k]==2:self.twice-=1
                self.repetitions[k]-=1
                if not self.repetitions[k]:del self.repetitions[k]
                self.active_length=previous

    def negamax(self,board,depth,alpha,beta,ply):
        with self.path(board):
            return super().negamax(board,depth,alpha,beta,ply)

    def quiesce(self,board,alpha,beta,remaining,ply):
        with self.path(board):
            return super().quiesce(board,alpha,beta,remaining,ply)

    def claim_available(self,board):
        if not self.claim_draw:return False
        # Most nodes have no position occurring twice anywhere in their history.
        result=False
        if board.halfmove_clock>=99 and board.can_claim_fifty_moves():
            result=True
        elif self.repetitions[repetition_key(board)]>=3:
            result=True
        elif self.twice:
            for move in board.legal_moves:
                board.push(move)
                try:
                    found=self.repetitions[repetition_key(board)]>=2
                finally:board.pop()
                if found:
                    result=True
                    break
        if self.verify_claims and result!=board.can_claim_draw():
            raise AssertionError('和棋判断不一致：'+board.fen())
        return result
