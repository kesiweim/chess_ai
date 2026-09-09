"""CPU neural evaluator with reusable inputs; original search semantics retained.
Each instance belongs to one search worker; not safe for concurrent calls.
"""
import math
import numpy as np
import torch
from search_engine import SearchEngine as OriginalSearchEngine

VALUES=(0,100,320,330,500,900,0)

class FastEvaluator:
    def __init__(self,model):
        self.model=model.eval()
        if next(model.parameters()).device.type!='cpu':
            raise ValueError('FastEvaluator requires a CPU model')
        self.array=np.zeros((1,19,8,8),dtype=np.float32)
        self.x=torch.from_numpy(self.array)
        self.mat_array=np.zeros(1,dtype=np.float32)
        self.mat=torch.from_numpy(self.mat_array)
        self.rows=np.array([7-(s//8) for s in range(64)])
        self.cols=np.array([s%8 for s in range(64)])
        self.validated=False

    def encode(self,b):
        a=self.array[0];a.fill(0)
        material_white=0
        # python-chess colors: white=True, black=False.
        for color,offset in [(True,0),(False,6)]:
            for piece in range(1,7):
                bits=b.pieces_mask(piece,color)
                squares=[]
                while bits:
                    low=bits & -bits;squares.append(low.bit_length()-1);bits^=low
                if squares:a[offset+piece-1,self.rows[squares],self.cols[squares]]=1
                material_white+=(1 if color else -1)*VALUES[piece]*len(squares)
        a[12].fill(float(b.turn))
        for plane,color,kingside in [(13,True,True),(14,True,False),(15,False,True),(16,False,False)]:
            flag=b.has_kingside_castling_rights(color) if kingside else b.has_queenside_castling_rights(color)
            a[plane].fill(float(flag))
        if b.ep_square is not None:a[17,self.rows[b.ep_square],self.cols[b.ep_square]]=1
        a[18].fill(b.halfmove_clock/100.0)
        self.mat_array[0]=material_white if b.turn else -material_white
        return self.x,self.mat

    @torch.inference_mode()
    def validate(self,boards):
        from board_encoder import encode_board
        from search_engine import material
        for b in boards:
            x,m=self.encode(b)
            expected=encode_board(b).unsqueeze(0).cpu()
            if not torch.equal(x,expected):
                raise ValueError('快速编码与本地board_encoder不一致，停止使用。FEN: '+b.fen())
            if float(m.item())!=material(b):raise ValueError('子力计算不一致')
        self.validated=True

    @torch.inference_mode()
    def __call__(self,b):
        if not self.validated:raise RuntimeError('请先调用validate验证本地编码兼容性')
        x,m=self.encode(b)
        value,_=self.model(x,m)
        return value.item()

class FastSearchEngine(OriginalSearchEngine):
    def static(self,b):
        # Includes all encoder inputs, with counters retained conservatively.
        # Only STATIC values are cached: never terminal/repetition/search bounds.
        k=(b.pawns,b.knights,b.bishops,b.rooks,b.queens,b.kings,
           b.occupied_co[0],b.occupied_co[1],b.turn,b.castling_rights,
           b.ep_square,b.halfmove_clock,b.fullmove_number,b.promoted,b.chess960)
        if k not in self.cache:
            v=float(self.evaluator(b))
            if not math.isfinite(v):raise ValueError('非有限局面评价')
            self.cache[k]=max(-1.,min(1.,v))
        return self.cache[k]
