import json
from pathlib import Path

import chess
import torch


INPUT_CHANNELS = 19


def encode_board(board):
    """将局面转换成 19×8×8 的数字数组。"""
    encoded = torch.zeros(
        (INPUT_CHANNELS, 8, 8),
        dtype=torch.float32,
    )

    # 第 0～5 层：白方棋子；第 6～11 层：黑方棋子
    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)

        channel = piece.piece_type - 1
        if piece.color == chess.BLACK:
            channel += 6

        encoded[channel, row, col] = 1.0

    # 第 12 层：轮到哪一方
    encoded[12].fill_(float(board.turn == chess.WHITE))

    # 第 13～16 层：王车易位权
    rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]

    for channel, allowed in enumerate(rights, start=13):
        encoded[channel].fill_(float(allowed))

    # 第 17 层：吃过路兵目标格
    if board.ep_square is not None:
        row = 7 - chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        encoded[17, row, col] = 1.0

    # 第 18 层：半回合计数，缩放到较小的数值
    encoded[18].fill_(board.halfmove_clock / 100.0)

    return encoded


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent / "samples.json"

    with data_path.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    # 将所有样本编码后，叠放成一批
    inputs = torch.stack([
        encode_board(chess.Board(sample["position"]))
        for sample in samples
    ])

    print("整批数据形状：", tuple(inputs.shape))
    print("第一条样本棋子数：", int(inputs[0, :12].sum().item()))
    print("第一条轮到白方：", bool(inputs[0, 12, 0, 0].item()))
    print("第二条轮到白方：", bool(inputs[1, 12, 0, 0].item()))