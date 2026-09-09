import io
from pathlib import Path

import chess
import chess.pgn
import torch

from board_encoder import encode_board
from move_encoder import encode_move
from model_cnn import ChessCNN


def main():
    device = torch.device("cuda")
    root = Path(__file__).resolve().parent

    # 加载已经训练好的参数
    model = ChessCNN().to(device)
    weights = torch.load(
        root / "chess_model_balanced.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(weights)
    model.eval()

    text = input(
        "输入棋谱，例如 1. e4 e5；直接回车使用初始局面：\n"
    ).strip()

    if text:
        game = chess.pgn.read_game(io.StringIO(text))

        if game is None or game.errors:
            print("棋谱读取失败，请检查格式和走法。")
            return

        board = game.end().board()
    else:
        board = chess.Board()

    print("\n当前棋盘：")
    print(board)
    print("轮到：", "白方" if board.turn else "黑方")

    # 检查将死、逼和等自动终局情况
    if board.is_game_over():
        print("对局已结束，结果：", board.result())
        return

    legal_moves = list(board.legal_moves)

        # 尝试所有合法走法，优先寻找一步将死
    mating_moves = []

    for move in legal_moves:
        board.push(move)

        if board.is_checkmate():
            mating_moves.append(move)

        board.pop()  # 撤销试走，恢复原局面

    if mating_moves:
        print("\n搜索发现一步将死：")

        for move in mating_moves:
            print(f"{board.san(move)} ({move.uci()})")

        return

    # 增加一维，表示这一批只有一个局面
    inputs = encode_board(board).unsqueeze(0).to(device)

    with torch.no_grad():
        scores = model(inputs)[0]

        # 只取合法走法的分数
        legal_ids = torch.tensor(
            [encode_move(move) for move in legal_moves],
            dtype=torch.long,
            device=device,
        )
        legal_scores = scores[legal_ids]

        # 在合法走法之间换算成概率
        probabilities = torch.softmax(legal_scores, dim=0)
        values, indices = probabilities.topk(
            min(3, len(legal_moves))
        )

    print("\n模型推荐：")

    for rank, (probability, index) in enumerate(
        zip(values.cpu().tolist(), indices.cpu().tolist()),
        start=1,
    ):
        move = legal_moves[index]
        print(
            f"{rank}. {board.san(move)} "
            f"({move.uci()})  概率：{probability:.1%}"
        )


if __name__ == "__main__":
    main()