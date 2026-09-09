from pathlib import Path

import chess
import torch

from board_encoder import encode_board
from move_encoder import encode_move
from model_cnn import ChessCNN


def parse_move(board, text):
    """支持 SAN（Nf3）和 UCI（g1f3）。"""
    text = text.strip()

    try:
        return board.parse_san(text)
    except ValueError:
        return board.parse_uci(text)


def recommend(board, model, device):
    legal_moves = list(board.legal_moves)

    # 优先检查全部合法走法中的一步将死
    for move in legal_moves:
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()

        if is_mate:
            return move, True

    with torch.inference_mode():
        inputs = encode_board(board).unsqueeze(0).to(device)
        scores = model(inputs)[0]

        legal_ids = torch.tensor(
            [encode_move(move) for move in legal_moves],
            dtype=torch.long,
            device=device,
        )
        index = scores[legal_ids].argmax().item()

    return legal_moves[index], False


def main():
    root = Path(__file__).resolve().parent
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ChessCNN().to(device)
    model.load_state_dict(torch.load(
        root / "chess_model_balanced.pt",
        map_location=device,
        weights_only=True,
    ))
    model.eval()

    while True:
        color = input("你执白还是黑？输入 白 / 黑：").strip().lower()

        if color in ("白", "白方", "w", "white"):
            my_color = chess.WHITE
            break
        if color in ("黑", "黑方", "b", "black"):
            my_color = chess.BLACK
            break

        print("请输入 白 或 黑。")

    board = chess.Board()

    print("\n从初始局面开始。")
    print("走法支持 Nf3、exd5、O-O，或 g1f3、e7e8q。")
    print("输入 board 查看棋盘，输入 quit 退出。")
    print("输入 undo 撤销最近一手，可以连续撤销。")

    while True:
        if board.is_game_over():
            print("\n对局结束：", board.result())
            print(board)
            break

        my_turn = board.turn == my_color
        suggested = None

        if my_turn:
            suggested, is_mate = recommend(board, model, device)

            print(
                f"\n建议你走：{board.san(suggested)}"
                f"（{suggested.uci()}）"
            )
            if is_mate:
                print("这一步可以直接将死。")

            prompt = (
                "在棋盘上走完后按回车确认；"
                "若走了别的棋，输入实际走法："
            )
        else:
            prompt = "\n请输入对手刚走的棋："

        text = input(prompt).strip()

        if text.lower() == "quit":
            break

        if text.lower() == "board":
            print(board)
            print("当前 FEN：", board.fen())
            continue

        if text.lower() == "undo":
            if board.move_stack:
                board.pop()
                print("已撤销最近一手。")
            else:
                print("目前没有可以撤销的走法。")
            continue

        if not text:
            if my_turn:
                move = suggested
            else:
                print("请先输入对手走法。")
                continue
        else:
            try:
                move = parse_move(board, text)
            except ValueError:
                print("无法识别，或该走法在当前局面不合法，请重新输入。")
                continue

        name = board.san(move)
        board.push(move)
        print(f"已记录：{name}（{move.uci()}）")


if __name__ == "__main__":
    main()