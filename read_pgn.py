import io
import chess.pgn

# 一小段棋谱：数字是回合号，* 表示尚未给出对局结果
pgn_text = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *"

# 读取棋谱
game = chess.pgn.read_game(io.StringIO(pgn_text))
board = game.board()

print("初始棋盘：")
print(board)

# 按棋谱依次走棋
for step, move in enumerate(game.mainline_moves(), start=1):
    side = "白方" if board.turn == chess.WHITE else "黑方"
    move_name = board.san(move)

    board.push(move)

    print(f"\n第 {step} 手：{side} {move_name}")
    print(board)

print("\n棋谱读取完成！")