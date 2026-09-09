import io
import chess.pgn

pgn_text = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *"

game = chess.pgn.read_game(io.StringIO(pgn_text))
board = game.board()

samples = []

for move in game.mainline_moves():
    # 必须在走棋之前记录局面，否则就把答案泄露给模型了
    sample = {
        "position": board.fen(),
        "next_move": move.uci(),
    }
    samples.append(sample)

    # 记录完毕，再走这一步
    board.push(move)

print(f"共生成 {len(samples)} 条训练样本")

for index, sample in enumerate(samples, start=1):
    print(f"\n样本 {index}")
    print("题目（局面）：", sample["position"])
    print("答案（走法）：", sample["next_move"])




import json
from pathlib import Path

# 将数据保存在本程序所在的文件夹
output_path = Path(__file__).resolve().parent / "samples.json"

with output_path.open("w", encoding="utf-8") as file:
    json.dump(samples, file, ensure_ascii=False, indent=2)

# 重新读取，检查保存结果
with output_path.open("r", encoding="utf-8") as file:
    loaded_samples = json.load(file)

assert loaded_samples == samples, "保存后的数据与原数据不一致"

print(f"\n保存成功：{output_path}")
print(f"校验通过：共 {len(loaded_samples)} 条样本")