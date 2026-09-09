import hashlib
import json
import random
from pathlib import Path

import chess
import chess.pgn


def main():
    root = Path(__file__).resolve().parent
    source = root / "twic1660.pgn"

    games = []
    seen = set()
    read_count = 0
    skipped = 0
    duplicates = 0

    print("正在读取棋谱……", flush=True)

    # 替换无法解码的字符；棋子走法本身使用 ASCII 字符
    with source.open("r", encoding="utf-8-sig", errors="replace") as file:
        while True:
            game = chess.pgn.read_game(file)

            if game is None:
                break

            read_count += 1

            if read_count % 500 == 0:
                print(f"已读取 {read_count} 盘", flush=True)

            board = game.board()

            # 只保留标准国际象棋、解析正常且有结果的对局
            if (
                game.errors
                or type(board) is not chess.Board
                or board.chess960
                or not board.is_valid()
                or game.headers.get("Result") not in (
                    "1-0", "0-1", "1/2-1/2"
                )
            ):
                skipped += 1
                continue

            moves = list(game.mainline_moves())

            if not moves:
                skipped += 1
                continue

            # 初始局面与全部走法一致，视为重复对局
            identity = (
                board.fen()
                + "|"
                + " ".join(move.uci() for move in moves)
            )
            game_id = hashlib.sha256(identity.encode()).hexdigest()

            if game_id in seen:
                duplicates += 1
                continue

            samples = []
            valid = True

            for move in moves:
                if move not in board.legal_moves:
                    valid = False
                    break

                samples.append({
                    "position": board.fen(),
                    "next_move": move.uci(),
                })
                board.push(move)

            if not valid:
                skipped += 1
                continue

            seen.add(game_id)
            games.append((game_id, samples))

    if len(games) < 2:
        raise ValueError("有效对局不足，无法划分训练集和验证集")

    # 固定随机种子，重复运行时划分一致
    random.Random(42).shuffle(games)

    validation_count = max(1, round(len(games) * 0.1))
    validation_games = games[:validation_count]
    training_games = games[validation_count:]

    def save_samples(filename, selected_games):
        count = 0

        with (root / filename).open("w", encoding="utf-8") as file:
            for game_id, samples in selected_games:
                for sample in samples:
                    record = {"game_id": game_id, **sample}
                    file.write(json.dumps(record) + "\n")
                    count += 1

        return count

    train_count = save_samples("train.jsonl", training_games)
    val_count = save_samples("val.jsonl", validation_games)

    print("\n处理完成")
    print(f"读取对局：{read_count}")
    print(f"跳过对局：{skipped}")
    print(f"重复对局：{duplicates}")
    print(f"训练集：{len(training_games)} 盘，{train_count} 条样本")
    print(f"验证集：{len(validation_games)} 盘，{val_count} 条样本")
    print("已保存：train.jsonl 和 val.jsonl")


if __name__ == "__main__":
    main()