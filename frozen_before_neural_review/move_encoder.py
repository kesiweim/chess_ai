import chess

# 0 表示没有升变，其他编号表示升变成哪种棋子
PROMOTIONS = [
    None,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
]

NUM_MOVES = 64 * 64 * 5


def encode_move(move):
    """将走法转换成整数编号。"""
    promotion_id = PROMOTIONS.index(move.promotion)

    return (
        (move.from_square * 64 + move.to_square) * 5
        + promotion_id
    )


def decode_move(move_id):
    """将整数编号还原成走法；不负责判断是否合法。"""
    if not 0 <= move_id < NUM_MOVES:
        raise ValueError("走法编号超出范围")

    squares, promotion_id = divmod(move_id, 5)
    from_square, to_square = divmod(squares, 64)

    return chess.Move(
        from_square,
        to_square,
        promotion=PROMOTIONS[promotion_id],
    )


if __name__ == "__main__":
    # 最后一个例子表示兵从 e7 到 e8，升变成后
    examples = ["e2e4", "g1f3", "e7e8q"]

    for text in examples:
        move = chess.Move.from_uci(text)
        move_id = encode_move(move)
        restored = decode_move(move_id)

        assert restored == move, "转换前后不一致"

        print(f"{text} → 编号 {move_id} → 还原 {restored.uci()}")

    print("\n走法编码检查通过！")