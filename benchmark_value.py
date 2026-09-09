import json
import random
import statistics
import time
from pathlib import Path

import chess
import torch

from board_encoder import encode_board
from model_residual_value import ResidualValueModel


VALUES = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900}


def material(board):
    return sum(
        value * (
            len(board.pieces(piece, board.turn))
            - len(board.pieces(piece, not board.turn))
        )
        for piece, value in VALUES.items()
    )


def measure(function, device, iterations=200):
    # 预热不计入结果
    for index in range(20):
        function(index)

    if device == "cuda":
        torch.cuda.synchronize()

    trials = []

    for _ in range(3):
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for index in range(iterations):
            function(index)

        if device == "cuda":
            torch.cuda.synchronize()

        trials.append(
            (time.perf_counter() - start) * 1000 / iterations
        )

    return statistics.median(trials)


def main():
    root = Path(__file__).resolve().parent

    with (root / "value_val.jsonl").open(
        "r", encoding="utf-8"
    ) as file:
        records = [
            json.loads(line) for line in file if line.strip()
        ]

    random.Random(42).shuffle(records)
    boards = [
        chess.Board(record["position"])
        for record in records[:256]
    ]

    if len(boards) < 64:
        raise ValueError("测速需要至少 64 个局面")

    weights = torch.load(
        root / "chess_model_ranked_epoch4.pt",
        map_location="cpu",
        weights_only=True,
    )

    original_threads = torch.get_num_threads()
    results = []

    def report(name, milliseconds):
        results.append({"name": name, "ms": milliseconds})
        print(f"{name}：{milliseconds:.4f} 毫秒", flush=True)

    print("PyTorch：", torch.__version__)
    print("CPU 默认线程数：", original_threads)
    print("GPU：", (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else "不可用"
    ))
    print("开始测速，各项使用三次测量的中位数……", flush=True)

    with torch.inference_mode():
        # 单独测量棋盘编码，不包含 FEN 解析
        def encoding(index):
            return encode_board(boards[index % len(boards)])

        report(
            "棋盘编码/局面",
            measure(encoding, "cpu"),
        )

        # 提前编码，用来区分网络与数据准备耗时
        x_cpu = torch.stack([encode_board(b) for b in boards])
        m_cpu = torch.tensor(
            [material(b) for b in boards],
            dtype=torch.float32,
        )

        for threads in sorted({1, 2, 4, original_threads}):
            torch.set_num_threads(threads)

            model = ResidualValueModel().eval()
            model.load_state_dict(weights)

            def cpu_network(index):
                i = index % len(boards)
                value, _ = model(x_cpu[i:i + 1], m_cpu[i:i + 1])
                return value.item()

            def cpu_full(index):
                board = boards[index % len(boards)]
                x = encode_board(board).unsqueeze(0)
                m = torch.tensor(
                    [material(board)], dtype=torch.float32
                )
                value, _ = model(x, m)
                return value.item()

            report(
                f"CPU {threads}线程，仅网络/局面",
                measure(cpu_network, "cpu"),
            )
            report(
                f"CPU {threads}线程，编码到取回评分/局面",
                measure(cpu_full, "cpu"),
            )

        if torch.cuda.is_available():
            # GPU 测量固定 CPU 线程数，减少线程设置干扰
            torch.set_num_threads(1)
            model = ResidualValueModel().to("cuda").eval()
            model.load_state_dict(weights)

            x_gpu = x_cpu.to("cuda")
            m_gpu = m_cpu.to("cuda")

            def gpu_network(index):
                i = index % len(boards)
                value, _ = model(x_gpu[i:i + 1], m_gpu[i:i + 1])
                # 搜索需要立即使用评分，包含等待结果的时间
                return value.item()

            def gpu_full(index):
                board = boards[index % len(boards)]
                x = encode_board(board).unsqueeze(0).to("cuda")
                m = torch.tensor(
                    [material(board)],
                    dtype=torch.float32,
                    device="cuda",
                )
                value, _ = model(x, m)
                return value.item()

            report(
                "GPU 单局面，仅网络并取回评分",
                measure(gpu_network, "cuda"),
            )
            report(
                "GPU 单局面，编码到取回评分",
                measure(gpu_full, "cuda"),
            )

            for batch_size in [16, 64]:
                def gpu_batch(index):
                    start = (
                        index * batch_size
                    ) % (len(boards) - batch_size + 1)

                    value, _ = model(
                        x_gpu[start:start + batch_size],
                        m_gpu[start:start + batch_size],
                    )
                    return value.cpu()

                batch_ms = measure(
                    gpu_batch, "cuda", iterations=100
                )

                report(
                    f"GPU 批量{batch_size}，预编码整批",
                    batch_ms,
                )
                report(
                    f"GPU 批量{batch_size}，预编码摊到每局面",
                    batch_ms / batch_size,
                )

    torch.set_num_threads(original_threads)

    output = root / "value_benchmark.json"
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n测速完成，结果保存到：", output)


if __name__ == "__main__":
    main()