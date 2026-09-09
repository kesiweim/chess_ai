import json
from pathlib import Path

import chess
import torch
from torch.utils.data import Dataset, DataLoader

from board_encoder import encode_board
from model_residual_value import ResidualValueModel


VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


class ValueDataset(Dataset):
    def __init__(self, path):
        with path.open("r", encoding="utf-8") as file:
            self.records = [
                json.loads(line) for line in file if line.strip()
            ]

        if not self.records:
            raise ValueError(f"数据为空：{path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        board = chess.Board(record["position"])

        material = sum(
            value * (
                len(board.pieces(piece_type, board.turn))
                - len(board.pieces(piece_type, not board.turn))
            )
            for piece_type, value in VALUES.items()
        )

        return (
            encode_board(board),
            torch.tensor(material, dtype=torch.float32),
            torch.tensor(record["value"], dtype=torch.float32),
        )


def evaluate(model, loader, device):
    model.eval()
    total = 0
    squared = 0.0
    absolute = 0.0

    with torch.inference_mode():
        for inputs, material, targets in loader:
            inputs = inputs.to(device)
            material = material.to(device)
            targets = targets.to(device)

            predictions, _ = model(inputs, material)
            errors = predictions - targets

            squared += errors.square().sum().item()
            absolute += errors.abs().sum().item()
            total += targets.numel()

    return squared / total, absolute / total


def main():
    torch.manual_seed(42)
    root = Path(__file__).resolve().parent

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA 显卡")
    device = torch.device("cuda")

    train_loader = DataLoader(
        ValueDataset(root / "value_train.jsonl"),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        num_workers=0,
    )
    val_loader = DataLoader(
        ValueDataset(root / "value_val.jsonl"),
        batch_size=256,
        num_workers=0,
    )

    model = ResidualValueModel().to(device)

    old_weights = torch.load(
        root / "chess_model_balanced.pt",
        map_location=device,
        weights_only=True,
    )

    feature_weights = {
        key.removeprefix("features."): value
        for key, value in old_weights.items()
        if key.startswith("features.")
    }
    model.features.load_state_dict(feature_weights)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": 0.00001},
            {"params": model.head.parameters(), "lr": 0.0001},
        ],
        weight_decay=0.001,
    )

    best_mse, baseline_mae = evaluate(model, val_loader, device)
    baseline_mse = best_mse
    output = root / "chess_model_residual_value.pt"

    # 先保存零修正版；只有验证 MSE 更低才替换
    torch.save(model.state_dict(), output)

    print(
        f"初始子力基准：MSE {best_mse:.4f} | "
        f"MAE {baseline_mae:.4f}",
        flush=True,
    )

    for epoch in range(1, 6):
        model.train()
        squared = 0.0
        total = 0

        for batch, (inputs, material, targets) in enumerate(
            train_loader, start=1
        ):
            inputs = inputs.to(device)
            material = material.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            predictions, correction = model(inputs, material)
            mse = (predictions - targets).square().mean()

            # 轻微约束修正量，减少不必要的大幅改动
            loss = mse + 0.001 * correction.square().mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            squared += mse.item() * targets.numel()
            total += targets.numel()

            if batch % 50 == 0:
                print(
                    f"第 {epoch}/5 轮：已处理 {total:,} 条",
                    flush=True,
                )

        val_mse, val_mae = evaluate(model, val_loader, device)

        print(
            f"\n第 {epoch} 轮 | "
            f"训练 MSE {squared / total:.4f} | "
            f"验证 MSE {val_mse:.4f} | "
            f"验证 MAE {val_mae:.4f}",
            flush=True,
        )

        if val_mse < best_mse:
            best_mse = val_mse
            torch.save(model.state_dict(), output)
            print("已保存优于此前版本的模型。", flush=True)

    print(f"\n子力基准 MSE：{baseline_mse:.4f}")
    print(f"最佳验证 MSE：{best_mse:.4f}")
    print("保存位置：", output)


if __name__ == "__main__":
    main()