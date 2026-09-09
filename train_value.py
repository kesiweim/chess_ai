import json
import random
from pathlib import Path

import chess
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from board_encoder import encode_board
from model_policy_value import PolicyValueModel


class ValueDataset(Dataset):
    def __init__(self, path):
        with path.open("r", encoding="utf-8") as file:
            self.records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        if not self.records:
            raise ValueError(f"数据为空：{path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        board = chess.Board(record["position"])

        return (
            encode_board(board),
            torch.tensor(record["value"], dtype=torch.float32),
        )


def evaluate(model, loader, device):
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    count = 0

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            features = model.features(inputs)
            predictions = model.value_head(features).squeeze(-1)

            errors = predictions - targets
            squared_error += errors.square().sum().item()
            absolute_error += errors.abs().sum().item()
            count += targets.numel()

    return squared_error / count, absolute_error / count


def main():
    random.seed(42)
    torch.manual_seed(42)

    root = Path(__file__).resolve().parent

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用的 CUDA 显卡")

    device = torch.device("cuda")

    train_data = ValueDataset(root / "value_train.jsonl")
    val_data = ValueDataset(root / "value_val.jsonl")

    train_loader = DataLoader(
        train_data,
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=256,
        num_workers=0,
    )

    model = PolicyValueModel().to(device)

    # 使用保留的平衡模型初始化已有部分
    weights = torch.load(
        root / "chess_model_balanced.pt",
        map_location=device,
        weights_only=True,
    )
    result = model.load_state_dict(weights, strict=False)

    # 只允许新增价值层的参数缺失
    expected_missing = {
        f"value_head.{name}"
        for name in model.value_head.state_dict()
    }

    if (
        set(result.missing_keys) != expected_missing
        or result.unexpected_keys
    ):
        raise ValueError(
            f"模型结构不匹配：{result}"
        )

    # 冻结原来的特征层和策略层
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for parameter in model.value_head.parameters():
        parameter.requires_grad_(True)

    optimizer = torch.optim.Adam(
        model.value_head.parameters(), lr=0.0003
    )
    loss_function = nn.MSELoss()

    # 基准：对所有局面都预测训练集平均价值
    train_mean = sum(
        r["value"] for r in train_data.records
    ) / len(train_data)

    baseline_mse = sum(
        (r["value"] - train_mean) ** 2
        for r in val_data.records
    ) / len(val_data)

    best_mse, initial_mae = evaluate(model, val_loader, device)
    output = root / "chess_model_policy_value.pt"

    # 先保存初始化版本，之后仅在验证 MSE 改善时更新
    torch.save(model.state_dict(), output)

    print(f"训练样本：{len(train_data):,}")
    print(f"验证样本：{len(val_data):,}")
    print(f"常数基准验证 MSE：{baseline_mse:.4f}")
    print(
        f"未训练价值层：MSE {best_mse:.4f}，"
        f"MAE {initial_mae:.4f}",
        flush=True,
    )

    for epoch in range(1, 6):
        model.eval()
        model.value_head.train()

        total_loss = 0.0
        count = 0

        for batch, (inputs, targets) in enumerate(
            train_loader, start=1
        ):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 原特征层不更新参数
            with torch.no_grad():
                features = model.features(inputs)

            optimizer.zero_grad(set_to_none=True)
            predictions = model.value_head(features).squeeze(-1)
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.numel()
            count += targets.numel()

            if batch % 50 == 0:
                print(
                    f"第 {epoch}/5 轮：已处理 {count:,} 条",
                    flush=True,
                )

        val_mse, val_mae = evaluate(model, val_loader, device)

        print(
            f"\n第 {epoch} 轮 | "
            f"训练 MSE {total_loss / count:.4f} | "
            f"验证 MSE {val_mse:.4f} | "
            f"验证 MAE {val_mae:.4f}",
            flush=True,
        )

        if val_mse < best_mse:
            best_mse = val_mse
            torch.save(model.state_dict(), output)
            print("已保存当前最佳价值层。", flush=True)

    print(f"\n最佳验证 MSE：{best_mse:.4f}")
    print(f"常数基准 MSE：{baseline_mse:.4f}")
    print("模型文件：", output)


if __name__ == "__main__":
    main()