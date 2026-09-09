import random
from collections import defaultdict
from pathlib import Path

import chess
import torch
import torch.nn.functional as F

from board_encoder import encode_board
from model_residual_value import ResidualValueModel
from evaluate_choices import read_records, key, material


def group_records(records):
    groups = defaultdict(list)

    for record in records:
        groups[record["parent_id"]].append(record)

    complete = []

    for records in groups.values():
        parent = chess.Board(records[0]["parent_position"])
        expected = {m.uci() for m in parent.legal_moves}
        actual = [r["move"] for r in records]

        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError("发现不完整或重复的候选组")

        for record in records:
            child = parent.copy()
            child.push_uci(record["move"])
            if child.fen() != chess.Board(record["position"]).fen():
                raise ValueError("候选走法与子局面不一致")

        complete.append(records)

    return complete


def make_batch(records, device):
    boards = [chess.Board(r["position"]) for r in records]

    inputs = torch.stack([
        encode_board(board) for board in boards
    ]).to(device)

    materials = torch.tensor(
        [material(board) for board in boards],
        dtype=torch.float32,
        device=device,
    )

    targets = torch.tensor(
        [r["value"] for r in records],
        dtype=torch.float32,
        device=device,
    )

    return inputs, materials, targets


def predict_children(model, records, device):
    inputs, materials, targets = make_batch(records, device)
    predictions, _ = model(inputs, materials)

    # 终局由规则判定，不依赖网络
    terminal = torch.tensor(
        [r["terminal"] for r in records],
        dtype=torch.bool,
        device=device,
    )
    predictions = torch.where(terminal, targets, predictions)

    return predictions, targets


def evaluate_ranking(model, groups, device):
    model.eval()
    hits = 0
    regret = 0.0

    with torch.inference_mode():
        for records in groups:
            predictions, _ = predict_children(model, records, device)

            # 子局面轮到对手，因此原走棋方价值取负号
            scores = (-predictions).cpu().tolist()
            targets = [r["parent_move_value"] for r in records]

            chosen = min(
                range(len(records)),
                key=lambda i: (-scores[i], records[i]["move"]),
            )

            gap = max(0.0, max(targets) - targets[chosen])
            regret += gap
            hits += int(gap <= 1e-6)

    return hits, regret / len(groups)


def evaluate_old(model, records, device):
    model.eval()
    squared = 0.0

    with torch.inference_mode():
        for start in range(0, len(records), 256):
            batch = records[start:start + 256]
            inputs, materials, targets = make_batch(batch, device)
            predictions, _ = model(inputs, materials)
            squared += (predictions - targets).square().sum().item()

    return squared / len(records)


def main():
    rng = random.Random(42)
    torch.manual_seed(42)
    root = Path(__file__).resolve().parent

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA 显卡")
    device = torch.device("cuda")

    old_train = read_records(root / "value_train.jsonl")
    old_val = read_records(root / "value_val.jsonl")
    choice_train = read_records(root / "move_choices_train.jsonl")
    choice_val = read_records(root / "move_choices_val.jsonl")

    train_groups = group_records(choice_train)
    val_groups = group_records(choice_val)

    training_keys = {
        key(r["position"]) for r in old_train + choice_train
    }

    # 与之前候选评估采用相同的整组排除方式
    val_groups = [
        group for group in val_groups
        if key(group[0]["parent_position"]) not in training_keys
        and not any(key(r["position"]) in training_keys for r in group)
    ]

    # 新增子局面也不能与旧评分验证样本重合
    old_val = [
        r for r in old_val
        if key(r["position"]) not in training_keys
    ]

    if not train_groups or not val_groups or not old_val:
        raise ValueError("清理后有数据集为空")
    if len(old_train) < 64:
        raise ValueError("复习数据不足")

    model = ResidualValueModel().to(device)
    model.load_state_dict(torch.load(
        root / "chess_model_residual_value.pt",
        map_location=device,
        weights_only=True,
    ))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.00001,
        weight_decay=0.001,
    )

    print(f"训练父局面：{len(train_groups)}")
    print(f"验证父局面：{len(val_groups)}")
    print(f"旧评分验证样本：{len(old_val)}")

    hits, regret = evaluate_ranking(model, val_groups, device)
    baseline_mse = evaluate_old(model, old_val, device)

    print(
        f"微调前：最优命中 {hits}/{len(val_groups)} | "
        f"平均价值损失 {regret:.4f} | "
        f"旧评分 MSE {baseline_mse:.4f}",
        flush=True,
    )

    for epoch in range(1, 4):
        rng.shuffle(train_groups)
        model.train()

        for step, group in enumerate(train_groups, start=1):
            optimizer.zero_grad(set_to_none=True)

            predictions, targets = predict_children(
                model, group, device
            )
            score_loss = F.mse_loss(predictions, targets)

            parent_predictions = -predictions
            parent_targets = -targets

            # 行 i 比列 j 明显更好，才构造排序要求
            target_gap = (
                parent_targets[:, None] - parent_targets[None, :]
            )
            prediction_gap = (
                parent_predictions[:, None]
                - parent_predictions[None, :]
            )

            mask = target_gap > 0.10

            if mask.any():
                margin = target_gap.clamp(max=0.20)
                ranking_loss = F.relu(
                    margin[mask] - prediction_gap[mask]
                ).mean()
            else:
                ranking_loss = predictions.sum() * 0.0

            replay = rng.sample(old_train, 64)
            inputs, materials, replay_targets = make_batch(
                replay, device
            )
            replay_predictions, _ = model(inputs, materials)
            replay_loss = F.mse_loss(
                replay_predictions, replay_targets
            )

            loss = score_loss + 0.5 * ranking_loss + replay_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 200 == 0:
                print(
                    f"第 {epoch}/3 轮：{step}/{len(train_groups)} 组",
                    flush=True,
                )

        hits, regret = evaluate_ranking(model, val_groups, device)
        old_mse = evaluate_old(model, old_val, device)

        output = root / f"chess_model_ranked_epoch{epoch}.pt"
        torch.save(model.state_dict(), output)

        print(
            f"\n第 {epoch} 轮："
            f"最优命中 {hits}/{len(val_groups)} | "
            f"平均价值损失 {regret:.4f} | "
            f"旧评分 MSE {old_mse:.4f}",
            flush=True,
        )
        print("已保存：", output)

    print("\n训练完成。先比较结果，再选择用于搜索的版本。")


if __name__ == "__main__":
    main()