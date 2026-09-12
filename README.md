# Neural 2.0.0 — v4 原版（稳定基线）

Neural 线的初版模型，也就是 `chess_studio` 里的 **v4 原版（稳定基线）**。
由一对网络组成：策略网负责走法排序，价值网负责叶节点评估。

本分支只包含**加载和运行这个模型所需的最小文件**：不含训练数据，不含自对弈轮次
（r8 / r15）的权重，也不接 Syzygy。

## 权重

| 角色 | 文件 | 字节 | SHA-256 |
|---|---|---|---|
| 策略网 | `chess_model_balanced.pt` | 13,157,801 | `60AB3E89EBF853380F67777825645C12C3716221F58F3B311A713FFA6D60CE00` |
| 价值网 | `value_full_runs/20260908_113914_349502/value_full_epoch2.pt` | 2,590,579 | `4E44421B422C8467A32C06C85A31A5591B8397AE24C2F54FDD39C778FE7DD4A0` |

`chess_model_balanced.pt` 同时是 Material 1.0.0 的冻结资产：Material 模式用同一个
策略网排序走法，只把叶节点评价换成子力公式。

## 代码

| 文件 | 作用 |
|---|---|
| `play_v4.py` | 命令行对弈入口 |
| `model_cnn.py` | 策略网 `ChessCNN`（19×8×8 → 20480 维走法分数） |
| `model_residual_value.py` | 价值网 `ResidualValueModel`（子力 + 残差修正，tanh 输出） |
| `board_encoder.py` | 局面 → 19×8×8 张量 |
| `move_encoder.py` | 走法 ↔ 索引（64×64×5） |
| `search_engine.py` | 基础 negamax + `policy_order` + `material` |
| `neural_fast.py` | 快速评估与搜索封装 |
| `neural_search_v2.py` | 神经网络搜索（历史感知 PUCT） |
| `neural_search_v4.py` | v4 搜索（v2 + 走法排序改进） |
| `neural_inference_v3.py` | FP32 `torch.jit.trace` + freeze 推理包装 |

## 运行

依赖 `python-chess`、`numpy`、`torch`（CPU 即可）。在仓库根目录：

```bash
python -m pip install chess numpy
python play_v4.py                 # 默认 5 秒/手、深度 5
python play_v4.py --seconds 3 --depth 4
```

`play_v4.py` 会校验价值网哈希必须是 `4e44421b…d4a0`，不符就直接退出，避免误用别的模型。

## 本分支未包含

- 训练数据（`*.jsonl`、`*.csv.zst`、`*.db`）
- 自对弈轮次权重 r8 / r15
- Syzygy 表库与 `syzygy_v4.py` / `syzygy_mate_candidate.py`（本入口不需要；
  需要带表库对弈时可另取 Material 1.0.0 的那两个模块）
- 竞技场、自对弈与训练脚本
