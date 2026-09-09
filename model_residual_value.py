import torch
from torch import nn

from model_cnn import ChessCNN


class ResidualValueModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 独立的价值特征层，初始化时复制原 CNN 的特征参数
        self.features = ChessCNN().features

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # 初始修正量严格为零
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, material):
        correction = self.head(self.features(x)).squeeze(-1)
        value = torch.tanh(material / 300.0 + correction)
        return value, correction