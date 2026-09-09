from torch import nn

from board_encoder import INPUT_CHANNELS
from move_encoder import NUM_MOVES


class ChessCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 保留棋盘的 8×8 排列，提取空间特征
        self.features = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        # 将提取出的特征转换成每种走法的分数
        self.policy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_MOVES),
        )

    def forward(self, x):
        return self.policy(self.features(x))