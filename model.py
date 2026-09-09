from torch import nn

from board_encoder import INPUT_CHANNELS
from move_encoder import NUM_MOVES


class ChessModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(INPUT_CHANNELS * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_MOVES),
        )

    def forward(self, x):
        return self.network(x)