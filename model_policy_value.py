from torch import nn

from model_cnn import ChessCNN


class PolicyValueModel(ChessCNN):
    def __init__(self):
        super().__init__()

        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        features = self.features(x)

        policy_scores = self.policy(features)
        value = self.value_head(features).squeeze(-1)

        return policy_scores, value