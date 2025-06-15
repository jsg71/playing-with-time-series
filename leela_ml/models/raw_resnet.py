"""
RawResNet1D – 6 residual blocks + 1×1 squeeze. ~130 k params.
Outputs raw logits (no Sigmoid).
"""
import torch.nn as nn
import torch.nn.functional as F

class RawResNet1D(nn.Module):
    def __init__(self, ch: int = 64, blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, 7, padding=3),
            nn.BatchNorm1d(ch),
            nn.ReLU()
        )

        def res_block(c):
            return nn.Sequential(
                nn.Conv1d(c, c, 3, padding=1), nn.BatchNorm1d(c), nn.ReLU(),
                nn.Conv1d(c, c, 3, padding=1), nn.BatchNorm1d(c)
            )

        self.body    = nn.Sequential(*[res_block(ch) for _ in range(blocks)])
        self.squeeze = nn.Conv1d(ch, ch, 1)
        self.head    = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ch, 1)
        )

    def forward(self, x):
        y = self.stem(x)
        y = self.body(y) + y
        y = F.relu(self.squeeze(F.relu(y)))
        return self.head(F.relu(y))
