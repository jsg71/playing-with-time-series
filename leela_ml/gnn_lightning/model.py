"""Graph neural network for multi-station lightning localisation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class NodeEncoder(nn.Module):
    """Encode waveform snippets into node embeddings.

    The encoder is a small 1‑D CNN mirroring the front‑end described by
    Tian et al. (2025).  It reduces each station's waveform (``T`` samples)
    to a fixed‑length feature vector used by the graph layers.
    """

    def __init__(self, out_channels: int = 64) -> None:
        super().__init__()
        # three layers of strided convolution followed by global pooling
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, out_channels, kernel_size=5, stride=2, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # slight dropout to help generalisation
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.pool(x)
        return x.squeeze(-1)


class LightningGNN(nn.Module):
    """Graph network mirroring the architecture in Tian et al. (2025)."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        # embed each station's waveform independently -------------------------
        self.encoder = NodeEncoder(hidden)

        # two layers of graph convolution over the sensor graph
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)

        # final linear readout maps pooled graph features to latitude/longitude
        self.lin = nn.Linear(hidden, 2)

    def forward(self, data):
        # ``data.x`` has shape (num_nodes, T); add channel dim for Conv1d
        x = self.encoder(data.x.unsqueeze(1))
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin(x)
