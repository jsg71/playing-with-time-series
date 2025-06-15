"""
ae_models.py
------------
‑ ConvAE      : original 2‑layer auto‑encoder (baseline / compatibility)
‑ DeepConvAE  : 5‑layer under‑complete AE (recommended)
"""
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────
class ConvAE(nn.Module):
    def __init__(self, ch: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, ch, 7, 2, 3), nn.ReLU(),
            nn.Conv1d(ch, ch, 7, 2, 3), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(ch, ch, 7, 2, 3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(ch, 1, 7, 2, 3, output_padding=1),  nn.Tanh()
        )
    def forward(self, x): return self.dec(self.enc(x))

# ─────────────────────────────────────────────────────────────────────
class DeepConvAE(nn.Module):
    """5‑layer under‑complete AE: 1‑>128 channels, 32× downsample."""
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv1d(1,   16, 7, 2, 3), nn.ReLU(),
            nn.Conv1d(16,  32, 7, 2, 3), nn.ReLU(),
            nn.Conv1d(32,  64, 7, 2, 3), nn.ReLU(),
            nn.Conv1d(64, 128, 7, 2, 3), nn.ReLU(),
            nn.Conv1d(128,128, 7, 2, 3), nn.ReLU()
        )
        # Decoder (mirror)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(128,128,7,2,3,output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(128, 64,7,2,3,output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(64,  32,7,2,3,output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32,  16,7,2,3,output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16,   1,7,2,3,output_padding=1), nn.Tanh()
        )
    def forward(self, x): return self.dec(self.enc(x))
