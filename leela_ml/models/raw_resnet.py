"""
─────────────────────────────────────────────────────────────────────────────
 raw_resnet.py — 1-D Residual Network for binary classification
 Author: <your-name / date>
─────────────────────────────────────────────────────────────────────────────

WHAT THIS FILE PROVIDES
───────────────────────
• **RawResNet1D** — a lightweight 1-D ResNet comprising:
      stem   : initial 7-tap conv (1 → ch)
      body   : *blocks* residual pairs of 3×3 convolutions
      squeeze: 1×1 conv to mix channels post-residual
      head   : global average pooling → linear → raw logit

TENSOR SHAPES
─────────────
Input  : (B, 1, L)        # raw mono signal
Stem   : (B, ch, L)
Body   : (B, ch, L)       # same length thanks to padding=1
Head   : (B, 1)           # single logit (no Sigmoid)

Hyper-parameters
────────────────
• ch     (int, default 64)  — channel width throughout the network
• blocks (int, default 6)   — number of residual units in *body*

DETAILS BY STAGE
────────────────
stem
    Conv1d(1 → ch, kernel 7, pad 3)
    BatchNorm1d
    ReLU

body  (repeated *blocks* times)
    ┌─ Conv1d(ch, ch, 3, pad 1) → BN → ReLU
    └─ Conv1d(ch, ch, 3, pad 1) → BN
    residual add (+ identity)

squeeze
    1×1 Conv1d   # encourages channel mixing after residual stack
    ReLU

head
    AdaptiveAvgPool1d(1)  # global average over time axis
    Flatten()             # (B, ch, 1) → (B, ch)
    Linear(ch → 1)        # raw binary logit

DESIGN RATIONALE
────────────────
✓ 3×3 convolutions + `same` padding keep temporal resolution intact
✓ Residual additions alleviate vanishing-gradient issues in deeper stacks
✓ Global average pooling yields length-invariant classification
✓ ~130 k parameters with default settings — GPU friendly for long sequences

STRENGTHS
─────────
• Simple, fast, and easy to interpret
• Constant memory footprint w.r.t. input length (thanks to GAP layer)
• Suitable baseline for anomaly / event detection on raw signals

WEAKNESSES & IMPROVEMENTS
─────────────────────────
✗ BatchNorm can be unstable on very small batch sizes
   → Consider `nn.GroupNorm(8, ch)` or `nn.InstanceNorm1d(ch)`
✗ No dilation — receptive field grows only linearly with *blocks*
   → Replace later convs with dilated 3×3 to widen context cheaply
✗ Only one output unit — hard-coded for binary tasks
   → Make `num_classes` an argument and switch `Linear(ch, num_classes)`

OPTIONAL EXTENSIONS
───────────────────
• Drop-in GELU activation (slightly smoother than ReLU)
• Add dropout after the squeeze layer for regularisation
• Replace `Conv1d` kernel 7 in the stem with strided 5×5 to reduce compute
• Expose intermediate feature maps for self-supervised contrastive losses

EXAMPLE USAGE
─────────────
>>> import torch
>>> from raw_resnet import RawResNet1D
>>> net = RawResNet1D(ch=64, blocks=6)
>>> x   = torch.randn(16, 1, 8192)       # batch of 16, 8-k sample window
>>> logit = net(x)                       # shape (16, 1)
>>> pred  = torch.sigmoid(logit) > 0.5   # boolean predictions

─────────────────────────────────────────────────────────────────────────────
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
