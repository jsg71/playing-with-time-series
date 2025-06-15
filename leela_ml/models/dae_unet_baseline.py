"""
─────────────────────────────────────────────────────────────────────────────
 UNet1D & residual Block – detailed walkthrough
 Author: jsg
─────────────────────────────────────────────────────────────────────────────

What this file contains
───────────────────────
• Block   – a *residual* double-convolution unit for 1-D feature maps.
• UNet1D  – a 1-D variant of the classical U-Net architecture, depth-4
            with skip connections and transposed-conv up-sampling.

Typical use-case
────────────────
Signal denoising or segmentation on 1-D series (audio, sensor data,
ECG, etc.).  Input and output are both **(B, 1, L)** where:
    B – batch size
    1 – single input channel (modify `c` if multi-channel)
    L – sequence length (any multiple of 2^depth — here 16).

Dependencies
────────────
• PyTorch ≥ 1.13   (tested on 2.3.0)

-----------------------------------------------------------------
Class: Block
-----------------------------------------------------------------
Args
• c_in   (int) – input channel count.
• c_out  (int) – output channel count.

Forward pass
    body = Conv1d(9) → BN → ReLU → Conv1d(9) → BN → ReLU
    skip = Conv1d(1×1)  **only if** c_in ≠ c_out
    return body(x) + skip(x)

Design notes
• Kernel = 9, padding = 4 keeps sequence length unchanged.
• Residual path eases gradient flow (deep nets train faster).
• Uses `inplace=True` ReLU – a small memory win.

Strengths & weaknesses
✓ Simple, proven residual design.
✗ BatchNorm1d can be brittle with tiny batch sizes; consider GroupNorm.

-----------------------------------------------------------------
Class: UNet1D
-----------------------------------------------------------------
Args
• depth (int, default 4) – number of down-/up-sampling levels.
• base  (int, default 16) – channel width of first encoder block.

Channel progression
    depth=4, base=16  ⇒  encoder: 16-32-64-128
                         decoder: 128-64-32-16

Forward pass
1. Encoder loop
       for each Block:
           x = Block(x)
           save x to skips
           x = MaxPool1d(2)(x)   # halves sequence length
2. Decoder loop (reverse order)
       x  = ConvTranspose1d(stride=2)(x)  # doubles length
       pad right by 1 if odd to match skip length
       x  = Block( concat([x, skip]) )
3. Final 1×1 Conv → return reconstruction/prediction.

Key implementation details & rationale
• ConvTranspose1d(4, stride=2, padding=1) – exact inverse of pooling.
• Padding step ensures even-length requirement is relaxed (odd lengths OK).
• Skip concatenation doubles feature depth; subsequent Block halves it.

Strengths
• Symmetric U-Net retains fine-grained detail via skips.
• Fully convolutional – works with arbitrary input length.
• Memory-friendly: `inplace` activations + channel tapering on the way up.

Weaknesses / caveats
• Uses BatchNorm – sensitive to batch size; swap to Instance/GroupNorm
  for noisy, variable-length data or small mini-batches.
• ConvTranspose1d can introduce checkerboard artefacts; an alternative
  is up-sampling (nearest/linear) + normal Conv1d.
• No dropout / weight-norm – may over-fit on tiny datasets.

Suggested improvements
• Replace BatchNorm1d with GroupNorm (e.g., 8 groups).
• Add configurable dropout inside `Block` for regularisation.
• Parameterise kernel size & activation (ReLU → GELU often helpful).
• Provide `return_skips` flag to expose intermediate features for
  multi-scale losses.
• Register `self.pool = nn.MaxPool1d(2)` once instead of recreating it.

Example instantiation & test-run
────────────────────────────────
>>> net = UNet1D(depth=4, base=16)
>>> x   = torch.randn(8, 1, 4096)   # batch of 8, length 4096
>>> y   = net(x)
>>> y.shape   # (8, 1, 4096)
─────────────────────────────────────────────────────────────────────────────
"""


import torch
import torch.nn.functional as F

# ── residual double-conv block ────────────────────────────────────
class Block(torch.nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv1d(c_in, c_out, 9, padding=4),
            torch.nn.BatchNorm1d(c_out),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(c_out, c_out, 9, padding=4),
            torch.nn.BatchNorm1d(c_out),
            torch.nn.ReLU(inplace=True),
        )
        self.skip = torch.nn.Conv1d(c_in, c_out, 1) if c_in != c_out else torch.nn.Identity()

    def forward(self, x): return self.body(x) + self.skip(x)

# ── 1-D U-Net ─────────────────────────────────────────────────────
class UNet1D(torch.nn.Module):
    """depth=4, base=16 ⇒ enc 16-32-64-128, dec 128-64-32-16"""
    def __init__(self, depth=4, base=16):
        super().__init__()
        self.enc, self.dec = torch.nn.ModuleList(), torch.nn.ModuleList()
        c = 1
        # encoder
        for d in range(depth):
            c_next = base * 2 ** d
            self.enc.append(Block(c, c_next)); c = c_next
        # decoder
        for d in reversed(range(depth)):
            c_skip  = base * 2 ** d
            self.dec.append(torch.nn.ConvTranspose1d(c, c // 2, 4, 2, 1))
            self.dec.append(Block(c // 2 + c_skip, c_skip))
            c = c_skip
        self.out = torch.nn.Conv1d(base, 1, 1)

    def forward(self, x):
        skips = []
        for blk in self.enc:
            x = blk(x); skips.append(x); x = torch.nn.MaxPool1d(2)(x)
        for deconv, blk in zip(self.dec[0::2], self.dec[1::2]):
            x = deconv(x); skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.pad(x, (0, 1))        # pad right if odd
            x = blk(torch.cat([x, skip], 1))
        return self.out(x)
