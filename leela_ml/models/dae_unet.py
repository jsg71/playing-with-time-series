"""
─────────────────────────────────────────────────────────────────────────────
 dae_unet.py — 1-D Denoising U-Net (Auto-Encoder)
 Author: <your-name / date>
─────────────────────────────────────────────────────────────────────────────

WHAT THIS FILE PROVIDES
───────────────────────
• **Block**   – a residual “double-conv” unit for 1-D feature maps
• **UNet1D**  – a 1-D U-Net (depth-4, base-16 by default) suitable for
  denoising or sequence-to-sequence tasks on time-series data

TENSOR CONVENTION
─────────────────
All tensors are (B, C, L)
  B = batch size C = channels (default 1) L = sequence length
Odd lengths are accepted; the decoder right-pads by 1 where necessary.

DEPENDENCIES PyTorch ≥ 1.13  (tested 2.3.0)
─────────────────────────────────────────────────────────────────────────────
CLASS: Block
────────────
Args
  c_in  – input channels
  c_out – output channels

Structure
  Conv1d(c_in→c_out, k=9, p=4) → BN → ReLU
  Conv1d(c_out→c_out, k=9, p=4) → BN → ReLU
  + skip(x)  # identity if c_in == c_out else 1×1 conv

Why it’s sensible
• Kernel 9 with padding 4 keeps length unchanged (same-size residual)
• Residual sum improves gradient flow and training speed
• 1×1 skip only when channel count changes → minimal parameters

Weak points
• BatchNorm1d is brittle on tiny batch sizes → swap for GroupNorm/InstanceNorm
• No dropout / weight norm → may over-fit very small datasets
─────────────────────────────────────────────────────────────────────────────
CLASS: UNet1D
─────────────
Args
  depth (int) – encoder/decoder levels (default 4)
  base  (int) – channels in first encoder block (default 16)

Channel schedule (depth 4, base 16)
  Encoder : 1 → 16 → 32 → 64 → 128
  Decoder : 128 → 64 → 32 → 16 → 1

Forward algorithm
  # Encoder
  for blk in enc:
      x = blk(x)               # residual convs
      skips.append(x)          # save for skip connection
      x = MaxPool1d(2)(x)      # halve sequence length

  # Decoder
  for deconv, blk in paired(deconvs, dec_blocks):
      x = deconv(x)            # double length, half channels
      if x.shape[-1] != skips[-1].shape[-1]:
          x = F.pad(x, (0,1))  # right-pad if input length was odd
      x = blk(torch.cat([x, skips.pop()], dim=1))

  return out_conv(x)           # 1×1 → single-channel output

Design rationale
• ConvTranspose1d(4, stride 2, pad 1) inverts 2× pooling exactly
• Skip concatenation preserves high-frequency detail
• Residual blocks in decoder halve channels back to skip size

Strengths
✓ Fully convolutional ⇒ arbitrary input length
✓ Skip paths retain detail; good for denoising/segmentation
✓ Handles odd lengths gracefully

Limitations / improvements
• Replace BatchNorm with GroupNorm(num_groups=8) for small batches
• Add dropout or stochastic depth for regularisation
• Use up-sample + Conv1d instead of ConvTranspose1d to avoid checkerboard artefacts
• Register MaxPool1d once (`self.pool`) instead of instantiating each call
• Expose deepest latent via `return_latent` flag for extra losses

Typical usage
─────────────
>>> net = UNet1D(depth=4, base=16)
>>> x   = torch.randn(8, 1, 4096)     # (batch, chan, length)
>>> y   = net(x)                      # y.shape → (8, 1, 4096)
>>> loss = F.mse_loss(y, target)

─────────────────────────────────────────────────────────────────────────────
"""



import torch
import torch.nn.functional as F

# ── Residual double-conv block ─────────────────────────────
class Block(torch.nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv1d(c_in, c_out, kernel_size=9, padding=4),
            torch.nn.BatchNorm1d(c_out),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(c_out, c_out, kernel_size=9, padding=4),
            torch.nn.BatchNorm1d(c_out),
            torch.nn.ReLU(inplace=True),
        )
        # Skip connection (1x1 conv if channel dim changes)
        self.skip = torch.nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else torch.nn.Identity()

    def forward(self, x):
        return self.body(x) + self.skip(x)

# ── 1D U-Net Autoencoder ───────────────────────────────────
class UNet1D(torch.nn.Module):
    """
    1D U-Net autoencoder. `depth` controls number of down/up sampling layers.
    `base` is the number of filters in the first layer (doubles each down-step).
    """
    def __init__(self, depth=4, base=16):
        super().__init__()
        self.enc = torch.nn.ModuleList()
        self.dec = torch.nn.ModuleList()
        c = 1  # input channels
        # Encoder: down-sampling with residual conv blocks
        for d in range(depth):
            c_next = base * (2 ** d)
            self.enc.append(Block(c, c_next))
            c = c_next
        # Decoder: up-sampling with transposed conv + skip concatenation
        for d in reversed(range(depth)):
            c_skip = base * (2 ** d)           # channels from corresponding encoder skip
            # Transposed conv (upsample) halves the channels
            self.dec.append(torch.nn.ConvTranspose1d(c, c // 2, kernel_size=4, stride=2, padding=1))
            # Residual block on concatenated skip + upsampled features
            self.dec.append(Block((c // 2) + c_skip, c_skip))
            c = c_skip
        # Final 1x1 conv to reconstruct 1-channel output
        self.out = torch.nn.Conv1d(base, 1, kernel_size=1)

    def forward(self, x):
        skips = []
        # Encoder: apply each conv block then downsample by 2
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = torch.nn.MaxPool1d(kernel_size=2)(x)
        # Decoder: upsample and merge with corresponding skip connections
        for deconv, blk in zip(self.dec[0::2], self.dec[1::2]):
            x = deconv(x)
            skip = skips.pop()
            # Pad if needed to match skip length (for odd-length cases)
            if x.shape[-1] != skip.shape[-1]:
                x = F.pad(x, (0, 1))
            # Concatenate skip connection and apply conv block
            x = blk(torch.cat([x, skip], dim=1))
        return self.out(x)
