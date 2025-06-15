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
