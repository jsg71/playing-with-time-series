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
