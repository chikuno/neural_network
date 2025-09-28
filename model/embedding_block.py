import torch
import torch.nn as nn

class ConvGLUBlock(nn.Module):
    """Optional convolutional + GLU gating + projection block to enrich local context
    before recurrent processing.
    Input: (B, S, E)
    Output: (B, S, E)
    Configurable channels and kernel size. Uses residual + LayerNorm.
    """
    def __init__(self, embed_dim: int, channels: int = 512, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = max(1, int(kernel_size))
        # Double channels for GLU split
        self.conv = nn.Conv1d(embed_dim, 2*channels, kernel_size=self.kernel_size, padding=self.kernel_size//2)
        self.proj = nn.Linear(channels, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            return x
        # x: (B,S,E) -> conv expects (B,E,S)
        x_in = x
        x_t = x.transpose(1,2)
        conv_out = self.conv(x_t)  # (B,2C,S)
        c, gate = conv_out.chunk(2, dim=1)
        gated = c * torch.sigmoid(gate)  # (B,C,S)
        gated = gated.transpose(1,2)  # (B,S,C)
        proj = self.proj(gated)  # (B,S,E)
        proj = self.act(proj)
        out = self.ln(proj + x_in)
        return out
