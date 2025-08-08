import torch
import torch.nn as nn
import torch_dct as dct
import kornia
from models.utils import RDDB, DoubleAttention

torch.set_float32_matmul_precision('medium')


class FoundationISP(nn.Module):
    def __init__(self, 
        out_channels = 16
    ):
        super(FoundationISP, self).__init__()
        self.dct_embed = nn.Sequential(
            RDDB(75,                out_channels // 8),
            RDDB(out_channels // 8, out_channels // 8),
            RDDB(out_channels // 8, out_channels // 4),
            RDDB(out_channels // 4, out_channels // 4),
            RDDB(out_channels // 4, out_channels // 2),
            RDDB(out_channels // 2, out_channels // 2),
            RDDB(out_channels // 2, out_channels)
        )
        self.compress = nn.Sequential(
            RDDB(out_channels + 3,  out_channels),
            nn.Tanh()
        )
        return

    def forward(self, x):
        x = kornia.color.rgb_to_ycbcr(x)
        xp = nn.functional.pad(x, (2, 2, 2, 2), mode='reflect')
        xp = xp.contiguous().unfold(2, 5, 1).unfold(3, 5, 1).flatten(-2, -1)
        xp = dct.dct(xp.permute(0, 2, 3, 1, 4).flatten(-2, -1), norm='ortho').permute(0, 3, 1, 2)
        y = self.dct_embed(xp)
        y = torch.concat((y, x), dim = 1)
        y = self.compress(y)
        return y
    


