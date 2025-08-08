import torch
import torch.nn as nn
import kornia

torch.set_float32_matmul_precision('medium')

def UNet(in_channels, out_channels):
    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 
        'unet',
        in_channels = in_channels, 
        out_channels = out_channels, 
        init_features = 4,
        pretrained = False
    )
    return model

class SSLearner(nn.Module):
    def __init__(self, 
        in_channels, out_channels
    ):
        super(SSLearner, self).__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels,      out_channels * 8, 1, padding='same', padding_mode='reflect'), nn.GELU(),
            nn.Conv2d(out_channels * 8, out_channels * 4, 1, padding='same', padding_mode='reflect'), nn.GELU(),
            nn.Conv2d(out_channels * 4, out_channels * 2, 1, padding='same', padding_mode='reflect'), nn.GELU(),
            nn.Conv2d(out_channels * 2, out_channels,     1, padding='same', padding_mode='reflect'), nn.Sigmoid()
        )
        return

    def forward(self, x, mode):
        y = torch.clip(self.compress(torch.concat((x, mode), dim = 1)), 0., 1.)
        return torch.clip(y, 0., 1.)


