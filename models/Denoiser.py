import torch
from torch import nn
import torch.nn.functional as F
import kornia
import torch_dct as dct
from models.utils import RDDB, DoubleAttention

torch.set_float32_matmul_precision('medium')

class Model(nn.Module):
    def __init__(self, out_channels):
        super(Model, self).__init__()
        self.ff = torch.jit.load('./foundation_ckpt/foundation.pth')

        self.dct_blend_y =  nn.Sequential(
            nn.Linear(49, 25), 
            nn.Linear(25, 9), 
            nn.Linear(9, 3)
        )
        self.dct_blend_cb = nn.Sequential(
            nn.Linear(49, 16), 
            nn.Linear(16, 9), 
            nn.Linear(9, 3)
        )
        self.dct_blend_cr = nn.Sequential(
            nn.Linear(49, 16), 
            nn.Linear(16, 9), 
            nn.Linear(9, 3)
        )


        self.compress_y = nn.Sequential(
            RDDB(out_channels + 3, 16),
            DoubleAttention(16, 4, 4),
            RDDB(16, 3)
        )
        self.compress_cb = nn.Sequential(
            RDDB(out_channels + 3, 16),
            DoubleAttention(16, 4, 4),
            RDDB(16, 3)
        )
        self.compress_cr = nn.Sequential(
            RDDB(out_channels + 3, 16),
            DoubleAttention(16, 4, 4),
            RDDB(16, 3)
        )

        self.window_Size = 21
        self.rgb2ycbcr = kornia.color.RgbToYcbcr()
        self.ycbcr2rgb = kornia.color.YcbcrToRgb()
        
        
        self.weights_reg  = nn.Sequential(
            nn.Linear(
                self.window_Size ** 2, 
                self.window_Size ** 2
            )
        )

        self.delta1 = RDDB(3, 16)
        self.delta2 = RDDB(16, 8)
        self.delta3 = RDDB(24, 8)
        self.delta4 = RDDB(32, 8)
        self.delta5 = nn.Sequential(
            RDDB(40, 8),
            nn.Conv2d(8, 3, 1, padding = 'same', padding_mode = 'reflect')
        )

        return

    def forward(self, x):
        
        x = torch.clip(x / 255.0, 0.0, 1.0)
        ff = self.ff(x)
        
        ycbcr = self.rgb2ycbcr(x)
        
        y_dct_compress  = self.compress_y(
            torch.concat(
                (
                    self._dct5x5(
                        ycbcr[:, 0:1], 
                        self.dct_blend_y
                    ),  
                    ff
                ), 
                dim = 1
            )
        )
        
        cb_dct_compress = self.compress_cb(
            torch.concat(
                (
                    self._dct5x5(
                        ycbcr[:, 1:2], 
                        self.dct_blend_cb
                    ), 
                    ff
                ), 
                dim = 1
            )
        )
        
        cr_dct_compress = self.compress_cr(
            torch.concat(
                (
                    self._dct5x5(
                        ycbcr[:, 2:3], 
                        self.dct_blend_cr
                    ), 
                    ff
                ), 
                dim = 1
            )
        )

        stage_1 = torch.concat(
            (
                self._nlm(
                    ycbcr[:, 0:1], 
                    y_dct_compress,  
                    self.window_Size, 
                    self.weights_reg
                ),
                self._nlm(
                    ycbcr[:, 1:2], 
                    cb_dct_compress, 
                    self.window_Size, 
                    self.weights_reg
                ),
                self._nlm(
                    ycbcr[:, 2:3], 
                    cr_dct_compress, 
                    self.window_Size, 
                    self.weights_reg
                ),
            ), dim = 1
        )

        stage_1 = torch.clip(self.ycbcr2rgb(stage_1), 0., 1.) 

        stage_2 = self._delta(
            torch.concat(
                (
                    stage_1,
                ), 
                dim = 1
            )
        ) + stage_1

        stage_1 = torch.clip(stage_1, 0., 1.) * 255.
        stage_2 = torch.clip(stage_2, 0., 1.) * 255.

        # stage_2 = torch.clip(self.ycbcr2rgb(stage_2), 0., 1.) * 255.
        return stage_1, stage_2

    def _nlm(self, y, feature_map, window_size, weights_reg):
        half_window_size = window_size // 2
        padding = (half_window_size, half_window_size, half_window_size, half_window_size)
        y_p = nn.functional.pad(y, padding, mode='reflect')
        feature_map_p = nn.functional.pad(feature_map, padding, mode='reflect')
        y_patches = y_p.contiguous().unfold(2, window_size, 1).unfold(3, window_size, 1).flatten(4, 5)
        feature_map_patches = feature_map_p.contiguous().unfold(2, window_size, 1).unfold(3, window_size, 1).flatten(4, 5)
        weights = torch.mean(
            torch.square(
                feature_map_patches - feature_map[..., None]
            ), dim=1
        ).contiguous()
        weights = weights_reg(weights)
        weights = nn.functional.softmax(-weights, dim=3)[:, None, ...]
        return torch.sum(y_patches * weights, 4)
    
    def _dct5x5(self, x, dct_blend):
        pad = (3, 3, 3, 3)
        x_padded = nn.functional.pad(x, pad, mode='reflect')
        x_padded_patches = x_padded.contiguous().unfold(2, 7, 1).unfold(3, 7, 1)
        x_padded_patches = dct.dct_2d(x_padded_patches, norm='ortho')
        x_padded_patches = dct_blend(x_padded_patches.flatten(-2, -1))
        x_padded_patches = x_padded_patches[:, 0].permute(0, 3, 1, 2)
        return x_padded_patches
    
    def _delta(self, x):
        x1 = self.delta1(x)
        x2 = self.delta2(x1)
        x3 = self.delta3(torch.concat((x1, x2), dim = 1))
        x4 = self.delta4(torch.concat((x1, x2, x3), dim = 1))
        x5 = self.delta5(torch.concat((x1, x2, x3, x4), dim = 1))
        return x5
    
    def get_stage1_params(self):
        params = []
        params.append(self.ff.parameters())
        params.append(self.dct_blend_y.parameters())
        params.append(self.dct_blend_cb.parameters())
        params.append(self.dct_blend_cr.parameters())
        params.append(self.compress_y.parameters())
        params.append(self.compress_cb.parameters())
        params.append(self.compress_cr.parameters())
        params.append(self.weights_reg.parameters())
        return params
        
    def get_stage2_params(self):
        params = []
        params.append(self.delta1.parameters())
        params.append(self.delta2.parameters())
        params.append(self.delta3.parameters())
        params.append(self.delta4.parameters())
        params.append(self.delta5.parameters())
        return params
    
    

# Good Model - 27.6 psnr
# class Model(nn.Module):
#     def __init__(self, out_channels):
#         super(Model, self).__init__()
#         self.ff = torch.jit.load('./foundation_ckpt/foundation.pth')
#         self.compressY = nn.Sequential(
#             nn.Conv2d(out_channels + 3,  16, 1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(16,                8,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(8,                 4,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(4,                 3,  1, padding='same', padding_mode='reflect'), nn.GELU()
#         )
#         self.compressU = nn.Sequential(
#             nn.Conv2d(out_channels + 3,  16, 1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(16,                8,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(8,                 4,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(4,                 3,  1, padding='same', padding_mode='reflect'), nn.GELU()
#         )
#         self.compressV = nn.Sequential(
#             nn.Conv2d(out_channels + 3,  16, 1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(16,                8,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(8,                 4,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(4,                 3,  1, padding='same', padding_mode='reflect'), nn.GELU()
#         )

#         self.window_Size = 17
#         self.rgb2ycbcr = kornia.color.RgbToYcbcr()
#         self.ycbcr2rgb = kornia.color.YcbcrToRgb()

#         self.dct_blend_y =  nn.Sequential(
#             nn.Linear(25, 16), 
#             nn.Linear(16, 9), 
#             nn.Linear(9, 3)
#         )
#         self.dct_blend_cb = nn.Sequential(
#             nn.Linear(25, 16), 
#             nn.Linear(16, 9), 
#             nn.Linear(9, 3)
#         )
#         self.dct_blend_cr = nn.Sequential(
#             nn.Linear(25, 16), 
#             nn.Linear(16, 9), 
#             nn.Linear(9, 3)
#         )
        
#         self.weights_reg = nn.Sequential(nn.Linear(self.window_Size ** 2, self.window_Size ** 2), nn.ReLU())

#         return

#     def forward(self, x):
#         x = x / 255.
#         x = torch.clip(x, 0.0, 1.0)
#         ff = self.ff(x)
#         ycbcr = self.rgb2ycbcr(x)
#         y_dct  = self._dct5x5(ycbcr[:, 0:1], self.dct_blend_y)
#         cb_dct = self._dct5x5(ycbcr[:, 1:2], self.dct_blend_cb)
#         cr_dct = self._dct5x5(ycbcr[:, 2:3], self.dct_blend_cr)

#         ycbcr = torch.concat(
#             (
#                 self._nlm(ycbcr[:, 0:1], self.compressY(torch.concat((y_dct,  ff), dim = 1)), self.window_Size),
#                 self._nlm(ycbcr[:, 1:2], self.compressU(torch.concat((cb_dct, ff), dim = 1)), self.window_Size),
#                 self._nlm(ycbcr[:, 2:3], self.compressV(torch.concat((cr_dct, ff), dim = 1)), self.window_Size),
#             ), dim = 1
#         )
        
#         out = torch.clip(self.ycbcr2rgb(ycbcr), 0., 1.) * 255.
#         return out

#     def _nlm(self, y, feature_map, window_size):
#         half_window_size = window_size // 2
#         padding = (half_window_size, half_window_size, half_window_size, half_window_size)
#         y_p = nn.functional.pad(y, padding, mode='reflect')
#         feature_map_p = nn.functional.pad(feature_map, padding, mode='reflect')
#         y_patches = y_p.contiguous().unfold(2, window_size, 1).unfold(3, window_size, 1).flatten(4, 5)
#         feature_map_patches = feature_map_p.contiguous().unfold(2, window_size, 1).unfold(3, window_size, 1).flatten(4, 5)
#         weights = torch.mean(
#             torch.square(
#                 feature_map_patches - feature_map[..., None]
#             ), dim=1
#         ).contiguous()
#         weights = self.weights_reg(weights)
#         weights = nn.functional.softmax(-weights, dim=3)[:, None, ...]
#         return torch.sum(y_patches * weights, 4)
    
#     def _dct5x5(self, x, dct_blend):
#         pad = (2, 2, 2, 2)
#         x_padded = nn.functional.pad(x, pad, mode='reflect')
#         x_padded_patches = x_padded.contiguous().unfold(2, 5, 1).unfold(3, 5, 1)
#         x_padded_patches = dct.dct_2d(x_padded_patches, norm='ortho')
#         x_padded_patches = dct_blend(x_padded_patches.flatten(-2, -1))
#         x_padded_patches = x_padded_patches[:, 0].permute(0, 3, 1, 2)
#         return x_padded_patches


# class Model(nn.Module):
#     def __init__(self, out_channels):
#         super(Model, self).__init__()
#         self.ff = torch.jit.load('./foundation_ckpt/foundation.pth')
#         self.compress_y = nn.Sequential(
#             nn.Conv2d(out_channels + 3,  16, 1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(16,                8,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(8,                 4,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(4,                 2,  1, padding='same', padding_mode='reflect'), nn.GELU()
#         )
#         self.compress_u = nn.Sequential(
#             nn.Conv2d(out_channels + 3,  16, 1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(16,                8,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(8,                 4,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(4,                 2,  1, padding='same', padding_mode='reflect'), nn.GELU()
#         )
#         self.compress_v = nn.Sequential(
#             nn.Conv2d(out_channels + 3,  16, 1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(16,                8,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(8,                 4,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(4,                 2,  1, padding='same', padding_mode='reflect'), nn.GELU()
#         )

#         self.window_Size = 17
#         self.rgb2ycbcr = kornia.color.RgbToYcbcr()
#         self.ycbcr2rgb = kornia.color.YcbcrToRgb()

#         self.dct_blend_y =  nn.Sequential(
#             nn.Linear(25, 16), 
#             nn.Linear(16, 9), 
#             nn.Linear(9, 3)
#         )
#         self.dct_blend_cb = nn.Sequential(
#             nn.Linear(25, 16), 
#             nn.Linear(16, 9), 
#             nn.Linear(9, 3)
#         )
#         self.dct_blend_cr = nn.Sequential(
#             nn.Linear(25, 16), 
#             nn.Linear(16, 9), 
#             nn.Linear(9, 3)
#         )
        
#         self.weights_reg = nn.Sequential(nn.Linear(self.window_Size ** 2, self.window_Size ** 2), nn.ReLU())

#         self.recover_y = nn.Sequential(
#             nn.Conv2d(4,  4,  3, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(4,  8,  3, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(8,  8,  3, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(8,  8,  3, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(8,  4,  1, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(4,  8,  3, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(8,  8,  3, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(8,  4,  3, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(4,  2,  3, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(2,  1,  1, padding='same', padding_mode='reflect')
#         )

#         self.recover_cb = nn.Sequential(
#             nn.Conv2d(4,  4,  3, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(4,  8,  3, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(8,  8,  3, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(8,  8,  3, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(8,  4,  1, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(4,  8,  3, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(8,  8,  3, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(8,  4,  3, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(4,  2,  3, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(2,  1,  1, padding='same', padding_mode='reflect')
#         )

#         self.recover_cr = nn.Sequential(
#             nn.Conv2d(4,  4,  3, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(4,  8,  3, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(8,  8,  3, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(8,  8,  3, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(8,  4,  1, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(4,  8,  3, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(8,  8,  3, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(8,  4,  3, padding='same', padding_mode='reflect'), nn.GELU(), 
#             nn.Conv2d(4,  2,  3, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(2,  1,  1, padding='same', padding_mode='reflect')
#         )

#         return

#     def forward(self, x):
#         x = x / 255.
#         x = torch.clip(x, 0.0, 1.0)
#         ff = self.ff(x)
#         ycbcr = self.rgb2ycbcr(x)
        
#         y_dct_compress  = self.compress_y(torch.concat((self._dct5x5(ycbcr[:, 0:1], self.dct_blend_y), ff), dim = 1))
#         cb_dct_compress = self.compress_u(torch.concat((self._dct5x5(ycbcr[:, 1:2], self.dct_blend_cb), ff), dim = 1))
#         cr_dct_compress = self.compress_v(torch.concat((self._dct5x5(ycbcr[:, 2:3], self.dct_blend_cr), ff), dim = 1))

#         ycbcr_nlm = torch.concat(
#             (
#                 self._nlm(ycbcr[:, 0:1], y_dct_compress, self.window_Size),
#                 self._nlm(ycbcr[:, 1:2], cb_dct_compress, self.window_Size),
#                 self._nlm(ycbcr[:, 2:3], cr_dct_compress, self.window_Size),
#             ), dim = 1
#         )

#         ycbcr_nlm[:, 0:1] += self.recover_y(torch.concat((ycbcr_nlm[:, 0:1], ycbcr_nlm[:, 0:1] - ycbcr[:, 0:1], y_dct_compress),  dim = 1))
#         ycbcr_nlm[:, 1:2] += self.recover_cb(torch.concat((ycbcr_nlm[:, 1:2], ycbcr_nlm[:, 1:2] - ycbcr[:, 1:2], y_dct_compress), dim = 1))
#         ycbcr_nlm[:, 2:3] += self.recover_cr(torch.concat((ycbcr_nlm[:, 2:3], ycbcr_nlm[:, 2:3] - ycbcr[:, 2:3], y_dct_compress), dim = 1))
        
#         out = torch.clip(self.ycbcr2rgb(ycbcr_nlm), 0., 1.) * 255.
#         return out

    # def _nlm(self, y, feature_map, window_size):
    #     half_window_size = window_size // 2
    #     padding = (half_window_size, half_window_size, half_window_size, half_window_size)
    #     y_p = nn.functional.pad(y, padding, mode='reflect')
    #     feature_map_p = nn.functional.pad(feature_map, padding, mode='reflect')
    #     y_patches = y_p.contiguous().unfold(2, window_size, 1).unfold(3, window_size, 1).flatten(4, 5)
    #     feature_map_patches = feature_map_p.contiguous().unfold(2, window_size, 1).unfold(3, window_size, 1).flatten(4, 5)
    #     weights = torch.mean(
    #         torch.square(
    #             feature_map_patches - feature_map[..., None]
    #         ), dim=1
    #     ).contiguous()
    #     weights = self.weights_reg(weights)
    #     weights = nn.functional.softmax(-weights, dim=3)[:, None, ...]
    #     return torch.sum(y_patches * weights, 4)
    
#     def _dct5x5(self, x, dct_blend):
#         pad = (2, 2, 2, 2)
#         x_padded = nn.functional.pad(x, pad, mode='reflect')
#         x_padded_patches = x_padded.contiguous().unfold(2, 5, 1).unfold(3, 5, 1)
#         x_padded_patches = dct.dct_2d(x_padded_patches, norm='ortho')
#         x_padded_patches = dct_blend(x_padded_patches.flatten(-2, -1))
#         x_padded_patches = x_padded_patches[:, 0].permute(0, 3, 1, 2)
#         return x_padded_patches
    
    

# Good Model - 27.6 psnr
# class Model(nn.Module):
#     def __init__(self, out_channels):
#         super(Model, self).__init__()
#         self.ff = torch.jit.load('./foundation_ckpt/foundation.pth')
#         self.compressY = nn.Sequential(
#             nn.Conv2d(out_channels + 3,  16, 1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(16,                8,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(8,                 4,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(4,                 3,  1, padding='same', padding_mode='reflect'), nn.GELU()
#         )
#         self.compressU = nn.Sequential(
#             nn.Conv2d(out_channels + 3,  16, 1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(16,                8,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(8,                 4,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(4,                 3,  1, padding='same', padding_mode='reflect'), nn.GELU()
#         )
#         self.compressV = nn.Sequential(
#             nn.Conv2d(out_channels + 3,  16, 1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(16,                8,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(8,                 4,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(4,                 3,  1, padding='same', padding_mode='reflect'), nn.GELU()
#         )

#         self.window_Size = 17
#         self.rgb2ycbcr = kornia.color.RgbToYcbcr()
#         self.ycbcr2rgb = kornia.color.YcbcrToRgb()

#         self.dct_blend_y =  nn.Sequential(
#             nn.Linear(25, 16), 
#             nn.Linear(16, 9), 
#             nn.Linear(9, 3)
#         )
#         self.dct_blend_cb = nn.Sequential(
#             nn.Linear(25, 16), 
#             nn.Linear(16, 9), 
#             nn.Linear(9, 3)
#         )
#         self.dct_blend_cr = nn.Sequential(
#             nn.Linear(25, 16), 
#             nn.Linear(16, 9), 
#             nn.Linear(9, 3)
#         )
        
#         self.weights_reg = nn.Sequential(nn.Linear(self.window_Size ** 2, self.window_Size ** 2), nn.ReLU())

#         return

#     def forward(self, x):
#         x = x / 255.
#         x = torch.clip(x, 0.0, 1.0)
#         ff = self.ff(x)
#         ycbcr = self.rgb2ycbcr(x)
#         y_dct  = self._dct5x5(ycbcr[:, 0:1], self.dct_blend_y)
#         cb_dct = self._dct5x5(ycbcr[:, 1:2], self.dct_blend_cb)
#         cr_dct = self._dct5x5(ycbcr[:, 2:3], self.dct_blend_cr)

#         ycbcr = torch.concat(
#             (
#                 self._nlm(ycbcr[:, 0:1], self.compressY(torch.concat((y_dct,  ff), dim = 1)), self.window_Size),
#                 self._nlm(ycbcr[:, 1:2], self.compressU(torch.concat((cb_dct, ff), dim = 1)), self.window_Size),
#                 self._nlm(ycbcr[:, 2:3], self.compressV(torch.concat((cr_dct, ff), dim = 1)), self.window_Size),
#             ), dim = 1
#         )
        
#         out = torch.clip(self.ycbcr2rgb(ycbcr), 0., 1.) * 255.
#         return out

#     def _nlm(self, y, feature_map, window_size):
#         half_window_size = window_size // 2
#         padding = (half_window_size, half_window_size, half_window_size, half_window_size)
#         y_p = nn.functional.pad(y, padding, mode='reflect')
#         feature_map_p = nn.functional.pad(feature_map, padding, mode='reflect')
#         y_patches = y_p.contiguous().unfold(2, window_size, 1).unfold(3, window_size, 1).flatten(4, 5)
#         feature_map_patches = feature_map_p.contiguous().unfold(2, window_size, 1).unfold(3, window_size, 1).flatten(4, 5)
#         weights = torch.mean(
#             torch.square(
#                 feature_map_patches - feature_map[..., None]
#             ), dim=1
#         ).contiguous()
#         weights = self.weights_reg(weights)
#         weights = nn.functional.softmax(-weights, dim=3)[:, None, ...]
#         return torch.sum(y_patches * weights, 4)
    
#     def _dct5x5(self, x, dct_blend):
#         pad = (2, 2, 2, 2)
#         x_padded = nn.functional.pad(x, pad, mode='reflect')
#         x_padded_patches = x_padded.contiguous().unfold(2, 5, 1).unfold(3, 5, 1)
#         x_padded_patches = dct.dct_2d(x_padded_patches, norm='ortho')
#         x_padded_patches = dct_blend(x_padded_patches.flatten(-2, -1))
#         x_padded_patches = x_padded_patches[:, 0].permute(0, 3, 1, 2)
#         return x_padded_patches

# ORIGINAL IMPLEMENTATION
# class Model(nn.Module):
#     def __init__(self, out_channels):
#         super(Model, self).__init__()
#         self.ff = torch.jit.load('./foundation_ckpt/foundation.pth')
#         self.compressY = nn.Sequential(
#             nn.Conv2d(out_channels + 1,  16, 1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(16,                8,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(8,                 4,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(4,                 1,  1, padding='same', padding_mode='reflect'), nn.GELU()
#         )
#         self.compressU = nn.Sequential(
#             nn.Conv2d(out_channels + 1,  16, 1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(16,                8,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(8,                 4,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(4,                 1,  1, padding='same', padding_mode='reflect'), nn.GELU()
#         )
#         self.compressV = nn.Sequential(
#             nn.Conv2d(out_channels + 1,  16, 1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(16,                8,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(8,                 4,  1, padding='same', padding_mode='reflect'), nn.GELU(),
#             nn.Conv2d(4,                 1,  1, padding='same', padding_mode='reflect'), nn.GELU()
#         )

#         self.window_Size = 19
#         self.rgb2ycbcr = kornia.color.RgbToYcbcr()
#         self.ycbcr2rgb = kornia.color.YcbcrToRgb()

#         self.weights_reg = nn.Sequential(
#             nn.Linear(self.window_Size ** 2, self.window_Size ** 2), nn.ReLU()
#         )
#         return

#     def forward(self, x):
#         x = x / 255.
#         x = torch.clip(x, 0.0, 1.0)
#         ff = self.ff(x)
#         ycbcr = self.rgb2ycbcr(x)
#         y  = ycbcr[:, 0:1]
#         cb = ycbcr[:, 1:2]
#         cr = ycbcr[:, 2:3]

#         y = torch.concat(
#             (
#                 self._nlm(ycbcr[:, 0:1], self.compressY(torch.concat((y,  ff), dim = 1)), self.window_Size),
#                 self._nlm(ycbcr[:, 1:2], self.compressU(torch.concat((cb, ff), dim = 1)), self.window_Size),
#                 self._nlm(ycbcr[:, 2:3], self.compressV(torch.concat((cr, ff), dim = 1)), self.window_Size),
#             ), dim = 1
#         )
#         out = torch.clip(self.ycbcr2rgb(y), 0., 1.) * 255.
#         return out

#     def _nlm(self, y, feature_map, window_size):
#         half_window_size = window_size // 2
#         padding = (half_window_size, half_window_size, half_window_size, half_window_size)
#         y_p = nn.functional.pad(y, padding, mode='reflect')
#         feature_map_p = nn.functional.pad(feature_map, padding, mode='reflect')
#         y_patches = y_p.contiguous().unfold(2, window_size, 1).unfold(3, window_size, 1).flatten(4, 5)
#         feature_map_patches = feature_map_p.contiguous().unfold(2, window_size, 1).unfold(3, window_size, 1).flatten(4, 5)
#         weights = torch.mean(
#             torch.square(
#                 feature_map_patches - feature_map[..., None]
#             ), dim=1
#         ).contiguous()
#         weights = self.weights_reg(weights)
#         weights = nn.functional.softmax(-weights, dim=3)[:, None, ...]
#         return torch.sum(y_patches * weights, 4)