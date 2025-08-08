import torch
import torch.nn as nn
import torch_dct as dct
import kornia
import torch.nn.functional as F

class RDDB(nn.Module):
    def __init__(self, 
        in_channels,
        out_channels
    ):
        super(RDDB, self).__init__()
        self.ddb1 = DDB(in_channels, out_channels)
        self.ddb2 = DDB(in_channels + out_channels, out_channels)
        self.ddb3 = DDB(in_channels + out_channels * 2, out_channels)
        return
        
    def forward(self, x):
        f1 = self.ddb1(x)
        f2 = self.ddb2(torch.concat((x, f1), dim = 1))
        f3 = self.ddb3(torch.concat((x, f1, f2), dim = 1))
        return f3


class DDB(nn.Module):
    def __init__(self, 
        in_channels,
        out_channels
    ):
        super(DDB, self).__init__()
        self.convH = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = (1, 5),
                padding = 'same',
                padding_mode = 'reflect'
            ), nn.GELU()
        )

        self.convV = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = (5, 1),
                padding = 'same',
                padding_mode = 'reflect'
            ), nn.GELU()
        )
        
        self.norm = nn.InstanceNorm2d(in_channels + (out_channels * 2))

        self.compress = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels + (out_channels * 2),
                out_channels = out_channels,
                kernel_size = 1,
                padding = 'same',
                padding_mode = 'reflect'
            )
        )
        
        return
        
    def forward(self, x):
        f = torch.concat(
            (
                x, 
                self.convH(x),
                self.convV(x)
            ), dim = 1
        )
        return self.compress(self.norm(f))
    
class DoubleAttention(nn.Module):
   
    def __init__(self, in_channels, c_m, c_n):
        
        super().__init__()
        self.c_m = c_m
        self.c_n = c_n
        self.in_channels = in_channels
        self.convA = nn.Conv2d(in_channels, c_m, kernel_size = 1)
        self.convB = nn.Conv2d(in_channels, c_n, kernel_size = 1)
        self.convV = nn.Conv2d(in_channels, c_n, kernel_size = 1)
        self.proj  = nn.Conv2d(c_m, in_channels, kernel_size = 1)

    def forward(self, x):
        b, c, h, w = x.shape
        A = self.convA(x)  
        B = self.convB(x)  
        V = self.convV(x)  
        tmpA = A.view(b, self.c_m, h * w)
        attention_maps = B.view(b, self.c_n, h * w)
        attention_vectors = V.view(b, self.c_n, h * w)
        attention_maps = F.softmax(attention_maps, dim = -1)  
        
        # step 1: feature gathering
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  
        
        # step 2: feature distribution
        attention_vectors = F.softmax(attention_vectors, dim = 1)  
        tmpZ = global_descriptors.matmul(attention_vectors)  
        tmpZ = tmpZ.view(b, self.c_m, h, w)
        tmpZ = self.proj(tmpZ)
        return tmpZ