import torch
import torch.nn as nn
import numpy as np 
import matplotlib as plt

class PositionalEncoder(nn.Module):
    def __init__(self,dim_out:int):
        super().__init__()
        self.dim_out=dim_out
        self.n=10_000

    def forward(self,seq_len):
        pos=torch.arange(seq_len).float().view(-1,1)
        i=torch.arange(self.dim_out).float().view(1,-1)

        angle_rates=1/torch.pow(self.n,(2*(i//2))/self.dim_out)
        angle_rads=pos*angle_rates

        angle_rads[:,0::2]=torch.sin(angle_rads[:,0::2])
        angle_rads[:,1::2]=torch.cos(angle_rads[:,1::2])

        return angle_rads


pe=PositionalEncoder(dim_out=64)
encoding=pe(seq_len=100)
print(encoding.shape)
