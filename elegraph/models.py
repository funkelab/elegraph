import torch

class Model(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        # x has dimensionality B C T Z Y X
        #x = x[:, 0:1, 0, ...]  # B C Z Y X - just first channel, eliminate time dimension
        # x = x[:, :, 0, ...]  # B C Z Y X - both channels, eliminate time dimension
        x = x[:, 0:2, 0, ...]
        return self.unet(x)

class UnsqueezeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.unsqueeze(x, 2)