import torch


class Model(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        # x has dimensionality B C T Z Y X
        x = x[:, 0:1, 0, ...]  # B C Z Y X
        return self.unet(x)
