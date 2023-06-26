import torch


class Model(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        x = x[:, 0:2, 0, ...]
        return self.unet(x)


class UnsqueezeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.unsqueeze(x, 2)
