import gunpowder as gp
import zarr
import torch
import csv
from funlib.learn.torch.models import UNet, ConvPass

raw_filename = "/groups/funke/home/lalitm/github/elegraph/img_data.zarr"
checkpoint_path = "/groups/funke/home/lalitm/github/elegraph/models/model"

unet = UNet(
    in_channels=1,  # TODO
    num_fmaps=6,
    fmap_inc_factor=4,
    downsample_factors=[[2, 2, 2], [2, 2, 2]],
    kernel_size_down=[[[3, 3, 3], [3, 3, 3]]] * 3,
    kernel_size_up=[[[3, 3, 3], [3, 3, 3]]] * 2,
    padding="valid",
)


class Model(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        # x has dimensionality B C T Z Y X
        x = x[:, 0:1, 0, ...]  # B C Z Y X
        return self.unet(x)


unet_custom = Model(unet)

model = torch.nn.Sequential(
    unet_custom,
    ConvPass(6, 1, [(1, 1, 1)], activation="Sigmoid"),  # TODO final 1x1x1 conv pass
)


class Loss(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.criterion = torch.nn.MSELoss()
        self.path = path
        self.iter = 1

    def forward(self, prediction, target):
        loss = self.criterion(prediction, target)
        print("Iter: " + str(self.iter) + " Loss: " + str(loss.item()))

        with open(self.path, "a") as f:
            writer = csv.writer(f, delimiter=" ")
            if self.iter == 1:
                writer.writerow(["Iteration", "Loss"])  # header
            writer.writerow([self.iter, loss.item()])
            self.iter += 1
        return loss


loss = Loss(path="train_loss.csv")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # TODO


voxel_size = gp.Coordinate(
    zarr.open(raw_filename, "r")["raw"].attrs["resolution"]
)  # already notated in zarr container # TODO

input_shape = gp.Coordinate((1, 128, 128, 128))  # TODO
output_shape = gp.Coordinate((1, 88, 88, 88))  # TODO
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size
batch_size = 10  # TODO
