import gunpowder as gp
import zarr
import torch
import csv
import glob
from funlib.learn.torch.models import UNet, ConvPass

raw_filename = "/groups/funke/home/tame/elegraph/img_data.zarr"
checkpoint_path = "/groups/funke/home/tame/elegraph/models/model"

# use this for local purposes
# regA_filenames = sorted(glob.glob('/Users/ethantam/Desktop/Raw/RegA/*.tif'))
# regB_filenames = sorted(glob.glob('/Users/ethantam/Desktop/Raw/RegB/*.tif'))
# csv_filenames = glob.glob('/Users/ethantam/Desktop/Raw/SeamCellCoordinates/*.csv')

# use this for cluster
regA_filenames = sorted(glob.glob("/groups/funke/home/tame/RegA/*.tif"))

unet = UNet(
    in_channels=2,
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
        x = torch.permute(x, (1,0,2,3,4))
        return self.unet(x)

unet_custom= Model(unet)

model = torch.nn.Sequential(
    unet_custom,
    ConvPass(6, 1, [(1, 1, 1)], activation=None),  # final 1x1x1 conv pass
    torch.nn.Sigmoid(),
)

class Loss(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.loss =  torch.nn.MSELoss()
        self.path = path
        self.iter = 1
    def forward(self, prediction, target):
        prediction = torch.squeeze(prediction, 1)
        #print("prediction shape is {} and target shape is {}".format(prediction.shape, target.shape))
        l = self.loss(prediction, target)
        print("Iter: " + str(self.iter) + " Loss: " + str(l.item()))

        with open(self.path, "a") as f:
            writer = csv.writer(f, delimiter=" ")
            if self.iter == 1:
                writer.writerow(["Iteration", "Loss"]) # header
            writer.writerow([self.iter, l.item()])
            self.iter += 1
        return l

loss = Loss("train_loss.csv")

optimizer = torch.optim.Adam(model.parameters())

voxel_size = gp.Coordinate(
    zarr.open(raw_filename, "r")["raw"].attrs["resolution"]
)  # already notated in zarr container
input_shape = gp.Coordinate((1, 128, 128, 128))
output_shape = gp.Coordinate((1, 88, 88, 88))
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size