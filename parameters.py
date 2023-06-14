import gunpowder as gp
import zarr
import torch
import glob
from funlib.learn.torch.models import UNet, ConvPass

raw_filename = "/groups/funke/home/tame/elegraph/img_data.zarr"

# use this for local purposes
# regA_filenames = sorted(glob.glob('/Users/ethantam/Desktop/Raw/RegA/*.tif'))
# regB_filenames = sorted(glob.glob('/Users/ethantam/Desktop/Raw/RegB/*.tif'))
# csv_filenames = glob.glob('/Users/ethantam/Desktop/Raw/SeamCellCoordinates/*.csv')

# use this for cluster
regA_filenames = sorted(glob.glob("/groups/funke/home/tame/RegA/*.tif"))

unet = UNet(
    in_channels=3,
    num_fmaps=6,
    fmap_inc_factor=4,
    downsample_factors=[[2, 2, 2], [2, 2, 2]],
    kernel_size_down=[[[3, 3, 3], [3, 3, 3]]] * 3,
    kernel_size_up=[[[3, 3, 3], [3, 3, 3]]] * 2,
    padding="valid",
)

model = torch.nn.Sequential(
    unet,
    ConvPass(6, 1, [(1, 1, 1)], activation=None),  # final 1x1x1 conv pass
    torch.nn.Sigmoid(),
)

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

voxel_size = gp.Coordinate(
    zarr.open(raw_filename, "r")["raw"].attrs["resolution"]
)  # already notated in zarr container
input_shape = gp.Coordinate((1, 128, 128, 128))
output_shape = gp.Coordinate((1, 88, 88, 88))
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size