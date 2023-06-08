import gunpowder as gp
import zarr
import torch
import glob
import tifffile
import numpy as np
import random
import csv
import pandas as pd
from funlib.learn.torch.models import UNet, ConvPass

# create zarr container
img_data = zarr.open("img_data")

# use this for local purposes
regA_filenames = sorted(glob.glob('/Users/ethantam/Desktop/Raw/RegA/*.tif'))
regB_filenames = sorted(glob.glob('/Users/ethantam/Desktop/Raw/RegB/*.tif'))
csv_filenames = glob.glob('/Users/ethantam/Desktop/Raw/SeamCellCoordinates/*.csv')
random.shuffle(csv_filenames)

d, h, w = tifffile.imread(regA_filenames[0]).shape

# make large csv file containing all seam cells
with open('all_seam_cells.csv', 'w') as f:
    count_id = 0
    count_time = 0
    writer = csv.writer(f)
    writer.writerow(["time", "z", "y", "x", "name", "id"])
    for file in csv_filenames:
        with open(file, 'r') as g:
            reader = csv.reader(g)
            for row in reader:
                if row[1] == 'x_voxels':
                    continue
                time = count_time
                z = row[3]
                y = row[2]
                x = row[1]
                name = row[0]
                id = count_id
                count_id += 1
                count_time += 5
                writer.writerow([time, z, y, x, name, id])
            count_time = 0

# shuffle large csv file
df = pd.read_csv("all_seam_cells.csv")
shuffled_df = df.sample(frac=1).reset_index(drop=True)
shuffled_df.to_csv('all_seam_cells.csv', index=False)

# split large csv file into 3 csv files (80% train, 10% validation, 10% test)
df = pd.read_csv("all_seam_cells.csv")
num_train = round(0.8 * len(df))
num_val = round(0.1 * len(df))
num_test = len(df) - num_train - num_val
train_df = df[:num_train]
val_df = df[num_train:num_train+num_val]
test_df = df[num_train+num_val:]
train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)
test_df.to_csv('test.csv', index=False)

# use this for cluster
# regA_filenames = sorted(glob.glob('/groups/funke/home/tame/regA*.tif'))
# regB_filenames = sorted(glob.glob('/groups/funke/home/tame/regB*.tif'))
# csv_filenames = sorted(glob.glob('/groups/funke/home/tame/seamcellcoordinates/*.csv'))

# make sure to use cluster for this!
all_img_data = np.zeros((2,76,308,425,325)) # all volumes from both channels from the same sequence

for i in range(len(regA_filenames)):
    all_img_data[0,i] = tifffile.imread(regA_filenames[i])
    all_img_data[1,i] = tifffile.imread(regB_filenames[i])

img_data['raw'] = all_img_data

unet = UNet(
  in_channels=3, 
  num_fmaps=6,
  fmap_inc_factor=4,
  downsample_factors=[[2, 2, 2], [2, 2, 2]], 
  kernel_size_down=[[[3, 3, 3], [3, 3, 3]]]*3, # hm this should prob be *2 if we're following the 128x128x128 input to 88x88x88 output
  kernel_size_up=[[[3, 3, 3], [3, 3, 3]]]*2, # does this have one less convolutional layer to avoid losing more information during upsampling? 
  padding='valid') 

model = torch.nn.Sequential(
    unet,
    ConvPass(4,1,[(1,1,1)], activation=None), 
    torch.nn.Sigmoid() 
)

loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

def train(num_iterations):
    raw = gp.ArrayKey("RAW") 
    seam_cells = gp.GraphKey("SEAM_CELLS") # i dont get how we're storing the csv information and then randomly splitting it?
    seam_cell_blobs = gp.ArrayKey("SEAM_CELL_BLOBS")
    prediction = gp.ArrayKey("PREDICTION")
    raw_source = gp.ZarrSource("img_data.zarr", { raw: "raw"})
    seam_cell_source = gp.CsvPointsSource("train.csv", seam_cells) # does this even work for our CSV file cuz of how its set up?
    combined_source = (raw_source, seam_cell_source) + gp.MergeProvider()
    # TODO: this should be read from the zarr array
    voxel_size = zarr.open("img_data.zarr", "r")["raw"].attrs["resolution"]
    gp.Coordinate((5, 1625, 1625, 1625)) # how to determine this using the zarr container?
    
    pipeline = (
        combined_source +
        gp.RandomLocation(ensure_nonempty=seam_cells) +
        gp.IntensityAugment(raw, scale=1.1, shift=0.1) +
        gp.RasterizeGraph(
            seam_cells,
            seam_cell_blobs, # turns the graph in 'seam_cells' into an  array, sets that array to seam_cell_blobs. 
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=20000,
                mode="peak" # this makes the blob have values 0-1 on a gaussian dist
            )
        ) +
        gp.torch.Train(
            model,
            loss,
            optimizer,
            inputs={
                "input": raw
            },
            outputs={
                0: prediction
            },
            loss_inputs={
                0: prediction,
                1: seam_cell_blobs # ground truth data
            },
            array_spec={
                prediction: gp.ArraySpec(voxel_size=voxel_size)
            },
            save_every=1000 # store learned weights at every 1000th iteration
        ) +
        gp.Snapshot(
            {
                raw: "raw",
                seam_cell_blobs: "target",
                prediction: "prediction"
            },
            every=10 # we will save our resulting data ^ at every 10 batches. prob just gets saved locally in the same directory.
        )
    )
    # shape = in voxels
    input_shape = gp.Coordinate((1, 128, 128, 128))
    output_shape = gp.Coordinate((1, 88, 88, 88)) # only 3 levels? could be deeper, no? since more levels means more eonvolutions, which means better feature extractions
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size
    request = gp.BatchRequest()
    request[raw] = input_size
    request[seam_cell_blobs] = output_size
    request[prediction] = output_size
    with gp.build(pipeline):
        for i in range(num_iterations): # is this the number of epochs?
            pipeline.request_batch(request)

# to analyze accuracy, would I calculate the sum of the difference between the seam_cell_blobs and prediction?
# ik i should def find a way to open up and look at the image with the cropped area to make sure we r looking at at least one seam cell

# a validate method
def validate(num_iterations):
    pass


# a test method
def test(num_iterations):
    model.eval()
    pipeline = (
      
    )

    pass