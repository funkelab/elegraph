import matplotlib.pyplot as plt
import numpy as np
import random
import zarr
from skimage import data
from skimage import filters
import tifffile as tiff
import glob
import pandas as pd
import numcodecs
import math
import operator
# note: napari coordinates is (z,y,x)

# we want to shuffle filenames WHILE ensuring that the corresponding CSV numpy array is at the SAME INDEX

# initialize zarr containers
#OHHHHH so these save your data locally even when you close the program. that's why you kept seeing data already inside
ztrain =zarr.open('train.zarr') #, 'a'
zval = zarr.open('val.zarr')
ztest = zarr.open('test.zarr')
train_count = 0
val_count = 0
test_count = 0

#filenames = sorted(glob.glob('/Users/ethantam/Desktop/Raw/RegA/*.tif'))
filenames = sorted(glob.glob('/groups/funke/home/tame/regA*.tif'))

# randomize order of volumes
random.shuffle(filenames)

# declare number of volumes in training, validation, and testing sets (80-10-10)
num_train = round(0.8 * len(filenames))
num_val = round(0.1 * len(filenames))
num_test = len(filenames) - num_train - num_val

#csv_filenames = sorted(glob.glob('/Users/ethantam/Desktop/Raw/SeamCellCoordinates/*.csv'))
csv_filenames = sorted(glob.glob('/groups/funke/home/tame/seamcellcoordinates/*.csv'))

d, h, w = tiff.imread(filenames[0]).shape

# initialize a certain key and set aside memory
ztrain.require_dataset('raw', 
                     shape=(num_train, d, h, w),
                     chunks=(1, d, h, w),
                     compression='gzip',
                     dtype=np.float32)

zval.require_dataset('raw', 
                     shape=(num_val, d, h, w),
                     chunks=(1, d, h, w),
                     compression='gzip',
                     dtype=np.float32)

ztest.require_dataset('raw', 
                     shape=(num_test, d, h, w),
                     chunks=(1, d, h, w),
                     compression='gzip',
                     dtype=np.float32)

# shape: (num elements, rows of each elem, columns of each elem)
ztrain.require_dataset('gt', 
                     shape=(num_train, 22, 4),
                     chunks=(1, 22, 4),
                     compression='gzip',
                     dtype=object,
                     object_codec=numcodecs.JSON())

zval.require_dataset('gt', 
                     shape=(num_val, 22, 4),
                     chunks=(1, 22, 4),
                     compression='gzip',
                     dtype=object,
                     object_codec=numcodecs.JSON())

ztest.require_dataset('gt', 
                     shape=(num_test, 22, 4),
                     chunks=(1, 22, 4),
                     compression='gzip',
                     dtype=object,
                     object_codec=numcodecs.JSON())

ztrain.require_dataset('gt_rep', 
                     shape=(num_train, d, h, w),
                     chunks=(1, d, h, w),
                     compression='gzip',
                     dtype=np.float32)

zval.require_dataset('gt_rep', 
                     shape=(num_val, d, h, w),
                     chunks=(1, d, h, w),
                     compression='gzip',
                     dtype=np.float32)

ztest.require_dataset('gt_rep', 
                     shape=(num_test, d, h, w),
                     chunks=(1, d, h, w),
                     compression='gzip',
                     dtype=np.float32)

# this dictionary maps the ID of a volume to a list containing which set that volume belongs to and what index it is at

numToSetandIndex = {}
# loop to go through the image
for i, filename in enumerate(filenames):
    # filename[-6:-4]] grabs the ID of the volume
    im = tiff.imread(filename)
    print(i)
    if i < num_train:
        ztrain['raw'][i] = im
        numToSetandIndex[filename[-6:-4]] = ["train", train_count]
        train_count += 1
    elif i < num_train + num_val:
        zval['raw'][val_count] = im
        numToSetandIndex[filename[-6:-4]] = ["val", val_count]
        val_count += 1
    else:
        ztest['raw'][test_count] = im
        numToSetandIndex[filename[-6:-4]] = ["test", test_count]
        test_count += 1

def normalize_dict_in_place(d):
    # min-max normalization
    # normalized distance makes indexes near the center have higher values, and indexes farther away lower
    min_distance = min(d.values())
    max_distance = max(d.values())
    for point, distance in d.items():
        normalized_value = 1 - ((distance - min_distance) / (max_distance - min_distance))
        d[point] = normalized_value

def create_gaussian_dist(ground_rep, z, y, x, cell_radius=3):
    print("hello")
    center = np.array([z,y,x])
    coordToDist = {}
    # find distances from indexes only cell_radius away to the center
    for i in range(ground_rep.shape[0]):
        for j in range(ground_rep.shape[1]):
            for k in range(ground_rep.shape[2]):
                distance = math.dist([i,j,k], center)
                if distance <= cell_radius:
                    # print((i,j,k), distance)
                    coordToDist[(i,j,k)] = distance
    # normalize distances at each coordinate
    normalize_dict_in_place(coordToDist)
    #print(coordToDist)
    # insert values into the ground_rep at their specified indexes
    for key in coordToDist.keys():
        z,y,x = key
        #not 100% sure, but I want to average the values if multiple seam cells overlap?
        if ground_rep[z][y][x] > 0:
            ground_rep[z][y][x] = (ground_rep[z][y][x] + coordToDist[key]) / 2
        else:
            ground_rep[z][y][x] = coordToDist[key]

# loop to go through the pandas dataframe storing coordinates                
for filename in csv_filenames:
    print("slay")
    df = pd.read_csv(filename)[['name','x_voxels','y_voxels', 'z_voxels']].to_numpy()
     # add two rows for the additional seam cells if missing them
    if df.shape[0] == 20:
        qL = np.array(['QL', None, None, None])
        qR = np.array(['QR', None, None, None])
        df = np.vstack([df, qL])
        df = np.vstack([df, qR])
    ground_truth_rep = np.zeros((d, h, w))
    for cell in df:
        if cell[1] is None:
            continue
        z = round(cell[1])
        y = round(cell[2])
        x = round(cell[3])
        ground_truth_rep[z][y][x] = 1
        create_gaussian_dist(ground_truth_rep, z, y, x)
    print(df.shape)

    set_group = numToSetandIndex[filename[-6:-4]][0]
    index = numToSetandIndex[filename[-6:-4]][1]
    if set_group == "train":
        ztrain['gt'][index] = df
        ztrain['gt_rep'][index] = ground_truth_rep
    elif set_group == "val":
        zval['gt'][index] = df
        zval['gt_rep'][index] = ground_truth_rep
    elif set_group == "test":
        ztest['gt'][index] = df
        ztest['gt_rep'][index] = ground_truth_rep


# create u-net to identify coordinates of seam cells

import torch
from funlib.learn.torch.models import UNet, ConvPass

unet = UNet(
  in_channels=3,
  num_fmaps=4,
  fmap_inc_factor=2,
  downsample_factors=[[2, 2], [2, 2]],
  kernel_size_down=[[[3, 3], [3, 3]]]*3,
  kernel_size_up=[[[3, 3], [3, 3]]]*2,
  padding='same')

loss = torch.nn.BCELoss()


    