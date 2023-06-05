import zarr
import numpy as np
import math
import pandas as pd
import json
import sys

# read json file
with open('save_data.json', 'r') as openfile:
    all_data = json.load(openfile)

numToSetandIndex = all_data["numToSetandIndex"]
d = all_data["d"]
h = all_data["h"]
w = all_data["w"]

filename = sys.argv[1]

ztrain =zarr.open('train.zarr')
zval = zarr.open('val.zarr')
ztest = zarr.open('test.zarr')

def normalize_dict_in_place(d):
    # min-max normalization
    # normalized distance makes indexes near the center have higher values, and indexes farther away lower
    min_distance = min(d.values())
    max_distance = max(d.values())
    for point, distance in d.items():
        normalized_value = 1 - ((distance - min_distance) / (max_distance - min_distance))
        d[point] = normalized_value

def create_gaussian_dist(ground_rep, z, y, x, cell_radius=3):
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
