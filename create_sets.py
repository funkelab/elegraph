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
import json

if __name__ == "__main__":
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
    
    all_data = {}
    all_data["numToSetandIndex"] = numToSetandIndex
    all_data['d'] = d
    all_data['h'] = h
    all_data['w'] = w
    
    # save dict and (d,h,w) in json file
    with open("save_data.json", "w") as outfile:
        json.dump(all_data, outfile)
