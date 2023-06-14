import zarr
import glob
import tifffile
import numpy as np
from tqdm import tqdm

# create zarr container
img_data = zarr.open("img_data.zarr", "a")

# use this for cluster
regA_filenames = sorted(glob.glob('/groups/funke/home/tame/RegA/*.tif'))
regB_filenames = sorted(glob.glob('/groups/funke/home/tame/RegB/*.tif'))

# instantiate 5-dim numpy array
all_img_data = np.zeros((2,96,308,425,325), dtype=np.float32) 
# all volumes from both channels from the same sequence

# insert volumes as matrices into the array
for filename in tqdm(regA_filenames):
    frame = int(filename.split(".")[0].split("_")[-1])
    print(frame)
    # raw data is uint16, divide by 2**16 to normalize
    all_img_data[0, frame] = tifffile.imread(filename)

for filename in tqdm(regB_filenames):
    frame = int(filename.split(".")[0].split("_")[-1])
    # raw data is uint16, divide by 2**16 to normalize
    all_img_data[1, frame] = tifffile.imread(filename)

min_int_0 = np.min(all_img_data[0])
max_int_0 = np.max(all_img_data[0])
scaled_max_int_0 = max_int_0*1.1

min_int_1 = np.min(all_img_data[1])
max_int_1 = np.max(all_img_data[1])
scaled_max_int_1 = max_int_1*1.1

print("Min intensity is {}, Max Intensity is {}".format(min_int_0, max_int_0))

all_img_data[0] = (all_img_data[0]-min_int_0)/(scaled_max_int_0-min_int_0)
all_img_data[1] = (all_img_data[1]-min_int_1)/(scaled_max_int_1-min_int_1)

print(f"Mean intensity: {all_img_data.mean()}")

# set array to zarr container
img_data['raw'] = all_img_data
img_data['raw'].attrs['resolution'] = (5,1625,1625,1625)
