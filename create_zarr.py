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
all_img_data = np.zeros((2,96,308,425,325), dtype=np.float32) # all volumes from both channels from the same sequence


# insert volumes as matrices into the array
for filename in tqdm(regA_filenames):
    frame = int(filename.split(".")[0].split("_")[-1])
    # raw data is uint16, divide by 2**16 to normalize
    all_img_data[0, frame] = tifffile.imread(filename) / 2**16

for filename in tqdm(regB_filenames):
    frame = int(filename.split(".")[0].split("_")[-1])
    # raw data is uint16, divide by 2**16 to normalize
    all_img_data[1, frame] = tifffile.imread(filename) / 2**16

print(f"Mean intensity: {all_img_data.mean()}")

# set array to zarr container
img_data['raw'] = all_img_data
img_data['raw'].attrs['resolution'] = (5,1625,1625,1625)
