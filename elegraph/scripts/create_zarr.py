import glob

import numpy as np
import tifffile
import zarr
from tqdm import tqdm

# create zarr container
img_data = zarr.open("/groups/funke/home/tame/data/img_data.zarr", "a")

# use this for cluster
regA_filenames = sorted(glob.glob("/groups/funke/home/tame/data/RegA/*.tif"))
regB_filenames = sorted(glob.glob("/groups/funke/home/tame/data/RegB/*.tif"))

# instantiate 5-dim numpy array
all_img_data = np.zeros((2, 96, 308, 425, 325), dtype=np.float32)


def normalize_min_max_percentile(
    x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32
):
    """
    Percentile-based image normalization.
    Function taken from StarDist repository  https://github.com/stardist/stardist
    """
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    """
    Percentile-based image normalization.
    Function taken from StarDist repository  https://github.com/stardist/stardist
    """
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr

        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


# insert volumes as matrices into the array
for filename in tqdm(regA_filenames):
    frame = int(filename.split(".")[0].split("_")[-1])
    all_img_data[0, frame] = normalize_min_max_percentile(
        tifffile.imread(filename), 1, 99.8, axis=(0, 1, 2)
    )

for filename in tqdm(regB_filenames):
    frame = int(filename.split(".")[0].split("_")[-1])
    all_img_data[1, frame] = normalize_min_max_percentile(
        tifffile.imread(filename), 1, 99.8, axis=(0, 1, 2)
    )

# set array to zarr container
img_data["raw"] = all_img_data
img_data["raw"].attrs["resolution"] = (1, 1625, 1625, 1625)
