from val_parameters import (
    raw_filename,
    experiment_name,
    checkpoint_path,
    in_channels,
    gaussian_sigma,
    num_iterations,
)

import gunpowder as gp
import os
from funlib.learn.torch.models import UNet, ConvPass
import zarr
import torch
from models import Model
from tqdm import tqdm
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import numpy as np
from skimage.filters import threshold_otsu


def get_predicted_segmentation(prediction, min_size=100):
    thresh = threshold_otsu(prediction)
    coords = peak_local_max(
        prediction, footprint=np.ones((3, 3, 3)), labels=prediction > thresh
    )
    mask = np.zeros(prediction.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-prediction, markers, mask=prediction > thresh)
    ids = np.unique(labels)[1:]
    for id in ids:
        z, y, x = np.where(labels == id)
        print(len(z), id)
        if len(z) <= min_size:
            labels[labels == id] = 0
    return labels


def validate(num_iterations):
    if not os.path.exists(experiment_name + "/inference/"):
        os.makedirs(experiment_name + "/inference/")

    # create model
    unet = UNet(
        in_channels=in_channels,
        num_fmaps=6,
        fmap_inc_factor=4,
        downsample_factors=[[2, 2, 2], [2, 2, 2]],
        kernel_size_down=[[[3, 3, 3], [3, 3, 3]]] * 3,
        kernel_size_up=[[[3, 3, 3], [3, 3, 3]]] * 2,
        padding="valid",
    )

    unet_custom = Model(unet)
    model = torch.nn.Sequential(
        unet_custom,
        ConvPass(6, 1, [(1, 1, 1)], activation="Sigmoid"),
    )

    # replace state of weight with pre-trained model weights
    state = torch.load(checkpoint_path)
    model.load_state_dict(state["model_state_dict"], strict=True)

    # set in eval mode
    model.eval()

    # get voxel size
    voxel_size = gp.Coordinate(zarr.open(raw_filename, "r")["raw"].attrs["resolution"])

    # get scan tile size
    input_shape = gp.Coordinate((1, 128, 128, 128))
    output_shape = gp.Coordinate((1, 88, 88, 88))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    raw = gp.ArrayKey("VAL_RAW")
    seam_cells = gp.GraphKey("VAL_SEAM_CELLS")
    seam_cell_blobs = gp.ArrayKey("VAL_SEAM_CELL_BLOBS")
    prediction = gp.ArrayKey("VAL_PREDICTION")
    raw_source = gp.ZarrSource(raw_filename, {raw: "raw"}) + gp.Pad(raw, None)
    seam_cell_source = gp.CsvPointsSource(
        "val.csv", seam_cells, ndims=4, scale=voxel_size
    )
    combined_source = (raw_source, seam_cell_source) + gp.MergeProvider()
    stack = gp.Stack(1)

    pipeline = (
        combined_source
        + gp.RandomLocation(ensure_nonempty=seam_cells)
        # + gp.IntensityAugment(
        #    raw, scale_min=1.1, scale_max=1.5, shift_min=0.1, shift_max=0.5
        # )
        + gp.RasterizeGraph(
            seam_cells,
            seam_cell_blobs,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=(0.01, gaussian_sigma, gaussian_sigma, gaussian_sigma),
                mode="peak",
            ),
        )
        + stack
        + gp.torch.Predict(
            model,
            inputs={"input": raw},
            outputs={0: prediction},
            array_specs={prediction: gp.ArraySpec(voxel_size=voxel_size)},
        )
        + gp.Snapshot(
            {
                raw: "raw",
                seam_cells: "seam_cells",
                seam_cell_blobs: "target",
                prediction: "prediction",
            },
            every=1,
            output_dir=os.path.join("experiments", experiment_name, "inference"),
            output_filename="test.zarr",
        )
    )

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(seam_cells, output_size)
    request.add(seam_cell_blobs, output_size)
    request.add(prediction, output_size)

    with gp.build(pipeline):
        for i in tqdm(range(num_iterations)):
            batch = pipeline.request_batch(request)
            get_predicted_segmentation(
                batch[prediction].data[0, 0]
            ) #TODO (compute how well we are doing ...)


validate(num_iterations)
