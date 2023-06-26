import argparse
import os
import sys

import gunpowder as gp
import torch
import zarr
from funlib.learn.torch.models import ConvPass, UNet
from train_parameters import (
    batch_size,
    experiment_name,
    fmap_inc_factor,
    gaussian_sigma,
    gaussian_threshold,
    in_channels,
    input_dim,
    levels,
    lr,
    num_fmaps,
    num_iterations,
    output_dim,
    raw_filename,
    save_model_every,
    save_snapshot_every,
    weight_fg,
)

from criterions import Loss
from models import Model, UnsqueezeModel

parser = argparse.ArgumentParser()
parser.add_argument("test", type=str)
args = parser.parse_args()
test = args.test

# To run in the terminal, type `python train.py test-00x`
# Make sure experiments directory has `test-00x` directory
# and `train_parameters.py` before running


directory_path = "/groups/funke/home/tame/experiments/" + test
sys.path.append(directory_path)


def train(num_iterations):
    # create model
    unet = UNet(
        in_channels=in_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=[[2, 2, 2], [2, 2, 2]],
        kernel_size_down=[[[3, 3, 3], [3, 3, 3]]] * levels,
        kernel_size_up=[[[3, 3, 3], [3, 3, 3]]] * (levels - 1),
        padding="valid",
    )

    unet_custom = Model(unet)
    unsqueeze_custom = UnsqueezeModel()

    model = torch.nn.Sequential(
        unet_custom,
        ConvPass(num_fmaps, 1, [(1, 1, 1)], activation="Sigmoid"),
        unsqueeze_custom,
    )

    # create loss
    loss = Loss(
        path=os.path.join(
            "/groups/funke/home/tame/experiments/", experiment_name, "train_loss.csv"
        ),
        weight_fg=weight_fg,
        gaussian_threshold=gaussian_threshold,
    )

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # get voxel size
    voxel_size = gp.Coordinate(zarr.open(raw_filename, "r")["raw"].attrs["resolution"])
    input_shape = gp.Coordinate((1, input_dim, input_dim, input_dim))
    output_shape = gp.Coordinate((1, output_dim, output_dim, output_dim))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    # create directories
    if not os.path.exists("/groups/funke/home/tame/experiments/" + experiment_name):
        os.makedirs(
            "/groups/funke/home/tame/experiments/" + experiment_name, exist_ok=True
        )
    if not os.path.exists(
        "/groups/funke/home/tame/experiments/" + experiment_name + "/models/"
    ):
        os.makedirs(
            "/groups/funke/home/tame/experiments/" + experiment_name + "/models/",
            exist_ok=True,
        )
    if not os.path.exists(
        "/groups/funke/home/tame/experiments/" + experiment_name + "/snapshots/"
    ):
        os.makedirs(
            "/groups/funke/home/tame/experiments/" + experiment_name + "/snapshots/",
            exist_ok=True,
        )

    raw = gp.ArrayKey("TRAIN_RAW")
    seam_cells = gp.GraphKey("TRAIN_SEAM_CELLS")
    seam_cell_blobs = gp.ArrayKey("TRAIN_SEAM_CELL_BLOBS")
    prediction = gp.ArrayKey("TRAIN_PREDICTION")
    raw_source = gp.ZarrSource(raw_filename, {raw: "raw"}) + gp.Pad(raw, None)
    seam_cell_source = gp.CsvPointsSource(
        "/groups/funke/home/tame/data/train.csv", seam_cells, ndims=4, scale=voxel_size
    )

    combined_source = (raw_source, seam_cell_source, *gp.MergeProvider())
    stack = gp.Stack(batch_size)
    precache = gp.PreCache(cache_size=50, num_workers=20)

    pipeline = (
        combined_source
        + gp.RandomLocation(ensure_nonempty=seam_cells)
        + gp.IntensityAugment(  # remove for next run
            raw, scale_min=1.1, scale_max=1.5, shift_min=0.1, shift_max=0.5, clip=False
        )
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
        + precache
        + gp.torch.Train(
            model,
            loss,
            optimizer,
            inputs={"input": raw},
            outputs={0: prediction},
            loss_inputs={0: prediction, 1: seam_cell_blobs},
            array_specs={prediction: gp.ArraySpec(voxel_size=voxel_size)},
            save_every=save_model_every,
            checkpoint_basename=os.path.join(
                "/groups/funke/home/tame/experiments", experiment_name, "models/model"
            ),
        )
        + gp.Unsqueeze([seam_cell_blobs], 1)
        + gp.Snapshot(
            {
                raw: "raw",
                seam_cells: "seam_cells",
                seam_cell_blobs: "target",
                prediction: "prediction",
            },
            every=save_snapshot_every,
            output_dir=os.path.join(
                "/groups/funke/home/tame/experiments", experiment_name, "snapshots"
            ),
            output_filename="{iteration}.zarr",
        )
    )
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(seam_cells, output_size)
    request.add(seam_cell_blobs, output_size)
    request.add(prediction, output_size)

    with gp.build(pipeline):
        for i in range(num_iterations):
            pipeline.request_batch(request)
            # print(batch[raw].data.shape, batch[prediction].data.shape, batch[seam_cell_blobs].data.shape)


train(num_iterations)
