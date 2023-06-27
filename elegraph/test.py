import os
import sys

import gunpowder as gp
import torch
import zarr
from funlib.learn.torch.models import ConvPass, UNet
from models import Model, UnsqueezeModel

base_dir = os.getcwd()
sys.path.append(base_dir)


from test_parameters import (
    raw_filename,
    checkpoint_path,
)

from train_parameters import (
    num_fmaps,
    fmap_inc_factor,
    levels,
    in_channels,
    input_dim,
    output_dim
)


def test():
    if not os.path.exists("inference/"):
        os.makedirs("inference/")

    # create model
    unet = UNet(
        in_channels=in_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=[[2, 2, 2], [2, 2, 2]],
        kernel_size_down=[[[3, 3, 3], [3, 3, 3]]] * levels,
        kernel_size_up=[[[3, 3, 3], [3, 3, 3]]] * (levels-1),
        padding="valid",
    )

    unet_custom = Model(unet)
    unsqueeze_custom = UnsqueezeModel()
    model = torch.nn.Sequential(
        unet_custom,
        ConvPass(6, 1, [(1, 1, 1)], activation="Sigmoid"),
        unsqueeze_custom
    )

    # replace state of weight with pre-trained model weights
    state = torch.load(checkpoint_path)
    model.load_state_dict(state["model_state_dict"], strict=True)

    # set in eval mode
    model.eval()

    # get voxel size
    voxel_size = gp.Coordinate(zarr.open(raw_filename, "r")["raw"].attrs["resolution"])

    # get scan tile size
    input_tile_shape = gp.Coordinate((1, input_dim, input_dim, input_dim))
    output_tile_shape = gp.Coordinate((1, output_dim, output_dim, output_dim))
    input_tile_size = input_tile_shape * voxel_size
    output_tile_size = output_tile_shape * voxel_size

    # get actual size
    input_shape = gp.Coordinate((96, 308, 425, 325))
    output_shape = gp.Coordinate((96, 308, 425, 325))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    raw = gp.ArrayKey("TEST_RAW")
    prediction = gp.ArrayKey("TEST_PREDICTION")
    raw_source = gp.ZarrSource(raw_filename, {raw: "raw"}) + gp.Pad(raw, None)
    stack = gp.Stack(1)


    # request matching the model input and output sizes
    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_tile_size)
    scan_request.add(prediction, output_tile_size)
    scan = gp.Scan(scan_request)

    # prepare the zarr dataset to write to
    f = zarr.open("inference/result.zarr")
    ds = f.create_dataset("prediction", shape=(1, 1, 96, 308, 425, 325))
    ds.attrs["resolution"] = (1, 1625, 1625, 1625)
    ds.attrs["offset"] = (0, 0, 0, 0)
    ds2 = f.create_dataset("raw", shape=(1, 2, 96, 308, 425, 325))
    ds2.attrs["resolution"] = (1, 1625, 1625, 1625)
    ds2.attrs["offset"] = (0, 0, 0, 0)

    # create a zarr write node to store the predictions
    zarr_write = gp.ZarrWrite(
        output_filename="inference/result.zarr",
        dataset_names={raw: "raw", prediction: "prediction"},
     )

    pipeline = (
        raw_source
        + stack
        + gp.torch.Predict(
            model,
            inputs={"input": raw},
            outputs={0: prediction},
            array_specs={prediction: gp.ArraySpec(voxel_size=voxel_size)},
        )
        + zarr_write
        + scan
    )

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(prediction, output_size)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
        print(np.min(batch[raw].data), np.max(batch[raw].data))

test()
