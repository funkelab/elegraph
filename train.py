from train_parameters import (
    raw_filename,
    experiment_name,
    in_channels,
    lr,
    weight_fg,
    gaussian_threshold,
    batch_size,
    gaussian_sigma,
    save_model_every,
    save_snapshot_every,
    num_iterations,
)

import gunpowder as gp
import os
from funlib.learn.torch.models import UNet, ConvPass
import zarr
import torch
from criterions import Loss
from models import Model


def train(num_iterations):
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

    # create loss
    loss = Loss(
        path=os.path.join('experiments', experiment_name, 'train_loss.csv'),
        weight_fg=weight_fg,
        gaussian_threshold=gaussian_threshold,
    )

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # get voxel size
    voxel_size = gp.Coordinate(zarr.open(raw_filename, "r")['raw'].attrs['resolution'])
    input_shape = gp.Coordinate((1, 128, 128, 128))
    output_shape = gp.Coordinate((1, 88, 88, 88))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    if not os.path.exists('experiments/' + experiment_name):
        os.makedirs('experiments/' + experiment_name)
    if not os.path.exists('experiments/' + experiment_name + '/models/'):
        os.makedirs('experiments/' + experiment_name + '/models/')
    if not os.path.exists('experiments/' + experiment_name + '/snapshots/'):
        os.makedirs('experiments/' + experiment_name + '/snapshots/')

    raw = gp.ArrayKey('TRAIN_RAW')
    seam_cells = gp.GraphKey('TRAIN_SEAM_CELLS')
    seam_cell_blobs = gp.ArrayKey('TRAIN_SEAM_CELL_BLOBS')
    prediction = gp.ArrayKey('TRAIN_PREDICTION')
    raw_source = gp.ZarrSource(raw_filename, {raw: 'raw'}) + gp.Pad(raw, None)
    seam_cell_source = gp.CsvPointsSource(
        'train.csv', seam_cells, ndims=4, scale=voxel_size
    )

    combined_source = (raw_source, seam_cell_source) + gp.MergeProvider()
    stack = gp.Stack(batch_size)
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
                mode='peak',
            ),
        )
        + stack
        + gp.torch.Train(
            model,
            loss,
            optimizer,
            inputs={'input': raw},
            outputs={0: prediction},
            loss_inputs={0: prediction, 1: seam_cell_blobs},
            array_specs={prediction: gp.ArraySpec(voxel_size=voxel_size)},
            save_every=save_model_every,
            checkpoint_basename=os.path.join('experiments', experiment_name, 'models/model'),
        )
        + gp.Snapshot(
            {
                raw: 'raw',
                seam_cells: 'seam_cells',
                seam_cell_blobs: 'target',
                prediction: 'prediction',
            },
            every=save_snapshot_every,
            output_dir=os.path.join('experiments', experiment_name, 'snapshots'),
            output_filename='{iteration}.zarr',
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


train(num_iterations)
