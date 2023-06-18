from parameters import (
    model,
    loss,
    optimizer,
    input_size,
    output_size,
    voxel_size,
    raw_filename,
    checkpoint_path,
    batch_size,
)

import gunpowder as gp
import numpy as np


def train(num_iterations):
    print(voxel_size)
    raw = gp.ArrayKey("TRAIN_RAW")
    seam_cells = gp.GraphKey("TRAIN_SEAM_CELLS")
    seam_cell_blobs = gp.ArrayKey("TRAIN_SEAM_CELL_BLOBS")
    prediction = gp.ArrayKey("TRAIN_PREDICTION")
    raw_source = gp.ZarrSource(raw_filename, {raw: "raw"}) + gp.Pad(raw, None)
    seam_cell_source = gp.CsvPointsSource(
        "train.csv", seam_cells, ndims=4, scale=voxel_size
    )  # TODO

    combined_source = (raw_source, seam_cell_source) + gp.MergeProvider()
    stack = gp.Stack(batch_size)  # TODO
    pipeline = (
        combined_source
        + gp.RandomLocation(ensure_nonempty=seam_cells)  # TODO
        + gp.IntensityAugment(
            raw, scale_min=1.1, scale_max=1.5, shift_min=0.1, shift_max=0.5
        )
        + gp.RasterizeGraph(
            seam_cells,
            seam_cell_blobs,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=(0.01, 20000, 20000, 20000),
                mode="peak",  # this makes the blob have values 0-1 on a gaussian dist
            ),
        )
        + stack  # TODO
        + gp.torch.Train(  # model's parameters update at each batch (1 sample of raw)
            model,
            loss,
            optimizer,
            inputs={"input": raw},
            outputs={0: prediction},
            loss_inputs={0: prediction, 1: seam_cell_blobs},  # ground truth data
            array_specs={prediction: gp.ArraySpec(voxel_size=voxel_size)},
            save_every=100,  # save model at every 1000th iteration
            checkpoint_basename=checkpoint_path,
        )
        + gp.Snapshot(
            {
                raw: "train_raw",
                seam_cells: "seam_cells",
                seam_cell_blobs: "train_target",
                prediction: "train_prediction",
            },
            every=10,  # save snapshot at every 10th batch
            output_filename="{iteration}".zfill(3) + ".zarr",
        )
    )
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(seam_cells, output_size)
    request.add(seam_cell_blobs, output_size)
    request.add(prediction, output_size)

    with gp.build(pipeline):
        for i in range(num_iterations):
            batch = pipeline.request_batch(request)
            for s in range(batch_size):
                print(
                    "s = {}, prediction, mean = {}".format(
                        s, np.mean(batch[prediction].data[s])
                    )
                )


train(10001)  # +1 to ensure we get the 10th model save
