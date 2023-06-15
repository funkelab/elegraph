from parameters import model, loss, optimizer, input_size, output_size, voxel_size, raw_filename, checkpoint_path

import gunpowder as gp

# sanity check for ground truth representations
def try_random(num_iterations):
    raw = gp.ArrayKey("TRY_RAW")
    seam_cells = gp.GraphKey("TRY_SEAM_CELLS")
    seam_cell_blobs = gp.ArrayKey("TRY_SEAM_CELL_BLOBS")
    raw_source = gp.ZarrSource(raw_filename, {raw: "raw"}) + gp.Pad(raw, None)
    seam_cell_source = gp.CsvPointsSource(
        "train.csv", seam_cells, ndims=4, scale=voxel_size
    )  
    combined_source = (raw_source, seam_cell_source) + gp.MergeProvider()

    test_pipeline = (
        combined_source
        + gp.RandomLocation(ensure_nonempty=seam_cells)
        + gp.RasterizeGraph(
            seam_cells,
            seam_cell_blobs,
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=(0.01, 10000, 10000, 10000),
                mode="peak",  # this makes the blob have values 0-1 on a gaussian dist
            ),
        )
        + gp.Snapshot(
            {
                raw: "raw",
                seam_cells: "seam_cells",
                seam_cell_blobs: "target",
            },
            every=10, # every tenth batch will be stored 
        )
    )

    test_request = gp.BatchRequest()
    test_request.add(raw, input_size)
    test_request.add(seam_cells, output_size)
    test_request.add(seam_cell_blobs, output_size)

    with gp.build(test_pipeline):
        test_pipeline.request_batch(test_request)

# try_random(10)

def train(num_iterations):
    raw = gp.ArrayKey("TRAIN_RAW")
    seam_cells = gp.GraphKey("TRAIN_SEAM_CELLS")
    seam_cell_blobs = gp.ArrayKey("TRAIN_SEAM_CELL_BLOBS")
    prediction = gp.ArrayKey("TRAIN_PREDICTION")
    raw_source = gp.ZarrSource(raw_filename, {raw: "raw"}) + gp.Pad(raw, None)
    seam_cell_source = gp.CsvPointsSource("train.csv", seam_cells, ndims=4, scale=voxel_size)
    combined_source = (raw_source, seam_cell_source) + gp.MergeProvider()

    pipeline = (
        combined_source
        + gp.RandomLocation(ensure_nonempty=seam_cells)
        + gp.IntensityAugment(raw, scale_min=1.1, scale_max=1.5, shift_min=0.1, shift_max=0.5)
        + gp.RasterizeGraph(  
            seam_cells,
            seam_cell_blobs, 
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=(0.01, 10000, 10000, 10000),
                mode="peak",  # this makes the blob have values 0-1 on a gaussian dist
            ),
        )
        + gp.torch.Train( # model's parameters update at each batch (1 sample of raw)
            model,
            loss,
            optimizer,
            inputs={"input": raw},
            outputs={0: prediction},
            loss_inputs={0: prediction, 1: seam_cell_blobs},  # ground truth data
            array_specs={prediction: gp.ArraySpec(voxel_size=voxel_size)},
            save_every=1000,  # save model at every 1000th iteration
            checkpoint_basename=checkpoint_path
        )
        + gp.Snapshot(
            {
                raw: "train_raw",
                seam_cells: "seam_cells",
                seam_cell_blobs: "train_target",
                prediction: "train_prediction",
            },
            every=10,  # save snapshot at every 10th batch
            output_filename='{iteration}'.zfill(3) + '.zarr'
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

train(10000) 

# Questions
# 1) how do we use gp.Stack() to add the necessary additional dimension when training?
# 2) are the IntensityAugment parameters reasonable?
# 3) to quanitify model accuracy:
#   -find the coordinates + areas of all ground truth gaussian blobs on a given volume (pi * 10000**2 world unit**2?)
#   -find the coordinates and area of all predicted gaussian blobs on a given volume
#   -use hungarian algorithm to find unique matches among both sets, maximizing score that assesses euclidean distance and Intersection Over Union
#   -if a matched pair has at least a score of x, we consider it a success. ex, pair 12 has success rate of 20/22 pairs
# 4) is the way I structured my files flexible enough for making changes to the model's parameters?
# 5) after this unet is trained/validated/tested, would we be able to pass in entire unlabeled volumes and identify seam cells with gaussian blobs chunk by chunk?
# ^ should we ask for another sequence?


# TO-DOs
# make sure to add in code to log loss on a CSV for every iteration 
# run training method on 10000 iterations on local computer using big computer!
# create a line plot showing the loss on every iteration
# visualize later zarr container on neuroglancer to ensure that model is learning
# for validation, you should pull the model's parameters from each zarr container and apply on val seam cell data
# log the mean loss of each saved zarr container on the val data on the same graph
# we want to find out the lowest loss for both train and val, but val is more important since it was never given gt