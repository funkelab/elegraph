from parameters import model, loss, optimizer, input_size, output_size, voxel_size, raw_filename 

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
    #stack = gp.Stack(2)

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
        #+ stack
        + gp.Snapshot(
            {
                raw: "raw",
                seam_cells: "seam_cells",
                seam_cell_blobs: "target",
            },
            every=10,
        )
    )

    test_request = gp.BatchRequest()
    test_request.add(raw, input_size)
    test_request.add(seam_cells, output_size)
    test_request.add(seam_cell_blobs, output_size)

    with gp.build(test_pipeline):
        test_pipeline.request_batch(test_request)

#try_random(10)

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
        + gp.IntensityAugment(raw, scale=1.1, shift=0.1)
        + gp.RasterizeGraph(  
            seam_cells,
            seam_cell_blobs, 
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=(0.01, 10000, 10000, 10000),
                mode="peak",  # this makes the blob have values 0-1 on a gaussian dist
            ),
        )
        + gp.torch.Train(
            model,
            loss,
            optimizer,
            inputs={"input": raw},
            outputs={0: prediction},
            loss_inputs={0: prediction, 1: seam_cell_blobs},  # ground truth data
            array_spec={prediction: gp.ArraySpec(voxel_size=voxel_size)},
            save_every=1000,  # store learned weights at every 1000th iteration
        )
        + gp.Snapshot(
            {
                raw: "train_raw",
                seam_cells: "seam_cells",
                seam_cell_blobs: "train_target",
                prediction: "train_prediction",
            },
            every=10,  
        )
    )
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(seam_cells, output_size)
    request.add(seam_cell_blobs, output_size)
    request.add(prediction, output_size)

    with gp.build(pipeline):
        for i in range(num_iterations):  # is this the number of epochs?
            pipeline.request_batch(request)

# train(100) 

# 1) how do we use gp.Stack() to add the necessary additional dimension when training?
# 2) are my IntensityAugment parameters reasonable?