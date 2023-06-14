from parameters import model, loss, optimizer, input_size, output_size, voxel_size, raw_filename

import gunpowder as gp

def validate(num_iterations):
    model.eval()
    raw = gp.ArrayKey("RAW")
    seam_cells = gp.GraphKey("VAL_SEAM_CELLS")
    seam_cell_blobs = gp.ArrayKey("VAL_SEAM_CELL_BLOBS")
    prediction = gp.ArrayKey("VAL_PREDICTION")
    raw_source = gp.ZarrSource(raw_filename, {raw: "raw"})
    seam_cell_source = gp.CsvPointsSource("val.csv", seam_cells)
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
        + gp.torchPredict(model, inputs={"input": raw}, outputs={0: prediction}) # how do we extract predicted area?
        + gp.Snapshot(
            {
                raw: "val_raw",
                seam_cell_blobs: "val_target",
                prediction: "val_prediction",
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

def test(num_iterations):
    model.eval()
    raw = gp.ArrayKey("TEST_RAW")
    seam_cells = gp.GraphKey("TEST_SEAM_CELLS")
    seam_cell_blobs = gp.ArrayKey("TEST_SEAM_CELL_BLOBS")
    prediction = gp.ArrayKey("TEST_PREDICTION")
    raw_source = gp.ZarrSource(raw_filename, {raw: "raw"})
    seam_cell_source = gp.CsvPointsSource("test.csv", seam_cells)
    combined_source = (raw_source, seam_cell_source) + gp.MergeProvider()
    pipeline = (
        combined_source
        + gp.RasterizeGraph(
            seam_cells,
            seam_cell_blobs,  
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=(0.01, 10000, 10000, 10000),
                mode="peak",  # this makes the blob have values 0-1 on a gaussian dist
            ),
        )
        + gp.torchPredict(model, inputs={"input": raw}, outputs={0: prediction})
        + gp.Snapshot(
            {
                raw: "test_raw",
                seam_cell_blobs: "test_target",
                prediction: "test_prediction",
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