import sys
directory_path = "/groups/funke/home/tame/experiments/test-001" #hm, i would have to change this and create a test_001 directory in experiments before running train.pyy
sys.path.append(directory_path)

from test_parameters import (
    model, 
    save_snapshot_every, 
    raw_filename, 
    test_csv, 
    f1_score_csv, 
    num_iterations,
    batch_size,
    input_dim,
    output_dim
)

def test(num_iterations):
    model.eval()
    raw = gp.ArrayKey("TEST_RAW")
    seam_cells = gp.GraphKey("TEST_SEAM_CELLS")
    seam_cell_blobs = gp.ArrayKey("TEST_SEAM_CELL_BLOBS")
    prediction = gp.ArrayKey("TEST_PREDICTION")
    raw_source = gp.ZarrSource(raw_filename, {raw: "raw"})
    seam_cell_source = gp.CsvPointsSource("test.csv", seam_cells, ndims=4, scale=voxel_size)
    combined_source = (raw_source, seam_cell_source) + gp.MergeProvider()
    stack = gp.Stack(1)
    
    # get voxel size
    voxel_size = gp.Coordinate(zarr.open(raw_filename, "r")['raw'].attrs['resolution'])
    input_shape = gp.Coordinate((1, input_dim, input_dim, input_dim))
    output_shape = gp.Coordinate((1, output_dim, output_dim, output_dim))
    final_output_shape = gp.Coordinate((1, 308, 425, 325))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size
    final_output_size = final_output_shape * voxel_size

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(prediction, output_size)

    scan = gp.Scan(scan_request)
    
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
        + stack
        + gp.torchPredict(model, inputs={"input": raw}, outputs={0: prediction})
        + gp.Snapshot(
            {
                raw: "raw",
                seam_cell_blobs: "target",
                prediction: "prediction",
            },
            every=10, 
        )
        + scan
    )
    request = gp.BatchRequest()
    request.add(raw, final_output_size)
    request.add(seam_cells, final_output_size)
    request.add(seam_cell_blobs, final_output_size)
    request.add(prediction, final_output_size)

    with gp.build(pipeline):
        avg_f1_score = 0
        count = 1
        with open(f1_score_csv, "a") as f:
            writer = csv.writer(f, delimiter=" ")
            for i in range(num_iterations):
                batch = pipeline.request_batch(request)
                f1_score = calc_f1_volumes(batch[seam_cell_blobs].data, batch[prediction].data) # batch size is 1, so [i] should only access one volume
                if count == 1:
                    writer.writerow(["Sample", "F1_Score"])  # header
                writer.writerow([count, f1_score])
                count += 1
                avg_f1_score += f1_score
            avg_f1_score = avg_f1_score / num_iterations
            writer.writerow(["Average F1 Score:", avg_f1_score])
        print("F1 Score: ", f1_score)