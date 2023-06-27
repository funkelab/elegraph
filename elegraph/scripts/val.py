import argparse

parser = argparse.ArgumentParser()
parser.add_argument("test", type=str)
args = parser.parse_args()
test = args.test

import sys

directory_path = "/groups/funke/home/tame/experiments/" + test
sys.path.append(directory_path)

import gunpowder as gp
from val_parameters import (
    batch_size,
    f1_score_csv,
    gaussian_sigma,
    input_dim,
    model,
    output_dim,
    raw_filename,
    save_snapshot_every,
    val_csv,
)

from metric import calc_f1_volumes


def validate(num_iterations):
    model.eval()

    # sizes
    voxel_size = gp.Coordinate(zarr.open(raw_filename, "r")["raw"].attrs["resolution"])
    input_shape = gp.Coordinate((1, input_dim, input_dim, input_dim))
    output_shape = gp.Coordinate((1, output_dim, output_dim, output_dim))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    raw = gp.ArrayKey("RAW")
    seam_cells = gp.GraphKey("VAL_SEAM_CELLS")
    seam_cell_blobs = gp.ArrayKey("VAL_SEAM_CELL_BLOBS")
    prediction = gp.ArrayKey("VAL_PREDICTION")
    raw_source = gp.ZarrSource(raw_filename, {raw: "raw"})
    seam_cell_source = gp.CsvPointsSource(
        val_csv, seam_cells, ndims=4, scale=voxel_size
    )
    combined_source = (raw_source, seam_cell_source, *gp.MergeProvider())
    stack = gp.Stack(batch_size)
    precache = gp.PreCache(cache_size=50, num_workers=20)

    pipeline = (
        combined_source
        + gp.RandomLocation(ensure_nonempty=seam_cells)
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
        + gp.torchPredict(model, inputs={"input": raw}, outputs={0: prediction})
        + gp.Snapshot(
            {
                raw: "raw",
                seam_cell_blobs: "target",
                prediction: "prediction",
            },
            every=save_snapshot_every,
        )
    )
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(seam_cells, output_size)
    request.add(seam_cell_blobs, output_size)
    request.add(prediction, output_size)

    with gp.build(pipeline):
        avg_f1_score = 0
        count = 1
        with open(f1_score_csv, "a") as f:
            writer = csv.writer(f, delimiter=" ")
            for i in range(num_iterations):
                batch = pipeline.request_batch(request)
                f1_score = calc_f1_volumes(
                    batch[seam_cell_blobs].data, batch[prediction].data
                )  # batch size is 1, so [i] should only access one volume
                print("F1 Score: ", f1_score)
                if count == 1:
                    writer.writerow(["Sample", "F1_Score"])  # header
                writer.writerow([count, f1_score])
                count += 1
                avg_f1_score += f1_score
            avg_f1_score = avg_f1_score / num_iterations
            writer.writerow(["Average F1 Score:", avg_f1_score])
