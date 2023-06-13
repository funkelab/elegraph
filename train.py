import gunpowder as gp
import zarr
import torch
import glob
import tifffile
import matplotlib.pyplot as plt
from funlib.learn.torch.models import UNet, ConvPass


raw_filename = "/groups/funke/home/tame/elegraph/img_data.zarr"


# helper function to show image(s)
def imshow(raw, ground_truth=None, prediction=None):
    rows = 1
    if ground_truth is not None:
        rows += 1
    if prediction is not None:
        rows += 1
    cols = raw.shape[0] if len(raw.shape) > 3 else 1
    fig, axes = plt.subplots(
        rows, cols, figsize=(10, 4), sharex=True, sharey=True, squeeze=False
    )
    if len(raw.shape) == 3:
        axes[0][0].imshow(raw.transpose(1, 2, 0))
    else:
        for i, im in enumerate(raw):
            axes[0][i].imshow(im.transpose(1, 2, 0))
    row = 1
    if ground_truth is not None:
        if len(ground_truth.shape) == 3:
            axes[row][0].imshow(ground_truth[0])
        else:
            for i, gt in enumerate(ground_truth):
                axes[row][i].imshow(gt[0])
        row += 1
    if prediction is not None:
        if len(prediction.shape) == 3:
            axes[row][0].imshow(prediction[0])
        else:
            for i, gt in enumerate(prediction):
                axes[row][i].imshow(gt[0])
    plt.show()


# use this for local purposes
# regA_filenames = sorted(glob.glob('/Users/ethantam/Desktop/Raw/RegA/*.tif'))
# regB_filenames = sorted(glob.glob('/Users/ethantam/Desktop/Raw/RegB/*.tif'))
# csv_filenames = glob.glob('/Users/ethantam/Desktop/Raw/SeamCellCoordinates/*.csv')

# use this for cluster
regA_filenames = sorted(glob.glob("/groups/funke/home/tame/RegA/*.tif"))

d, h, w = tifffile.imread(regA_filenames[0]).shape

unet = UNet(
    in_channels=3,
    num_fmaps=6,
    fmap_inc_factor=4,
    downsample_factors=[[2, 2, 2], [2, 2, 2]],
    kernel_size_down=[[[3, 3, 3], [3, 3, 3]]] * 3,
    kernel_size_up=[[[3, 3, 3], [3, 3, 3]]] * 2,
    padding="valid",
)

model = torch.nn.Sequential(
    unet,
    ConvPass(6, 1, [(1, 1, 1)], activation=None),  # final 1x1x1 conv pass
    torch.nn.Sigmoid(),
)

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

voxel_size = gp.Coordinate(
    zarr.open(raw_filename, "r")["raw"].attrs["resolution"]
)  # already notated in zarr container
input_shape = gp.Coordinate((1, 128, 128, 128))
output_shape = gp.Coordinate((1, 88, 88, 88))
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size


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


try_random(10)


def train(num_iterations):
    raw = gp.ArrayKey("TRAIN_RAW")
    seam_cells = gp.GraphKey("TRAIN_SEAM_CELLS")
    seam_cell_blobs = gp.ArrayKey("TRAIN_SEAM_CELL_BLOBS")
    prediction = gp.ArrayKey("TRAIN_PREDICTION")
    raw_source = gp.ZarrSource("/groups/funke/home/tame/img_data", {raw: "raw"})
    seam_cell_source = gp.CsvPointsSource("train.csv", seam_cells)
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
                radius=20000,
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
                seam_cell_blobs: "train_target",
                prediction: "train_prediction",
            },
            every=10,  
        )
    )
    request = gp.BatchRequest()
    request[raw] = input_size
    request[seam_cell_blobs] = output_size
    request[prediction] = output_size
    with gp.build(pipeline):
        for i in range(num_iterations):  # is this the number of epochs?
            pipeline.request_batch(request)

# a validate method
def validate(num_iterations):
    raw = gp.ArrayKey("RAW")
    seam_cells = gp.GraphKey("VAL_SEAM_CELLS")
    seam_cell_blobs = gp.ArrayKey("VAL_SEAM_CELL_BLOBS")
    prediction = gp.ArrayKey("VAL_PREDICTION")
    raw_source = gp.ZarrSource("/groups/funke/home/tame/img_data", {raw: "raw"})
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
                radius=20000,
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
                raw: "val_raw",
                seam_cell_blobs: "val_target",
                prediction: "val_prediction",
            },
            every=10, 
        )
    )
    request = gp.BatchRequest()
    request[raw] = input_size
    request[seam_cell_blobs] = output_size
    request[prediction] = output_size
    with gp.build(pipeline):
        for i in range(num_iterations):  # is this the number of epochs?
            pipeline.request_batch(request)


# a test method
def test(num_iterations):
    model.eval()
    raw = gp.ArrayKey("TEST_RAW")
    seam_cells = gp.GraphKey("TEST_SEAM_CELLS")
    seam_cell_blobs = gp.ArrayKey("TEST_SEAM_CELL_BLOBS")
    prediction = gp.ArrayKey("TEST_PREDICTION")
    raw_source = gp.ZarrSource("/groups/funke/home/tame/img_data", {raw: "raw"})
    seam_cell_source = gp.CsvPointsSource("test.csv", seam_cells)
    combined_source = (raw_source, seam_cell_source) + gp.MergeProvider()
    pipeline = (
        combined_source
        + gp.RasterizeGraph(
            seam_cells,
            seam_cell_blobs,  
            array_spec=gp.ArraySpec(voxel_size=voxel_size),
            settings=gp.RasterizationSettings(
                radius=20000,
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
    request[raw] = input_size
    request[prediction] = output_size
    with gp.build(pipeline):
        for i in range(num_iterations):  # is this the number of epochs?
            pipeline.request_batch(request)


# train(100) 
