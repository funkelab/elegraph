import glob
import csv
import pandas as pd

csv_filenames = sorted(glob.glob("/groups/funke/home/tame/seamcellcoordinates/*.csv"))

# write large csv file containing all seam cells
with open("all_seam_cells.csv", "w") as f:
    count_id = 0
    writer = csv.writer(
        f, delimiter=" ", quotechar="|"
    )  # need to make sure that there are spaces between values
    # writer.writerow(["time", "z", "y", "x", "name", "id"])
    for file in csv_filenames:
        frame = int(file.split(".")[0].split("_")[-1])
        with open(file, "r") as g:
            reader = csv.reader(g)
            for row in reader:
                if row[1] == "x_voxels":
                    continue
                z = row[3]
                y = row[2]
                x = row[1]
                name = row[0]
                id = count_id
                count_id += 1
                writer.writerow(
                    [frame, z, y, x]
                )  

# shuffle large csv file
df = pd.read_csv("all_seam_cells.csv")
# TODO: make deterministic by setting a seed
shuffled_df = df.sample(frac=1).reset_index(drop=True)
shuffled_df.to_csv("all_seam_cells.csv", index=False)

# split large csv file into 3 csv files (80% train, 10% validation, 10% test)
df = pd.read_csv("all_seam_cells.csv")
num_train = round(0.8 * len(df))
num_val = round(0.1 * len(df))
num_test = len(df) - num_train - num_val
train_df = df[:num_train]
val_df = df[num_train : num_train + num_val]
test_df = df[num_train + num_val :]
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)