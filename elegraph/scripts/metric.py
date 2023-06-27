import numpy as np
from scipy.optimize import linear_sum_assignment


def max_pooling(volume):
    """
    Use a 3x3x3 kernel to perform max pooling. Output should be the same shape as the original predicted volume.
    """
    # pad volume
    padded_volume = np.pad(
        volume, ((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=0
    )
    output_volume = np.zeros(volume.shape)

    for z in range(output_volume.shape[0]):
        for y in range(output_volume.shape[1]):
            for x in range(output_volume.shape[2]):
                # Extract the 3x3x3 region
                region = padded_volume[z : z + 3, y : y + 3, x : x + 3]
                # Take the maximum value in the region and assign to output
                output_volume[z, y, x] = np.max(region)
    return output_volume


def find_local_max(volume, compare):
    """
    Find and return list of volume's coordinates where the local max values reside.
    NOTE: Parameter 'compare' can be the max pool volume or an integer
    """
    # create boolean matrix, 'True' indicating local max is at that coordinate, 'False' otherwise
    binary_matrix = volume == compare
    local_maxes = []

    for z in range(binary_matrix.shape[0]):
        for y in range(binary_matrix.shape[1]):
            for x in range(binary_matrix.shape[2]):
                if binary_matrix[z, y, x]:
                    local_maxes.append((z, y, x))
    return local_maxes


def f1_score(tp, fp, fn):
    """
    Computes the F1 score using true positive, false positive, and false negative scores
    """
    return tp / (tp + (0.5 * (fp + fn)))


def calc_f1_volumes(predicted_volume, gt_volume, radius):
    """
    Calculate F1 score between predicted and ground_truth volumes using the Hungarian Algorithm.
    """
    max_pooled_volume = max_pooling(predicted_volume)
    predicted_local_max = find_local_max(predicted_volume, max_pooled_volume)
    gt_local_max = find_local_max(
        gt_volume, 1
    )  # ground truth maxes are where the coordinate's intensity is 1 due to gaussian blobs
    euclid_dist_array = np.zeros((len(gt_local_max), len(predicted_local_max)))
    np.zeros(
        euclid_dist_array.shape
    ) == 1  # all false array of same size as euclid_dist_array

    # set array filled with euclidean distance between every ground truth point to a predicted point
    for row, gt in enumerate(gt_local_max):
        for col, pred in enumerate(predicted_local_max):
            euclid_dist = np.sqrt(
                (gt[0] - pred[0]) ** 2 + (gt[1] - pred[1]) ** 2 + (gt[2] - pred[2]) ** 2
            )
            euclid_dist_array[row, col] = euclid_dist

    # if distance is higher than radius, set to highest value to avoid using it during hungarian alg
    highest_val = max(euclid_dist_array)
    false_neg = 0
    for col in range(euclid_dist_array.shape[1]):
        all_highest_val = True
        for row in range(euclid_dist_array.shape[0]):
            if euclid_dist_array[row, col] > radius:
                euclid_dist_array[row, col] = highest_val
            else:
                all_highest_val = False
        # if a column has rows that are all the highest value, then that pred point is false negative
        if all_highest_val:
            false_neg += 1

    # find best matches using hungarian alg
    row_ind, col_ind = linear_sum_assignment(euclid_dist_array)
    true_pos = len(row_ind) - false_neg
    false_pos = euclid_dist_array.shape[0] - len(row_ind)

    return f1_score(true_pos, false_pos, false_neg)


def test_f1(voxel_size, radius, pred_volume, gt_volume):
    """
    Iterate over test.csv file and find average f1 scores for each time volume.
    Average these times for the entire sequence.

    radius (in world units) is the length of half of a side starting from a seam cell of the area we want to capture
    """
    all_seam_cells_path = "/groups/funke/home/tame/data/all_seam_cells.csv"
    test_csv_path = "/groups/funke/home/tame/data/test.csv"

    all_data = np.genfromtxt(all_seam_cells_path, delimiter=" ")
    test_data = np.genfromtxt(test_csv_path, delimiter=" ")

    # find smallest and largest time frames
    min_time = np.min(all_data[:, 0])
    max_time = np.max(all_data[:, 0])

    total_f1 = 0
    count = 0

    # find f1 score for each time frame
    for time in range(min_time, max_time + 1):
        # get all rows in test.csv with the same time
        time_matrix = test_data[test_data[:, 0] == time]
        for row in time_matrix:
            time = row[0]
            z, y, z = row[1], row[2], row[3]
            z_start, z_end = z - radius, z + radius
            y_start, y_end = y - radius, y + radius
            x_start, x_end = x - radius, x + radius

            cut_pred = pred_volume[z_start:z_end, y_start:y_end, x_start:x_end]
            cut_gt = pred_volume[z_start:z_end, y_start:y_end, x_start:x_end]

            total_f1 += calc_f1_volumes(
                cut_pred, cut_gt
            )  # won't the cut_pred sometimes include blobs we've already seen in training/validation?
            count += 1

    return total_f1 / count
