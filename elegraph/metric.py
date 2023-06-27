import numpy as np
from scipy.optimize import linear_sum_assignment
import glob
import csv

def max_pooling(volume):
    """
    Use a 3x3x3 kernel to perform max pooling. Output should be the same shape as the original predicted volume.
    Each element in the volume should be at the center of the kernel during an iteration. 
    The element at the same index in the output volume will be the maximum value of the entire kernel at that iteration.
    """
    # pad volume
    padded_volume = np.pad(volume, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
    output_volume = np.zeros(volume.shape)
    
    for z in range(output_volume.shape[0]):
        for y in range(output_volume.shape[1]):
            for x in range(output_volume.shape[2]):
                pad_z = z+1
                pad_y = y+1
                pad_x = x+1
                # Extract the 3x3x3 region
                region = padded_volume[pad_z-1:pad_z+2, pad_y-1:pad_y+2, pad_x-1:pad_x+2]
                # Take the maximum value in the region and assign to output
                output_volume[z, y, x] = np.max(region)
    return output_volume

def find_local_max(volume, compare):
    """
    Find and return list of volume's coordinates where the local max values reside.  
    NOTE: Parameter 'compare' can be the max pool volume or an integer
    """
    # create boolean matrix, 'True' indicating local max is at that coordinate, 'False' otherwise
    binary_matrix = (volume == compare)
    local_maxes = []

    for z in range(binary_matrix.shape[0]):
        for y in range(binary_matrix.shape[1]):
            for x in range(binary_matrix.shape[2]):
                if binary_matrix[z,y,x]:
                    local_maxes.append((z,y,x))
    return local_maxes

def get_gt_seam_cell_locations():
    """
    Output should be list of lists, with each internal list representing a time frame 
    of tuples of (z,y,x) seam cell coordinates
    """
    csv_filenames = sorted(glob.glob("/groups/funke/home/tame/data/seamcellcoordinates/*.csv"))
    all_seam_cells_list = []

    for file in csv_filenames:
        with open(file, "r") as f:
            vol = []
            reader = csv.reader(f)
            for row in reader:
                if row[1] == "x_voxels":
                    continue
                z,y,x = row[3], row[2], row[1]
                vol.append((z,y,x))
            all_seam_cells_list.append(vol)
    return all_seam_cells_list

def f1_score(tp, fp, fn):
    """
    Computes the F1 score using true positive, false positive, and false negative scores
    """
    return tp / (tp + (0.5 * (fp + fn)))

def calc_f1_volumes(predicted_volume, gt_locations, max_dist):
    """
    Calculate F1 score between predicted and ground_truth volumes using the Hungarian Algorithm.
    """
    max_pooled_volume = max_pooling(predicted_volume)
    predicted_local_max = find_local_max(predicted_volume, max_pooled_volume)
    euclid_dist_array = np.zeros((len(gt_locations), len(predicted_local_max)))
    visited_array = np.zeros(euclid_dist_array.shape) == 1 # all false array of same size as euclid_dist_array
    
    # set array filled with euclidean distance between every ground truth point to a predicted point 
    for row, gt in enumerate(gt_locations):
        for col, pred in enumerate(predicted_local_max):
            euclid_dist = np.sqrt((gt[0] - pred[0]) ** 2 + (gt[1] - pred[1]) ** 2 + (gt[2] - pred[2]) ** 2)
            euclid_dist_array[row, col] = euclid_dist
    
    # if distance is higher than max_dist, set to highest value to avoid using it during hungarian alg
    highest_val = max(euclid_dist_array) 
    false_neg = 0
    for row in range(euclid_dist_array.shape[0]):
        all_highest_val = True
        for col in range(euclid_dist_array.shape[1]):
            if euclid_dist_array[row,col] > max_dist:
                euclid_dist_array[row,col] = highest_val
            else:
                all_highest_val = False
        # if a row has columns that are all the highest value (euclid distance higher than max_dist), then that pred point is a false negative since it was not detected
        if all_highest_val:
            false_neg += 1
    
    # find best matches using hungarian alg
    row_ind, col_ind = linear_sum_assignment(euclid_dist_array)
    true_pos = len(row_ind) - false_neg
    false_pos = euclid_dist_array.shape[1] - euclid_dist_array.shape[0]

    print("True Positive: ", true_pos)
    print("False Positive: ", false_pos)
    print("False Negative: ", false_neg)

    return f1_score(true_pos, false_pos, false_neg)

def coordInMatrix(coord, radius, seam_cell):
    """
    Assume coord and matrix are both in 3D. seam_cell is the coordinate at the center of the matrix, with all dims of radius x 2.
    Return true if coordinate can be in the matrix, else return false.
    """
    z,y,x = seam_cell[0], seam_cell[1], seam_cell[2]
    z_start, z_end = z - radius, z + radius
    y_start, y_end = y - radius, y + radius
    x_start, x_end = x - radius, x + radius
    return coord[0] >= z_start and coord[1] <= z_end and coord[1] >= y_start and coord[1] <= y_end and coord[2] >= x_start and coord[2] <= x_end


def find_total_f1_scores(pred_volume, gt_locations, radius, test_data, time):
    """
    For a given time volume, find the TOTAL F1 scores of the seam cells in test.csv. 
    
    radius (in world units) is the length of half of a side starting from a seam cell of the area we want to capture
    """
    sum_f1 = 0
    # get all rows in test.csv with the same time
    time_matrix = test_data[test_data[:, 0] == time]
    for row in time_matrix:
        time = row[0]
        z,y,x = row[3], row[2], row[1]
        z_start, z_end = z - radius, z + radius + 1
        y_start, y_end = y - radius, y + radius + 1
        x_start, x_end = x - radius, x + radius + 1

        # get chunk in the volumes that contains the seam cell coordinate within the radius
        cut_pred = pred_volume[z_start:z_end, y_start:y_end, x_start: x_end]
        #print(cut_pred.shape)

        # only use the gt seam cells that are within this cut_pred
        gt_locations_in_cut_pred = []
        for gt in gt_locations:
            if coordInMatrix(gt, radius, (z,y,x)):
                gt_locations_in_cut_pred.append(gt)
        
        # calculate f1 score with those chunks and gt seam cells
        sum_f1 += calc_f1_volumes(cut_pred, gt_locations_in_cut_pred) # won't the cut_pred sometimes include blobs we've already seen in training/validation?
    return sum_f1

def test_f1(large_predicted_volume, radius=5):
    """
    'large_predicted_volume' is the entire predicted volume from test.py with dimensions: (1, 2, 96, 308, 425, 325)
    return the average f1 score of this prediction from all seam cell points in test.csv
    """
    large_predicted_volume = large_predicted_volume[0,0]
    final_f1 = 0
    gt_locations = get_gt_seam_cell_locations()
    test_data = np.genfromtxt(test_csv_path, delimiter=' ')
    num_test = len(test_data)
    img_data = zarr.open("/groups/funke/home/tame/data/img_data.zarr", "a")
    test_csv_path = "/groups/funke/home/tame/data/test.csv"
    test_data = np.genfromtxt(test_csv_path, delimiter=' ')

    for i in range(len(gt_locations)):
        small_pred_vol = large_predicted_volume[i+20] # remember to switch back to regular 20 instead of 20.5
        small_gt_location = gt_locations[i]
        final_f1 += find_total_f1_scores(small_pred_vol, small_gt_location, radius, test_data, i+20)
    return final_f1 / num_test

def test_f1_diff_radius(large_predicted_volume, low_radius, high_radius):
    """
    Plots F1 scores for the large_predicted_volume's test.csv seam cell points for all radiuses between low_radius to high_radius.
    Radius is the distance from the center of the chunk where the test seam cell coordinate is.
    """
    radius = []
    all_f1s = []
    for i in range(low_radius, high_radius+1):
        radius.append(i)
        all_f1s.append(test_f1(large_predicted_volume, i))
    plt.plot(radius, all_f1s, label='F1 Score')
    plt.title('F1 Score from Different Radiuses')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.xticks(range(low_radius, high_radius+1, 1))
    plt.legend(loc='best')
    plt.show()
    plt.savefig("/groups/funke/home/tame/elegraph/elegraph/f1_scores_radiuses.png")