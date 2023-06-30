import numpy as np
import glob
import csv
import zarr
from tqdm import tqdm

path = "/groups/funke/home/tame/elegraph/experiments/demo_run_1/inference/result.zarr"
large_predicted_volume = zarr.open(path, "r")["prediction"]
print(large_predicted_volume.shape)

# bright means at least 0.7
# to find true positives and false negatives: 
    # look at each predicted time vol
    # see if any of val.csv/test.csv are in that vol
        # if so, see whether or not the volume has a bright spot at that point 
            # if so, count it as a true positive.
            # if not, count it as a false negative.
        # if not, move onto the next time vol
    # also, keep count of general matches! 
    # iterate over the ground truth seam_cell.csv for that time vol and check that each point has a bright spot in the pred
        # if so, count += 1
        # else, don't do anything
     
# to find false positives:
    # look at each predicted time vol
    # find all local max in that vol (at least 0.5/6)
    # make sure you apply the greedy_filter to eliminate nearby max
    # number of these pred local max - count_matches = number of false positives

def get_gt_seam_cell_locations():
    """
    Returns list of lists, with each internal list representing a time frame 
    of tuples of (z,y,x) all seam cell coordinates
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
                z,y,x = float(row[3]), float(row[2]), float(row[1])
                vol.append((z,y,x))
            all_seam_cells_list.append(vol)
    return all_seam_cells_list

def count_valid_matches(time_vol, gt_time_list, bright_threshold=0.2):
    """
    Returns number of ground truth points appearing on predicted time volume.
    """
    matches = 0
    for cell in gt_time_list:
        z,y,x = int(cell[0]), int(cell[1]), int(cell[2])
        if time_vol[z,y,x] >= bright_threshold:
            matches += 1
    return matches

def count_bright(time_vol, test_data, time, bright_threshold=0.2):
    """
    Returns number of true positives and false negatives in a predicted time volume.
    """
    true_pos = 0
    false_neg = 0

    time_matrix = test_data[test_data[:, 0] == time]
    for row in time_matrix:
        z,y,x = int(row[1]), int(row[2]), int(row[3])
        if time_vol[z,y,x] >= bright_threshold:
            true_pos += 1
        else:
            false_neg += 1
    return (true_pos, false_neg)

def max_pool(volume, kernel):
    """
    Use a 3D kernel (tuple) on a volume to perform max pooling. 
    Output should be the same size as volume.
    """
    # Get volume shape
    depth, height, width = volume.shape

    # Get kernel dimensions
    kernel_depth, kernel_height, kernel_width = kernel

    # Calculate padding size
    pad_depth = kernel_depth - depth % kernel_depth
    pad_height = kernel_height - height % kernel_height
    pad_width = kernel_width - width % kernel_width

    # Pad the volume
    padded_volume = np.pad(volume, ((0, pad_depth), (0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    # Calculate output dimensions
    output_depth = depth + pad_depth
    output_height = height + pad_height
    output_width = width + pad_width

    # Create an empty array for the pooled output
    pooled_volume = np.zeros((output_depth, output_height, output_width))

    for d in range(0, output_depth, kernel_depth):
        for h in range(0, output_height, kernel_height):
            for w in range(0, output_width, kernel_width):
                # Calculate the end indices of the current pooling window
                end_d = d + kernel_depth
                end_h = h + kernel_height
                end_w = w + kernel_width

                # Extract the current window from the padded volume
                window = padded_volume[d:end_d, h:end_h, w:end_w]

                # Find the maximum value within the window
                pooled_value = np.max(window)

                # Store the maximum value in the pooled output
                pooled_volume[d:end_d, h:end_h, w:end_w] = pooled_value

    # Crop the pooled volume to match the original volume size
    pooled_volume = pooled_volume[:depth, :height, :width]

    return pooled_volume

def find_local_max(volume, compare, boundary=0.6):
    """
    Find and return list of volume's (z,y,x) coordinates where the local max values reside.  
    NOTE: Parameter 'compare' can be the max pool volume or an integer
    """
    # create boolean matrix, 'True' indicating local max is at that coordinate, 'False' otherwise
    binary_matrix = volume == compare
    local_maxes = []

    # local max must be bigger than boundary for it to be counted as a local max

    for z in range(binary_matrix.shape[0]):
        for y in range(binary_matrix.shape[1]):
            for x in range(binary_matrix.shape[2]):
                if binary_matrix[z, y, x]:
                    # we do not want to count potential local maxes that were simply the highest value of the background 
                    if type(compare) == np.ndarray and compare[z,y,x] < boundary:
                        continue
                    local_maxes.append((z, y, x))
    return local_maxes

def get_seam_cell_list(time_vol, time):
    """
    For a predicted time volume, return all seam 
    cell coordinates (z,y,x) in a list using non-maxima suppression.
    """
    all_seam_cells = []
    max_pooled_vol = max_pool(time_vol, (40,40,40))
    local_maxes = find_local_max(time_vol, max_pooled_vol)
    for coord in local_maxes:
        x,y,z = coord[2], coord[1], coord[0]
        updated_coord = (time, x, y, z, time_vol[z, y, x]) # T X Y Z Intensity
        all_seam_cells.append(updated_coord) 
    return all_seam_cells

def euclid_dist(x,y):
    """
    Returns the euclidean distance between two 3D coordinates.
    """
    return np.sqrt(((y[0] - x[0]) ** 2) + ((y[1] - x[1]) ** 2) + ((y[2] - x[2]) ** 2))

def greedy_filter(time_vol, time, elim_dist=25):
    """
    After obtaining and sorting the seam cells list by highest intensity, greedily eliminate 
    points that are within 'elim_dist' of the brightest point. Pop the brightest point off the list
    and add to a new list. Continue until the initial list is empty.
    """
    all_seam_cells_list = get_seam_cell_list(time_vol, time)
    all_seam_cells_list.sort(key=lambda x: (x[0], x[4]), reverse=True)
   
    final_seam_cells_list = []

    for i in range(len(all_seam_cells_list)):
        deleted_count = 0 # new count when there is a new top cell
        if len(all_seam_cells_list) == 0:
            break
        top_cell = all_seam_cells_list[0]
        for j in range(1, len(all_seam_cells_list)):
            j = j - deleted_count
            if j >= len(all_seam_cells_list):
                break
            curr_cell = all_seam_cells_list[j]
            # if the points are of the same time volume
            if top_cell[0] == curr_cell[0]:
                # if point is within this euclidean distance, eliminate it.
                if euclid_dist(top_cell[1:4], curr_cell[1:4]) <= elim_dist:
                    all_seam_cells_list.pop(j)
                    deleted_count += 1
            else:
                # means we reached end of a time vol, move onto next top cell
                break
        all_seam_cells_list.pop(0)
        final_seam_cells_list.append(top_cell)
    print(final_seam_cells_list)
    print(len(final_seam_cells_list))
    return final_seam_cells_list

def f1_score(tp, fp, fn):
    """
    Computes the F1 score using true positive, false positive, and false negative scores
    """
    return tp / (tp + (0.5 * (fp + fn)))

def test_f1(large_predicted_volume):
    true_pos = 0
    false_neg = 0
    false_pos = 0
    
    large_predicted_volume=large_predicted_volume[0,0]
    test_csv_path = "/groups/funke/home/tame/data/test.csv"
    test_data = np.genfromtxt(test_csv_path, delimiter=' ')
    gt_locations = get_gt_seam_cell_locations()

    for time in tqdm(range(20, large_predicted_volume.shape[0])):
        # part 1
        time_vol = large_predicted_volume[time]
        true_pos_and_false_neg = count_bright(time_vol, test_data, time)
        curr_true_pos = true_pos_and_false_neg[0]
        false_neg += true_pos_and_false_neg[1]
        matches = count_valid_matches(time_vol, gt_locations[time - 20])
        true_pos += curr_true_pos

        # part 2
        pred_local_max = greedy_filter(time_vol, time)
        temp = len(pred_local_max) - matches
        if temp < 0:
            temp = 0
        false_pos += temp
        print("True Positives: ", true_pos)
        print("False Negatives: ", false_neg)
        print("False Positives: ", false_pos)
        print("Matches: ", matches)

    return f1_score(true_pos, false_pos, false_neg)


print(test_f1(large_predicted_volume))