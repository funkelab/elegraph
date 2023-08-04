import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D
import glob
import torch
from torch_tps import ThinPlateSpline
from sklearn.preprocessing import normalize

def graph(cells):
    """
    Graphs a 3D representation of the worm by connecting seam cells.
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(cells) - 2):
        x1,x2 = cells[i,0], cells[i+2,0]
        y1,y2 = cells[i,1], cells[i+2,1]
        z1,z2 = cells[i,2], cells[i+2,2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], c="b", label='Line Connecting Points')
        ax.scatter(x1, y1, z1, c='r', marker='o')
        ax.scatter(x2, y2, z2, c='r', marker='o')
    for i in range(len(cells) // 2):
        i = i * 2
        x1,x2 = cells[i,0], cells[i+1,0]
        y1,y2 = cells[i,1], cells[i+1,1]
        z1,z2 = cells[i,2], cells[i+1,2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], c="r", label='Line Connecting Points')
        ax.scatter(x1, y1, z1, c='r', marker='o')
        ax.scatter(x2, y2, z2, c='r', marker='o')
    ax.set_title("3D plot")
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    plt.show()

def assemble_T(shift, rotation):
    """
    shift is a 1x3 numpy array.
    rotation is 3x3 numpy array.
    Returns a 4x4 transformation matrix. 
    """
    T = np.eye(4,4)
    T[:3,:3] = rotation
    T[:3, 3] = shift.transpose()
    return T

def sample_shift(min_shift=-0.05, max_shift=0.05): #0.075, 0.025
    """
    Returns a randomly sampled 1x3 numpy array for a shift.
    """
    return np.random.uniform(min_shift,max_shift,3)

def sample_rotation_axis(theta_min=-(np.pi * 90/180), theta_max=np.pi * 90/180, axis="z"):
    """
    Returns a randomly sampled 3x3 numpy array for a rotation.
    """
    theta = np.random.uniform(theta_min,theta_max,1)[0]
    if axis == "x":
        rotation = np.array([[1,0,0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    elif axis == "y":
        rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == "z":
        rotation = np.array([[np.cos(theta),-np.sin(theta),0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return rotation

def turn_into_points(ctrl_points):
    """
    Returns a Nx1 version numpy array of ctrl_points.
    """
    points=np.array([0,0,0])
    for layer in ctrl_points:
        for i in range(4):
            points = np.vstack((points, layer[:3, i]))
    return points[1:]

def build_worm(layers=3, base_ctrl_pts=None, min_shift=0, max_shift=0):
    """
    Return numpy array of transformed, layered control points.
    """
    ctrl_pts = [] 
    base_ctrl_pts=np.hstack((base_ctrl_pts, np.ones((4,1)))).transpose()
    ctrl_pts.append(base_ctrl_pts)
    while len(ctrl_pts) < layers:
        temp = np.matmul(sample_rotation_axis(axis="x"), sample_rotation_axis(axis="y"))
        rotation = np.matmul(temp, sample_rotation_axis(axis="z"))
        T = assemble_T(sample_shift(min_shift=min_shift, max_shift=max_shift), rotation)
        proposed_pt = np.matmul(T, ctrl_pts[-1])
        if len(ctrl_pts) > 1:
            hull = turn_into_points(ctrl_pts)
            if not isinstance(hull,Delaunay):
                hull = Delaunay(hull)
            # positive/0 value mean a point lies in or on the hull boundary
            # negative value mean a point lies outside the hull boundary
            points = proposed_pt.transpose()[:,:3]
            check = hull.find_simplex(points)>=0
            check_segment = hull.find_simplex(get_all_segment_points(points[0], points[1], points[2], points[3], 5))>=0
            if sum(check) == 0 and sum(check_segment) == 0:
                ctrl_pts.append(proposed_pt)
        else:
            # first layer
            temp = base_ctrl_pts.copy()
            temp[:3,:] += 0.05 # 0.05
            ctrl_pts.append(temp)
            hull = turn_into_points(ctrl_pts)
            if not isinstance(hull,Delaunay):
                hull = Delaunay(hull)
            check = hull.find_simplex(proposed_pt.transpose()[:,:3])>=0
            if sum(check) == 0:
                ctrl_pts.append(proposed_pt)
            ctrl_pts.pop(1)
    return np.array(ctrl_pts)

# starting control points should be equidistant around worm, same number as layers * 4
def set_starting_ctrl(layers=3, base_ctrl_pts=None):
    """
    Assume worm has same Y coordinates. Each layer should have same 4 y coords.
    Same order of base_ctrl_pts as in build_worm()
    Returns numpy array of starting, layered control points.
    """
    # order of these should match how the target points are set
    starting_ctrl = np.array([0,0,0])
    for i in base_ctrl_pts:
        starting_ctrl = np.vstack((starting_ctrl, i))
    z_splits = np.linspace(0.0,0.1,layers)[1:]
    for z in z_splits:
        points = base_ctrl_pts.copy()
        for p in points:
            p[2] = z
            starting_ctrl = np.vstack((starting_ctrl, p))
    starting_ctrl=starting_ctrl[1:]
    return starting_ctrl

def transform(frame,layers, width=0.035, xz_min=0.0, xz_max=0.1, min_shift=-0.05, max_shift=0.05):
    """
    frame is a N x 3 numpy array of seam cell locations.
    layers is an integer representing number of control point layers.
    Returns a tensor of the transformed seam cell locations.
    """
    # abs(shift value) must be greater than width to avoid edge case where control points always stay in hull
    assert abs(min_shift) > width
    assert abs(max_shift) > width
    # normalize frame
    frame = normalize(frame, axis=0, norm="l1")
    y = frame[:, 1]
    # important base control points! sets how wide our points should be avoiding other points
    y_min, y_max = y[0] - width, y[0] + width
    base_ctrl_pts = np.array([[xz_min,y_min,xz_min], [xz_max,y_min,xz_min], [xz_min,y_max,xz_min], [xz_max,y_max,xz_min]])
    starting_ctrl = set_starting_ctrl(layers=layers, base_ctrl_pts=base_ctrl_pts)
    worm = build_worm(layers=layers, base_ctrl_pts=base_ctrl_pts, min_shift=min_shift, max_shift=max_shift)
    worm = turn_into_points(worm)
    tps = ThinPlateSpline(alpha=0.0)
    starting_ctrl_tensor = torch.from_numpy(starting_ctrl).float()
    worm_tensor = torch.from_numpy(worm).float()
    frame_tensor = torch.from_numpy(frame).float()
    tps.fit(starting_ctrl_tensor, worm_tensor)
    transformed_points = tps.transform(frame_tensor)
    return transformed_points

import random
def random_pts_between(pointA, pointB, n):
    x1, y1, z1 = pointA
    x2, y2, z2 = pointB
    points = []
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    for _ in range(n):
        t = random.uniform(0, 1)
        x = x1 + t * dx
        y = y1 + t * dy
        z = z1 + t * dz
        points.append((x, y, z))
    return np.array(points)

def get_all_segment_points(pointA, pointB, pointC, pointD, n):
    all_points = random_pts_between(pointA, pointB, n)
    all_points = np.vstack((all_points, random_pts_between(pointA, pointC, n)))
    all_points = np.vstack((all_points, random_pts_between(pointB, pointD, n)))
    all_points = np.vstack((all_points, random_pts_between(pointC, pointD, n)))
    return all_points

def graph(array):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = array[:,0]
    y = array[:,1]
    z = array[:,2]
    ax.scatter(x, y, z, c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud')
    plt.show()
    plt.savefig("testttt.png")

if __name__ == "__main__":
    # pointA = np.array([1,2,3])
    # pointB = np.array([5,5,5])
    # n = 10
    # random_points_on_segment = generate_random_points_on_segment(pointA, pointB, n)
    # print(random_points_on_segment)

    # all_points = np.vstack((pointA, pointB))
    # all_points = np.vstack((all_points, random_points_on_segment))
    # graph(all_points)
    # np.savetxt("testttt.csv", all_points, delimiter=',')
    train_path=sorted(glob.glob("/groups/funke/home/tame/data/Untwisted/SeamCellCoordinates/*.csv"))
    actual_path=sorted(glob.glob("/groups/funke/home/tame/data/seamcellcoordinates/*.csv"))
    node_positions = np.genfromtxt(train_path[21], delimiter=',')[1:, 1:4]
    actual_positions = np.genfromtxt(actual_path[21], delimiter=',')[1:, 1:4]
    np.random.seed(3)
    for i in range(5,11):
        transformed_points = transform(node_positions, 8)
        np.savetxt("cells" + str(i) + ".csv", transformed_points, delimiter=',')
