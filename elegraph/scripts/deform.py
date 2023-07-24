import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D

def spline_graph(points, control, filename):
    colors=["r", "g", "b", "m", "c"]
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for i in range(0,len(points), 4):
        # control points
        x = control[i:i+4,0]
        y = control[i:i+4,1]
        z = control[i:i+4,2]
        ax.scatter(x, y, z, c=colors[i//4], marker='o')
    
    # seam cell coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    ax.scatter(x, y, z, c="k", marker='*')
        
    ax.set_title('Point Cloud')
    plt.show()
    plt.savefig(filename)

def assemble_T(shift, rotation):
    """
    shift is a 1x3 numpy array
    rotation is 3x3 numpy array
    """
    # print("Rotation: ", rotation)
    T = np.eye(4,4)
    T[:3,:3] = rotation
    T[:3, 3] = shift.transpose()
    return T

# test if making max smaller fixes issue of tight worm
def sample_shift(min_val=0.0, max_val=0.1):
    r = np.random.uniform(min_val,max_val,3)
    # r[0] = 0
    # r[1] = 0
    return r

def sample_rotation_axis(theta_min=-(np.pi * 60/180), theta_max=np.pi * 60/180, axis="z"):
    theta = np.random.uniform(theta_min,theta_max,1)[0]
    if axis == "x":
        rotation = np.array([[1,0,0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    elif axis == "y":
        rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == "z":
        rotation = np.array([[np.cos(theta),-np.sin(theta),0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return rotation

def is_intersecting(new_ctrl_pts, prev_ctrl_pts_list):
    new_ctrl_pts=new_ctrl_pts.transpose()[:,:3]
    print(new_ctrl_pts)
    min_x_new = np.min(new_ctrl_pts[:, 0])
    max_x_new = np.max(new_ctrl_pts[:, 0])
    min_y_new = np.min(new_ctrl_pts[:, 1])
    max_y_new = np.max(new_ctrl_pts[:, 1])
    min_z_new = np.min(new_ctrl_pts[:, 2])
    max_z_new = np.max(new_ctrl_pts[:, 2])
    print((min_x_new, max_x_new))
    print((min_y_new, max_y_new))
    print((min_z_new, max_z_new))

    for pt in prev_ctrl_pts_list:
        pt = pt.transpose()[:, :3]
        print(pt)
        min_x_pt = np.min(new_ctrl_pts[:, 0])
        max_x_pt = np.max(new_ctrl_pts[:, 0])
        min_y_pt = np.min(new_ctrl_pts[:, 1])
        max_y_pt = np.max(new_ctrl_pts[:, 1])
        min_z_pt = np.min(new_ctrl_pts[:, 2])
        max_z_pt = np.max(new_ctrl_pts[:, 2])
        print((min_x_pt, max_x_pt))
        print((min_y_pt, max_y_pt))
        print((min_z_pt, max_z_pt))
    
        count = 0
        if min_x_new >= min_x_pt or max_x_new <= max_x_pt:
            count += 1
        if min_y_new >= min_y_pt or max_y_new <= max_y_pt:
            count += 1
        if min_z_new >= min_z_pt or max_z_new <= max_z_pt:
            count += 1
        if count == 3:
            return True
    return False

def turn_into_points(ctrl_points):
    points=np.array([0,0,0])
    for layer in ctrl_points:
        for i in range(4):
            points = np.vstack((points, layer[:3, i]))
    return points[1:]

def build_worm(layers=3, base_ctrl_pts=None):
    
    ctrl_pts = [] 

    # find all straight layers of worm
    #base_ctrl_pts=np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]]) # this needs to be a square that is a good starting place for every worm
    base_ctrl_pts=np.hstack((base_ctrl_pts, np.ones((4,1)))).transpose()
    # print(f"Base is {base_ctrl_pts}")
    # print("="*10)

    ctrl_pts.append(base_ctrl_pts)
    while len(ctrl_pts) < layers:
        temp = np.matmul(sample_rotation_axis(axis="x"), sample_rotation_axis(axis="y"))
        rotation = np.matmul(temp, sample_rotation_axis(axis="z"))
        T = assemble_T(sample_shift(), rotation)
        proposed_pt = np.matmul(T, ctrl_pts[-1])
        if len(ctrl_pts) > 1:
            hull = turn_into_points(ctrl_pts)
            if not isinstance(hull,Delaunay):
                hull = Delaunay(hull)
            check = hull.find_simplex(proposed_pt.transpose()[:,:3])>=0
            if sum(check) == 0:
                ctrl_pts.append(proposed_pt)
            else:
                print("hello")
        else:
            temp = base_ctrl_pts.copy()
            temp[:3,:] += 0.2
            ctrl_pts.append(temp)
            hull = turn_into_points(ctrl_pts)
            if not isinstance(hull,Delaunay):
                hull = Delaunay(hull)
            check = hull.find_simplex(proposed_pt.transpose()[:,:3])>=0
            if sum(check) == 0:
                ctrl_pts.append(proposed_pt)
            else:
                print("hello")
            ctrl_pts.pop(1)
    return np.array(ctrl_pts)

def graph_straightened(filename):
    frame = np.genfromtxt(filename, delimiter=',')[1:, 1:4]
    normalized_frame = normalize(frame, axis=0, norm="l1")
    np.savetxt("testing2.csv", frame, delimiter=',')
    np.savetxt("testing2norm.csv", normalized_frame, delimiter=',')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    x=frame[:,0]
    y=frame[:,1]
    z=frame[:,2]
    ax.scatter(x, y, z, c='b', marker='o')
    ax.set_title('Point Cloud')
    plt.show()
    plt.savefig("straightened.png")


# need to make sure control points are around the actual worm
import glob
import torch
from torch_tps import ThinPlateSpline
from sklearn.preprocessing import normalize

# starting control points should be equidistant around worm, same number as layers * 4
def set_starting_ctrl(layers=3, base_ctrl_pts=None):
    """
    Assume worm has same y coordinates. Each layer should have same 4 y coords.
    Same order of base_ctrl_pts as in build_worm()
    Returns numpy array of starting control point coordinates
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

# for a given frame...
# path_untwisted="/groups/funke/home/tame/data/Untwisted/SeamCellCoordinates/*.csv"
# cells=sorted(glob.glob(path_untwisted))

def transform(frame,layers):
    """
    frame is a N x 3 numpy array of seam cell locations
    layers is an integer representing number of control point layers

    returns a tensor of the transformed seam cell locations
    """
    # frame = np.genfromtxt(vol, delimiter=',')[1:, 1:4]
    frame = normalize(frame, axis=0, norm="l1") # make sure to normalize
    y = frame[:, 1]
    y_min, y_max = y[0] - 0.03, y[0] + 0.03
    base_ctrl_pts = np.array([[0,y_min,0], [0.1,y_min,0], [0,y_max,0], [0.1,y_max,0]])
    starting_ctrl = set_starting_ctrl(layers=layers, base_ctrl_pts=base_ctrl_pts)
    worm = build_worm(layers=layers, base_ctrl_pts=base_ctrl_pts)
    worm = turn_into_points(worm)
    tps = ThinPlateSpline(alpha=0.0)
    starting_ctrl_tensor = torch.from_numpy(starting_ctrl).float()
    worm_tensor = torch.from_numpy(worm).float()
    frame_tensor = torch.from_numpy(frame).float()
    tps.fit(starting_ctrl_tensor, worm_tensor)
    transformed_points = tps.transform(frame_tensor)
    return transformed_points

# spline_graph(frame, starting_ctrl, "testing.png")
# spline_graph(transformed_points, worm, "testing1.png")
# np.savetxt("cells.csv", frame, delimiter=',')
# np.savetxt("ctrl.csv", starting_ctrl, delimiter=',')
# np.savetxt("transformed_cells.csv", transformed_points, delimiter=',')
# np.savetxt("transformed_ctrl.csv", worm, delimiter=',')
