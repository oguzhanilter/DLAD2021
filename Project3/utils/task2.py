from os import replace
from typing import OrderedDict
import numpy as np
from numpy.core.numeric import indices


def rot_matrix(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])


def label2corners(label, delta):
    '''
    Task 1
    input
        label (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
    output
        corners (N,8,3) corner coordinates in the rectified reference frame
    '''
    N = label.shape[0]
    corners = np.empty((N,8,3))

    i = 0
    for bb in label:
        R = rot_matrix(bb[6])
        h,w,l = bb[3:6]
    
        h += delta
        w += 2*delta
        l += 2*delta


        # TODO: WHY here -h but not in task 1 !!!! 

        x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
        y_corners = [-h,-h,-h,-h,0,0,0,0]
        z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
        corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
        corners_3d[0,:] = corners_3d[0,:] + bb[0]
        corners_3d[1,:] = corners_3d[1,:] + bb[1]
        corners_3d[2,:] = corners_3d[2,:] + bb[2]
        corners_3d = np.transpose(corners_3d)
        corners[i,:,:] = corners_3d
        i += 1

    return corners

def create_set(indices, max_points):

    len_indices = len(indices)

    if len_indices > max_points:
        indices = np.random.choice(indices, max_points, replace=False)
    
    elif len_indices < max_points:
        extend = np.random.choice(indices, max_points - len_indices, replace=True)
        indices = np.append(indices, extend)

    return indices


def roi_pool(pred, xyz, feat, config):
    '''
    Task 2
    a. Enlarge predicted 3D bounding boxes by delta=1.0 meters in all directions.
       As our inputs consist of coarse detection results from the stage-1 network,
       the second stage will benefit from the knowledge of surrounding points to
       better refine the initial prediction.
    b. Form ROI's by finding all points and their corresponding features that lie 
       in each enlarged bounding box. Each ROI should contain exactly 512 points.
       If there are more points within a bounding box, randomly sample until 512.
       If there are less points within a bounding box, randomly repeat points until
       512. If there are no points within a bounding box, the box should be discarded.
    input
        pred (N,7) bounding box labels
        xyz (N,3) point cloud
        feat (N,C) features
        config (dict) data config
    output
        valid_pred (K',7)
        pooled_xyz (K',M,3)
        pooled_feat (K',M,C)
            with K' indicating the number of valid bounding boxes that contain at least
            one point
    useful config hyperparameters
        config['delta'] extend the bounding box by delta on all sides (in meters)
        config['max_points'] number of points in the final sampled ROI
    '''
    N = pred.shape[0]
    M = config['max_points']
    C = feat.shape[1]

    print(pred.shape)

    valid_pred = np.empty((0,7))
    pooled_xyz = np.empty((0,M,3))
    pooled_feat = np.empty((0,M,C))

    pred_corners = label2corners(pred, config['delta'])

    
    for ind in range(N):

        corners = pred_corners[ind]

        x_min, x_max = np.min(corners[:,0]), np.max(corners[:,0])
        y_min, y_max = np.min(corners[:,1]), np.max(corners[:,1])
        z_min, z_max = np.min(corners[:,2]), np.max(corners[:,2])

        indices = np.argwhere((xyz[:,0]>=x_min) & (xyz[:,0]<=x_max) &
                              (xyz[:,1]>=y_min) & (xyz[:,1]<=y_max) &
                              (xyz[:,2]>=z_min) & (xyz[:,2]<=z_max))


        if len(indices) == 0:
            continue

        indices  = indices.reshape(len(indices))

        valid_pred = np.append(valid_pred, [pred[ind]], axis=0)
        
        indices = create_set(indices, config['max_points'])

        inlier_points = xyz[indices]
        inlier_feat   = feat[indices]

        pooled_xyz  = np.append(pooled_xyz, [inlier_points], axis=0 )
        pooled_feat = np.append(pooled_feat, [inlier_feat], axis=0 )

    return valid_pred, pooled_xyz, pooled_feat