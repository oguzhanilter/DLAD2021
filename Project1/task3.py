import os
import cv2
import data_utils
import numpy as np
from PIL import Image
from load_data import load_data

RELATIVE_PATH_TO_DATA = 'data'
DATA_FILE_NAME = 'demo.p'
Q1 = False
Q2 = True

def get_data():
    script_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(script_path, RELATIVE_PATH_TO_DATA,DATA_FILE_NAME)
    data = load_data(data_path)
    return data


def visualize(image, xy, color_map):
    """ Visualize 2D points on image according to their labels and color map.

    Args:
        image:      Type does not matter (numpy, list ...)
                    3D matrix of an RGB image
        xy:         3xNumberOfPoints (float) (numpy, list)
                    First row : x in image coordinates and for sure in frame 
                    Second row: y in image coordinates and for sure in frame
                    Third row : index of the point in the original original data source
        color_map:  dictionary
                    maps numeric semantic labels to a BGR color for visualization   
    """
    angles = xy[3,:]*57.295779513
    bins = int((max(angles)-min(angles))/0.18)
    a = angles[:-1] 
    b = angles[1:]
    diff = abs(angles[-1] - angles[1:])
   
    _ , bin_edges = np.histogram(angles, bins=bins)
    xy= xy[0:3,:].astype(int)
    x       = xy[0,:]
    y       = xy[1,:]

    for i in range(bins):
        if i != bins:
            group       = np.logical_and( ( angles>=bin_edges[i] ) , ( angles<=bin_edges[i+1] ) )
        else:
            group       = np.logical_and( ( angles>=bin_edges[i] ) , ( angles<bin_edges[i+1] ) )

        group_inds  = np.argwhere(group).flatten()

        sorted_inds = x[group_inds].argsort()
        fliped_inds = np.flip(sorted_inds)

        inds = group_inds[fliped_inds]
              

        xy[2, inds] = i%4

    for i in range(len(xy[0])):
        
    
        color = color_map[xy[2, i]]
        image = cv2.circle(image, (xy[1,i], xy[0,i]), radius=1, color=color, thickness=-1)
        #image[xy[0,i], xy[1,i],:] = [color[2], color[1],color[0]]

    return np.asarray(image).astype(np.uint8)


def filter_indices_xy(xy, image_size):
    """ Filter the points if they are not projected inside the frame of the image.

    Args:
        xy:         4xNumberOfPoints (float) (numpy, list)
                    First row : x in image coordinates 
                    Second row: y in image coordinates 
                    Third row : index of the point in the original original data source
                    Forth row : angles on the LIDAR frame   
        image_size: (tuple of two int)
                    Size of the image in pixel 
                    First element x; Second element y  
    Returns: 
        filtered_xy: 4xNumberOfPoints (float) (numpy, list)
                    Points that are inside the frame
                    First row : x in image coordinates 
                    Second row: y in image coordinates 
                    Third row : index of the point in the original original data source
                    Forth row : angles on the LIDAR frame   
    """      

    x = xy[1,:]
    y = xy[0,:]
    indices = xy[2,:]
    angles = xy[3,:]

    inside_frame_x = np.logical_and((x > 0), (x < image_size[0]))
    inside_frame_y = np.logical_and((y > 0), (y < image_size[1]))
    inside_frame_indices = np.argwhere(np.logical_and(inside_frame_x,inside_frame_y)).flatten()

    filtered_xy = np.stack((x[inside_frame_indices],y[inside_frame_indices],indices[inside_frame_indices],angles[inside_frame_indices] ))
    
    return filtered_xy 


def projection_3D_2D(points, extrinsic, intrinsic):
    """ Projection of 3D points to 2D image
    [x;y;1] = K*[R|t] * [X;Y;Z;1] = cam_mat_P * velo_mat_T * [X;Y;Z;1]

    Args:
        points:     (num points x 4) numpy.array object.
                    First column : x 
                    Second column: y 
                    Third column : z
                    Forth column : reflection
        extrinsic:  4x4 (numpy.array)
                    The homogeneous velodyne to rectified camera coordinate transformations
        instrinsic: 3x4 (numpy.array)
                    The intrinsic projection matrices to Cam X after rectification
    Return:
        xy:         (num points x 4) numpy.array object.
                    First row : x 
                    Second row: y 
                    Third row : indices of original dataset
                    Forth row : angles on the LIDAR frame     

    """   
    front_hemisphere = points[:, 0] > 0 
    front_hemisphere_indices = np.argwhere(front_hemisphere).flatten()

    front_hemisphere_x = points[front_hemisphere_indices, 0]
    front_hemisphere_y = points[front_hemisphere_indices, 1]
    front_hemisphere_z = points[front_hemisphere_indices, 2]
    front_hemisphere_angles = np.arctan2(front_hemisphere_y,front_hemisphere_x)
    homo_coor = np.ones(len(front_hemisphere_x))

    XYZ = np.stack((front_hemisphere_x,front_hemisphere_y,front_hemisphere_z,homo_coor))
        
    projection_matrix = np.matmul(intrinsic,extrinsic) 
        
    xy = np.matmul(projection_matrix,XYZ)
    xy = xy / xy[2,None]

    xy[2,:] = front_hemisphere_indices
    xy = np.vstack((xy, front_hemisphere_angles))  

    return xy

if __name__ =="__main__":
    print("**** Running task3 ****")

    data = get_data()

    velo_point_cloud    = data['velodyne']
    velo_mat_T          = data['T_cam2_velo']   # extrinsic camera parameters
    cam_mat_P           = data['P_rect_20']     # intrinsic camera parameters
    cam_image           = data['image_2']

    label_color_map = [[255,0,0], [0,255,0], [0,0,255,], [255,0,255]]

    xy          = projection_3D_2D(velo_point_cloud,velo_mat_T,cam_mat_P)
    filtered_xy = filter_indices_xy(xy, cam_image.shape)
    new_image   = visualize(cam_image, filtered_xy, label_color_map)

    im = Image.fromarray(new_image)
    im.show()


    a  = 10 

   
