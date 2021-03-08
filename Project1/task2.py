import os
import data_utils
import numpy as np
from PIL import Image
from load_data import load_data

RELATIVE_PATH_TO_DATA = 'data'
DATA_FILE_NAME = 'data.p'

def get_data():
    script_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(script_path, RELATIVE_PATH_TO_DATA,DATA_FILE_NAME)
    data = load_data(data_path)
    return data

def find_label(xy, label_map):
    asd = 0

# [x;y;1] = K*[R|t] * [X;Y;Z;1] = cam_mat_P * velo_mat_T * [X;Y;Z;1]
def projection(points, extrinsic, intrinsic):

    front_hemisphere = points[:, 0] > 0 
    front_hemisphere_indices = np.argwhere(front_hemisphere).flatten()

    front_hemisphere_x = points[front_hemisphere_indices, 0]
    front_hemisphere_y = points[front_hemisphere_indices, 1]
    front_hemisphere_z = points[front_hemisphere_indices, 2]
    homo_coor = np.ones(len(front_hemisphere_x))

    XYZ = np.stack((front_hemisphere_x,front_hemisphere_y,front_hemisphere_z,homo_coor))

    projection_matrix = np.matmul(intrinsic,extrinsic)
    xy = np.matmul(projection_matrix,XYZ)
    xy = xy / xy[2,None]

    return xy[0:1,:]


if __name__ =="__main__":
    print("**** Running task2 ****")

    data = get_data()

    velo_point_cloud    = data['velodyne']
    cam_mat_K           = data['K_cam2']  
    velo_mat_T          = data['T_cam2_velo']   # extrinsic camera parameters
    cam_mat_P           = data['P_rect_30']     # intrinsic camera parameters
    cam_image           = data['image_2']
    labels              = data['labels']
    labels_color_map    = data['color_map']
    objects             = data['objects']

    xy = projection(velo_point_cloud,velo_mat_T,cam_mat_P)
    asd = 10