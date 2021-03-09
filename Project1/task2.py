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

def visualize(image, filtered_xy, point_labels, label_color_map):   
    
    filtered_xy = filtered_xy.astype(int)

    for i in range(len(filtered_xy[0])):
        image[filtered_xy[0,i], filtered_xy[1,i],:] = label_color_map[point_labels[filtered_xy[2,i],0]]

    return np.asarray(image).astype(np.uint8)

def filter_indices_xy(xy, image_size):

    x = xy[1,:]
    y = xy[0,:]
    indices = xy[2,:]

    inside_frame_x = np.logical_and((x > 0), (x < image_size[0]))
    inside_frame_y = np.logical_and((y > 0), (y < image_size[1]))
    inside_frame_indices = np.argwhere(np.logical_and(inside_frame_x,inside_frame_y)).flatten()

    filtered_xy = np.stack((x[inside_frame_indices],y[inside_frame_indices],indices[inside_frame_indices]))
    
    return filtered_xy 

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

    xy[2,:] = front_hemisphere_indices

    return xy


if __name__ =="__main__":
    print("**** Running task2 ****")

    data = get_data()

    velo_point_cloud    = data['velodyne']
    cam_mat_K           = data['K_cam2']  
    velo_mat_T          = data['T_cam2_velo']   # extrinsic camera parameters
    cam_mat_P           = data['P_rect_20']     # intrinsic camera parameters
    cam_image           = data['image_2']
    labels              = data['labels']
    label_color_map     = data['color_map']
    point_labels        = data['sem_label']
    objects             = data['objects']


    xy = projection(velo_point_cloud,velo_mat_T,cam_mat_P)
    filtered_xy = filter_indices_xy(xy, cam_image.shape)
    new_image = visulize(cam_image, filtered_xy, point_labels, label_color_map)
    im = Image.fromarray(new_image)
    im.show()