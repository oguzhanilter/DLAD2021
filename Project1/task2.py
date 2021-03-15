import os
import cv2
import data_utils
import numpy as np
from PIL import Image
from load_data import load_data

RELATIVE_PATH_TO_DATA = 'data'
DATA_FILE_NAME = 'data.p'
Q1 = False
Q2 = True

def get_data():
    script_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(script_path, RELATIVE_PATH_TO_DATA,DATA_FILE_NAME)
    data = load_data(data_path)
    return data

def visualize_3D_box(image, filtered_xy):

    filtered_xy = filtered_xy.astype(int)

    for i in range(int(len(filtered_xy[0])/8)):

        point0 = (filtered_xy[0, 8*i], filtered_xy[1, 8*i])
        point1 = (filtered_xy[0, 8*i+1], filtered_xy[1, 8*i+1])
        point2 = (filtered_xy[0, 8*i+2], filtered_xy[1, 8*i+2])
        point3 = (filtered_xy[0, 8*i+3], filtered_xy[1, 8*i+3])
        point4 = (filtered_xy[0, 8*i+4], filtered_xy[1, 8*i+4])
        point5 = (filtered_xy[0, 8*i+5], filtered_xy[1, 8*i+5])
        point6 = (filtered_xy[0, 8*i+6], filtered_xy[1, 8*i+6])
        point7 = (filtered_xy[0, 8*i+7], filtered_xy[1, 8*i+7])

        #image = cv2.rectangle(image,point0,point3,(0,255,0),1)
        #image = cv2.rectangle(image,point5,point6,(0,255,0),1)
        image = cv2.line(image, point0, point1, (0,255,0),1)
        image = cv2.line(image, point1, point3, (0,255,0),1)
        image = cv2.line(image, point3, point2, (0,255,0),1)
        image = cv2.line(image, point2, point0, (0,255,0),1)

        image = cv2.line(image, point4, point5, (0,255,0),1)
        image = cv2.line(image, point5, point7, (0,255,0),1)
        image = cv2.line(image, point7, point6, (0,255,0),1)
        image = cv2.line(image, point6, point4, (0,255,0),1)

        image = cv2.line(image, point1, point5, (0,255,0),1)
        image = cv2.line(image, point3, point7, (0,255,0),1)
        image = cv2.line(image, point0, point4, (0,255,0),1)
        image = cv2.line(image, point2, point6, (0,255,0),1)
    """
    i= 0 
    point0 = (filtered_xy[0, 8*i], filtered_xy[1, 8*i])
    point1 = (filtered_xy[0, 8*i+1], filtered_xy[1, 8*i+1])
    point2 = (filtered_xy[0, 8*i+2], filtered_xy[1, 8*i+2])
    point3 = (filtered_xy[0, 8*i+3], filtered_xy[1, 8*i+3])
    point4 = (filtered_xy[0, 8*i+4], filtered_xy[1, 8*i+4])
    point5 = (filtered_xy[0, 8*i+5], filtered_xy[1, 8*i+5])
    point6 = (filtered_xy[0, 8*i+6], filtered_xy[1, 8*i+6])
    point7 = (filtered_xy[0, 8*i+7], filtered_xy[1, 8*i+7])

    image = cv2.rectangle(image,point0,point3,(0,255,0),1)
    image = cv2.rectangle(image,point5,point6,(0,255,0),1)
    image = cv2.line(image, point1, point5, (0,255,0),1)
    image = cv2.line(image, point3, point7, (0,255,0),1)
    image = cv2.line(image, point0, point4, (0,255,0),1)
    image = cv2.line(image, point2, point6, (0,255,0),1) 
    
    # image = cv2.circle(image, point4, 3, (255, 0, 0), -1)
    # image = cv2.circle(image, point5, 3, (0, 0, 0), -1)
    # image = cv2.circle(image, point6, 3, (255, 255, 0), -1)
    # image = cv2.circle(image, point7, 3, (0, 255, 0), -1)
    #image = cv2.circle(image, point5, 3, (0, 0, 255), -1)
    #image = cv2.circle(image, point6, 3, (255, 255, 255), -1)
    """
    
    return image


def visualize(image, filtered_xy, point_labels, label_color_map):   
    
    filtered_xy = filtered_xy.astype(int)

    for i in range(len(filtered_xy[0])):
        color = label_color_map[point_labels[filtered_xy[2,i],0]]
        image[filtered_xy[0,i], filtered_xy[1,i],:] = [color[2], color[1],color[0]]

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
def projection_3D_2D(points, extrinsic, intrinsic,should_in_frame = True):


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

def object_box_points(objects):

    number_of_objects = len(objects)
    box_edges = np.ones((4, 8*number_of_objects))

    obj_edges = np.ones((3, 8))
    
    x1 = [0,1,2,3]
    x2 = [4,5,6,7]
    z1 = [0,2,4,6]
    z2 = [1,3,5,7]
    y1 = [0,1,4,5]
    y2 = [2,3,6,7]

    for i in range(number_of_objects):

        ty              = objects[i][14]
        object_dim      = [objects[i][8], objects[i][9], objects[i][10]] #  height, width, length 
        object_center   = [objects[i][11], objects[i][12], objects[i][13]]
        
        Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
        # Ry = Ry.reshape([3,3])


        obj_edges[0,x1] = - object_dim[2]/2 # x
        obj_edges[0,x2] = object_dim[2]/2 # x

        obj_edges[1,y1] = 0
        obj_edges[1,y2] = - object_dim[0]

        obj_edges[2,z1] = object_dim[1]/2
        obj_edges[2,z2] = - object_dim[1]/2

        obj_edges = np.dot(Ry, obj_edges)
        
        obj_edges[0,x1] += object_center[0] 
        obj_edges[0,x2] += object_center[0] 

        obj_edges[1,y1] += object_center[1] 
        obj_edges[1,y2] += object_center[1]

        obj_edges[2,z1] += object_center[2] 
        obj_edges[2,z2] += object_center[2] 
        box_edges[0:3, 8*i : 8*i+8 ] = obj_edges

    return box_edges


if __name__ =="__main__":
    print("**** Running task2 ****")

    data = get_data()

    velo_point_cloud    = data['velodyne']
    cam_mat_K           = data['K_cam2']  
    velo_mat_T          = data['T_cam2_velo']   # extrinsic camera parameters
    cam_mat_P           = data['P_rect_20']     # intrinsic camera parameters
    velo_mat_T          = data['T_cam2_velo']
    cam_image           = data['image_2']
    labels              = data['labels']
    label_color_map     = data['color_map']
    point_labels        = data['sem_label']
    objects             = data['objects']

   
    xy          = projection_3D_2D(velo_point_cloud,velo_mat_T,cam_mat_P)
    filtered_xy = filter_indices_xy(xy, cam_image.shape)
    new_image   = visualize(cam_image, filtered_xy, point_labels, label_color_map)

    object_box_points_3D = object_box_points(objects)
    object_box_points_2D = np.matmul(cam_mat_P,object_box_points_3D)
    object_box_points_2D = object_box_points_2D / object_box_points_2D[2,None]
    # object_box_points_2D = projection_3D_2D(object_box_points_3D, None,cam_mat_P, False)
    image_with_3D_box    = visualize_3D_box(new_image,object_box_points_2D)

  
    if Q1:
        im = Image.fromarray(new_image)
        im.show()

    if Q2:
        im = Image.fromarray(image_with_3D_box)
        im.show()
    