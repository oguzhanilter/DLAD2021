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

def visualize_3D_box(image, xy):
    """ Visualize 3D box by using edge points xy on image.

    Args:
        image:  Type does not matter (numpy, list ...)
                3D matrix of an RGB image
        xy:     2x(8*NumberOfPoints) (numpy, list ...) 
                First row : x in image coordinates
                Second row: y in image coordinates
                There are 8*NumberOfObjects columns. Every consecutive group of 8
                defines a bounding box for an object. 
    """ 
    xy = xy.astype(int)
    g = (0,255,0)
    p1 = (0,0,0,3,3,3,5,5,5,6,6,6)
    p2 = (1,2,4,1,2,7,1,4,7,7,2,4)

    for i in range(int(len(xy[0])/8)):
        edges = list()
        for j in range(8):
            edges.append( (xy[0, 8*i+j], xy[1, 8*i+j]) )
        
        for j in range(len(p1)):
            image = cv2.line(image, edges[p1[j]], edges[p2[j]], g, 2)

    return image
    

def visualize(image, xy, labels, color_map):
    """ Visualize 2D points on image according to their labels and color map.

    Args:
        image:      Type does not matter (numpy, list ...)
                    3D matrix of an RGB image
        xy:         3xNumberOfPoints (float) (numpy, list)
                    First row : x in image coordinates and for sure in frame 
                    Second row: y in image coordinates and for sure in frame
                    Third row : index of the point in the original original data source
        labels:     (numpy.array)
                    object that gives the semantic label of each point within the scene.  

        color_map:  dictionary
                    maps numeric semantic labels to a BGR color for visualization   
    """       
    xy = xy.astype(int)
    for i in range(len(xy[0])):
        color = color_map[labels[xy[2,i],0]][::-1]
        image = cv2.circle(image, (xy[1,i], xy[0,i]), radius=1, color=color, thickness=-1)
        #image[xy[0,i], xy[1,i],:] = [color[2], color[1],color[0]]

    return np.asarray(image).astype(np.uint8)


def filter_indices_xy(xy, image_size):
    """ Filter the points if they are not projected inside the frame of the image.

    Args:
        xy:         3xNumberOfPoints (float) (numpy, list)
                    First row : x in image coordinates 
                    Second row: y in image coordinates 
                    Third row : index of the point in the original original data source
        image_size: (tuple of two int)
                    Size of the image in pixel 
                    First element x; Second element y  
    """      

    x = xy[1,:]
    y = xy[0,:]
    indices = xy[2,:]

    inside_frame_x = np.logical_and((x > 0), (x < image_size[0]))
    inside_frame_y = np.logical_and((y > 0), (y < image_size[1]))
    inside_frame_indices = np.argwhere(np.logical_and(inside_frame_x,inside_frame_y)).flatten()

    filtered_xy = np.stack((x[inside_frame_indices],y[inside_frame_indices],indices[inside_frame_indices]))
    
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
    """   
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
    """ Calculates edges of the 3D boxes for given object location and dimensions

    Args:
        xy: contains a list of lists with 16 elements for each object

    """   

    number_of_objects = len(objects)
    box_edges = np.ones((4, 8*number_of_objects))

    obj_edges = np.ones((3, 8))
    
    # This order is important. This order is used in visualization part 
    # to decide which points have edge between them.

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

        # Define edges as if the object is at the origin
        obj_edges[0,x1] = - object_dim[2]/2 
        obj_edges[0,x2] = object_dim[2]/2 

        obj_edges[1,y1] = 0
        obj_edges[1,y2] = - object_dim[0]

        obj_edges[2,z1] = object_dim[1]/2
        obj_edges[2,z2] = - object_dim[1]/2

        # Rotate the edges
        obj_edges = np.dot(Ry, obj_edges)
        
        # Shift the center of the object true position
        obj_edges[0,:] += object_center[0]
        obj_edges[1,:] += object_center[1]
        obj_edges[2,:] += object_center[2]   

        box_edges[0:3, 8*i : 8*i+8 ] = obj_edges

    return box_edges


if __name__ =="__main__":
    print("**** Running task2 ****")

    data = get_data()

    velo_point_cloud    = data['velodyne'] 
    velo_mat_T          = data['T_cam0_velo']   # extrinsic camera parameters
    cam_mat_P           = data['P_rect_20']     # intrinsic camera parameters
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
    image_with_3D_box    = visualize_3D_box(new_image,object_box_points_2D)

  
    if Q1:
        im = Image.fromarray(new_image)
        im.show()

    if Q2:
        im = Image.fromarray(image_with_3D_box)
        im.show()
    