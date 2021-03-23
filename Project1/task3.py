import os
import cv2
import data_utils
import numpy as np
from PIL import Image
from load_data import load_data
import data_utils as du


RELATIVE_PATH_TO_DATA = 'data'
DATA_FILE_NAME = 'demo.p'
Q1 = False
Q2 = True

def get_data():
    script_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(script_path, RELATIVE_PATH_TO_DATA,DATA_FILE_NAME)
    data = load_data(data_path)
    return data


#Creating a Function.
def normal_dist(x , mean , sd):
    prob = (np.sqrt(2*np.pi)*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob

def visualize(image, xy, labels, color_map):
    """ Visualize 2D points on image according to their labels and color map.

    Args:
        image:      Type does not matter (numpy, list ...)
                    3D matrix of an RGB image
        xy:         3xNumberOfPoints (float) (numpy, list)
                    First row : x in image coordinates and for sure in frame 
                    Second row: y in image coordinates and for sure in frame
                    Third row : index of the point in the original original data source
<<<<<<< HEAD
                    Forth row : vertical angles on the LIDAR frame
                    Fifth row : horizontal angles on the LIDAR frame  
        color_map:  dictionary
                    maps numeric semantic labels to a BGR color for visualization   
    """
    """
    angles = xy[3,:]*57.295779513
    inside_fov = np.logical_and((angles > -40), (angles < 40))
    inside_frame_indices = np.argwhere(inside_fov).flatten()
    xy = xy[:, inside_frame_indices]
    """
    angles_vert = xy[4,:]*57.295779513
    a = 0
    min_l = -24.9 - a
    max_l = 2 + a
=======
        labels:     (numpy.array)
                    object that gives the semantic label of each point within the scene.  

        color_map:  dictionary
                    maps numeric semantic labels to a BGR color for visualization   
    """       
    xy = xy.astype(int)
    labels = labels[xy[2,:]]
>>>>>>> main

    for i in range(len(xy[0])):
        color = color_map[ int(labels[i]%4) ]
        image = cv2.circle(image, (xy[1,i], xy[0,i]), radius=1, color=color, thickness=-1)
        #image[xy[0,i], xy[1,i],:] = [color[2], color[1],color[0]]

    return np.asarray(image).astype(np.uint8)

def identification_angle_3D(angles, number_of_group):
    """ Gives every angle an ID. 

    Args:
        xy:         2xNumberOfPoints (float) (numpy, list)
                    First row : horizontal angles
                    Second row: vertical angles

        number_of_ID: (int)
                      Number of groups in points 
    """  

    _ , bin_edges = np.histogram(angles[1,:], bins=64)
    IDs = np.zeros(len(angles[0,:]))

    for i in range(number_of_group):

        if i != number_of_group:
            group       = np.logical_and( ( angles[1,:]>=bin_edges[i] ) , ( angles[1,:]<bin_edges[i+1] ) )
        else:
            group       = np.logical_and( ( angles[1,:]>=bin_edges[i] ) , ( angles[1,:]<=bin_edges[i+1] ) )

<<<<<<< HEAD
        #color = color_map[i%4]
        hue = du.line_color(np.array([[i]]))
        c = np.uint8([[[hue,255,255 ]]])   
        cHSV = cv2.cvtColor(c, cv2.COLOR_HSV2RGB)
        color = cHSV[0,0].tolist()

=======
>>>>>>> main
        group_inds  = np.argwhere(group).flatten()
        IDs[group_inds] = i

    return IDs




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
<<<<<<< HEAD
    Returns: 
        filtered_xy: 4xNumberOfPoints (float) (numpy, list)
                    Points that are inside the frame
                    First row : x in image coordinates 
                    Second row: y in image coordinates 
                    Third row : index of the point in the original original data source
                    Forth row : vertical angles on the LIDAR frame
                    Fifth row : horizontal angles on the LIDAR frame  
=======
>>>>>>> main
    """      

    x   = xy[1,:]
    y   = xy[0,:]
    ind = xy[2,:]

    inside_frame_x = np.logical_and((x >= 0), (x < image_size[0]))
    inside_frame_y = np.logical_and((y >= 0), (y < image_size[1]))
    inside_frame_indices = np.argwhere(np.logical_and(inside_frame_x,inside_frame_y)).flatten()

    filtered_xy = np.stack((x[inside_frame_indices],y[inside_frame_indices],ind[inside_frame_indices]))

    return filtered_xy 

def angles_3D(points):
    """ Calculates horizontal and vertical angles (in degree) of the given 3D points.

    Args:
        points:     (num points x 4) numpy.array object.
                    First column : x 
                    Second column: y 
                    Third column : z
                    Forth column : reflection 
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    angles_horizontal = np.arctan2(y,x)*57.295779513
    angles_vertical = np.arctan2(z, np.sqrt(np.square( x )+ np.square( y ) ) )*57.295779513

    angles = np.vstack((angles_horizontal, angles_vertical))
    return angles 


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
<<<<<<< HEAD
    Return:
        xy:         (num points x 4) numpy.array object.
                    First row : x 
                    Second row: y 
                    Third row : indices of original dataset
                    Forth row : vertical angles on the LIDAR frame
                    Fifth row : horizontal angles on the LIDAR frame     

=======
>>>>>>> main
    """   
    front_hemisphere = points[:, 0] > 0 
    front_hemisphere_indices = np.argwhere(front_hemisphere).flatten()      

    front_hemisphere_x = points[front_hemisphere_indices, 0]
    front_hemisphere_y = points[front_hemisphere_indices, 1]
    front_hemisphere_z = points[front_hemisphere_indices, 2]
<<<<<<< HEAD
    angles_horizontal = np.arctan2(front_hemisphere_y,front_hemisphere_x)

    a = np.sqrt(np.square(front_hemisphere_x )+ np.square(front_hemisphere_y))
    angles_vertical = np.arctan2(front_hemisphere_z,a)
=======
>>>>>>> main
    homo_coor = np.ones(len(front_hemisphere_x))
    
    XYZ = np.stack((front_hemisphere_x,front_hemisphere_y,front_hemisphere_z,homo_coor))
        
    projection_matrix = np.matmul(intrinsic,extrinsic) 
    xy = np.matmul(projection_matrix,XYZ)
    xy = xy / xy[2,None]

    xy[2,:] = front_hemisphere_indices
<<<<<<< HEAD
    xy = np.vstack((xy, angles_horizontal)) 
    xy = np.vstack((xy, angles_vertical)) 
=======
>>>>>>> main
    return xy

if __name__ =="__main__":
    print("**** Running task3 ****")

    data = get_data()

    velo_point_cloud    = data['velodyne']
    velo_mat_T          = data['T_cam0_velo']   # extrinsic camera parameters
    cam_mat_P           = data['P_rect_20']     # intrinsic camera parameters
    cam_image           = data['image_2']

    label_color_map = [[255,0,0], [0,255,0], [0,0,255,], [0,255,255]]

    angles      = angles_3D(velo_point_cloud)
    IDs         = identification_angle_3D(angles, 64) 

    xy          = projection_3D_2D(velo_point_cloud,velo_mat_T,cam_mat_P)
    filtered_xy = filter_indices_xy(xy, cam_image.shape)

    new_image   = visualize(cam_image, filtered_xy,IDs, label_color_map)

    im = Image.fromarray(new_image)
    im.show()

<<<<<<< HEAD
=======

>>>>>>> main
   
