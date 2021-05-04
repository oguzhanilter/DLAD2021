import os
import cv2
import data_utils
import numpy as np
from PIL import Image
from load_data import load_data
import data_utils as du


RELATIVE_PATH_TO_DATA = 'data'
DATA_FILE_NAME = 'data.p'

ID_MODE = 0 # 0 for histogram , 1 for prob, 2 for rotation, 3 for rot and prob

NUMBER_OF_BIN = 64
ROT_ANGLE = -1.5
IMAGE_NAME = "task3_result.png"



def get_data():
    script_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(script_path, RELATIVE_PATH_TO_DATA,DATA_FILE_NAME)
    data = load_data(data_path)
    return data

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
    labels = labels[xy[2,:]]

    for i in range(len(xy[0])):
        color = color_map[ int(labels[i]%4) ]
        image = cv2.circle(image, (xy[1,i], xy[0,i]), radius=1, color=color, thickness=-1)
        #image[xy[0,i], xy[1,i],:] = [color[2], color[1],color[0]]

    return np.asarray(image).astype(np.uint8)

def identification_angle_3D(angles, number_of_group):
    """ Gives every angle an ID using histogram 

    Args:
        angles:     2xNumberOfPoints (float) (numpy, list)
                    First row : horizontal angles
                    Second row: vertical angles

        number_of_ID: (int)
                      Number of groups in points 
    """  

    _ , bin_edges = np.histogram(angles[1,:], bins=NUMBER_OF_BIN)
    IDs = np.zeros(len(angles[0,:]))

    for i in range(number_of_group):

        if i != number_of_group:
            group       = np.logical_and( ( angles[1,:]>=bin_edges[i] ) , ( angles[1,:]<bin_edges[i+1] ) )
        else:
            group       = np.logical_and( ( angles[1,:]>=bin_edges[i] ) , ( angles[1,:]<=bin_edges[i+1] ) )

        group_inds  = np.argwhere(group).flatten()
        IDs[group_inds] = i

    return IDs


def normal_dist(x , mean , sdt):
    prob_density = (np.sqrt(2*np.pi)*sdt) * np.exp(-0.5*((x-mean)/sdt)**2)
    return prob_density

def identification_prob_3D(angles, mean, std):
    """ Gives every angle an ID using the normal distribution 

    Args:
        angles:     2xNumberOfPoints (float) (numpy, list)
                    First row : horizontal angles
                    Second row: vertical angles

        number_of_ID: (int)
                      Number of groups in points 
    """  

    IDs = np.zeros(len(angles[0,:]))

    for i in range(len(angles[0,:])):
        prob = [normal_dist(angles[1,i],m,std ) for m in mean]
        IDs[i] = np.argmax(prob)

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

    angles_horizontal = np.degrees(np.arctan2(y,x))
    angles_vertical = np.degrees(np.arctan2(z, np.sqrt(np.square( x )+ np.square( y ) ) ))

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

if __name__ =="__main__":
    print("**** Running task3 ****")

    data = get_data()

    velo_point_cloud    = data['velodyne']
    velo_mat_T          = data['T_cam0_velo']   # extrinsic camera parameters
    cam_mat_P           = data['P_rect_20']     # intrinsic camera parameters
    cam_image           = data['image_2']

    label_color_map = [[255,0,0], [0,255,0], [0,0,255,], [0,255,255]]

    if(ID_MODE == 1):
        angles      = angles_3D(velo_point_cloud)
        mean        = np.linspace(min(angles[1,:]), max(angles[1,:]), NUMBER_OF_BIN)
        std         = np.abs(mean[0]-mean[1])/5
        IDs         = identification_prob_3D(angles,mean,std )

    elif(ID_MODE == 2):
        tx = np.radians(ROT_ANGLE)
        Rx = np.array([[1, 0, 0],[0, np.cos(tx), -np.sin(tx)] , [0, np.sin(tx), np.cos(tx)]])
        velo_point_cloud = np.matmul(velo_point_cloud[:,0:3],Rx)
        angles      = angles_3D(velo_point_cloud)
        IDs         = identification_angle_3D(angles, 64)

    elif(ID_MODE == 3):
        tx = np.radians(ROT_ANGLE)
        Rx = np.array([[1, 0, 0],[0, np.cos(tx), -np.sin(tx)] , [0, np.sin(tx), np.cos(tx)]])
        velo_point_cloud = np.matmul(velo_point_cloud[:,0:3],Rx)
        angles      = angles_3D(velo_point_cloud)

        mean        = np.linspace(min(angles[1,:]), max(angles[1,:]), 66)
        std         = np.abs(mean[0]-mean[1])/4
        IDs         = identification_prob_3D(angles,mean,std )

    else:
        angles      = angles_3D(velo_point_cloud)
        IDs         = identification_angle_3D(angles, 64)

    xy          = projection_3D_2D(velo_point_cloud,velo_mat_T,cam_mat_P)
    filtered_xy = filter_indices_xy(xy, cam_image.shape)

    new_image   = visualize(cam_image, filtered_xy,IDs, label_color_map)

    im = Image.fromarray(new_image)
    im.show()
    im.save(IMAGE_NAME)

<<<<<<< HEAD
=======

<<<<<<< HEAD
>>>>>>> main
   
=======
   
>>>>>>> main
