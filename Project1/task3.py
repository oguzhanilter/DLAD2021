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


def visualize(image, xy, color_map):
    """ Visualize 2D points on image according to their labels and color map.

    Args:
        image:      Type does not matter (numpy, list ...)
                    3D matrix of an RGB image
        xy:         3xNumberOfPoints (float) (numpy, list)
                    First row : x in image coordinates and for sure in frame 
                    Second row: y in image coordinates and for sure in frame
                    Third row : index of the point in the original original data source
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


    d = np.logical_and( ( angles_vert>=min_l ) , ( angles_vert<=max_l ) )
    d_i  = np.argwhere(d).flatten()
    xy = xy[:,d_i]
    angles_vert = angles_vert[d_i]
 

    _ , bin_edges = np.histogram(angles_vert, bins=64) # ,range=(min_l,max_l)

    xy = np.around(xy[0:3,:])
    xy = xy.astype(int)
    x       = xy[0,:]
    y       = xy[1,:]

    

    for i in range(64):
        if i != 64:
            group       = np.logical_and( ( angles_vert>=bin_edges[i] ) , ( angles_vert<bin_edges[i+1] ) )
        else:
            group       = np.logical_and( ( angles_vert>=bin_edges[i] ) , ( angles_vert<=bin_edges[i+1] ) )

        #color = color_map[i%4]
        hue = du.line_color(np.array([[i]]))
        c = np.uint8([[[hue,255,255 ]]])   
        cHSV = cv2.cvtColor(c, cv2.COLOR_HSV2RGB)
        color = cHSV[0,0].tolist()

        group_inds  = np.argwhere(group).flatten()

        for j in range(len(group_inds)):
            image = cv2.circle(image, (xy[1,group_inds[j]], xy[0,group_inds[j]]), radius=1, color=color, thickness=-1)
        
        #im = Image.fromarray(image)
        #im.show()

        # For debugginh to see every different angle bin 
        """

        print(i)
        im = np.asarray(image).astype(np.uint8)
        
        cv2.imshow('asd', im)
        cv2.waitKey(0)  
  
        #xy[2,group_inds] = i % 4
        """
        
    """
    for i in range(len(xy[0])):
        
        color = color_map[int(xy[2, i])]
        image = cv2.circle(image, (xy[1,i], xy[0,i]), radius=1, color=color, thickness=-1)
        #image[xy[0,i], xy[1,i],:] = [color[2], color[1],color[0]]
    """
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
                    Forth row : vertical angles on the LIDAR frame
                    Fifth row : horizontal angles on the LIDAR frame  
    """      

    x = xy[1,:]
    y = xy[0,:]
    indices = xy[2,:]
    angles = xy[3,:]
    angles_vert = xy[4,:]

    #inside_frame_x = np.logical_and((x > 0), (x < image_size[0]))
    #inside_frame_y = np.logical_and((y > 0), (y < image_size[1]))
    #inside_frame_indices = np.argwhere(np.logical_and(inside_frame_x,inside_frame_y)).flatten()

    # filtered_xy = np.stack((x[inside_frame_indices],y[inside_frame_indices],indices[inside_frame_indices],angles[inside_frame_indices],angles_vert[inside_frame_indices] ))
    filtered_xy = np.stack((x,y,indices,angles,angles_vert ))
    
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
                    Forth row : vertical angles on the LIDAR frame
                    Fifth row : horizontal angles on the LIDAR frame     

    """   
    front_hemisphere = points[:, 0] > 0 
    front_hemisphere_indices = np.argwhere(front_hemisphere).flatten()

    front_hemisphere_x = points[front_hemisphere_indices, 0]
    front_hemisphere_y = points[front_hemisphere_indices, 1]
    front_hemisphere_z = points[front_hemisphere_indices, 2]
    angles_horizontal = np.arctan2(front_hemisphere_y,front_hemisphere_x)

    a = np.sqrt(np.square(front_hemisphere_x )+ np.square(front_hemisphere_y))
    angles_vertical = np.arctan2(front_hemisphere_z,a)
    homo_coor = np.ones(len(front_hemisphere_x))

    XYZ = np.stack((front_hemisphere_x,front_hemisphere_y,front_hemisphere_z,homo_coor))
        
    projection_matrix = np.matmul(intrinsic,extrinsic) 
        
    xy = np.matmul(projection_matrix,XYZ)
    xy = xy / xy[2,None]

    xy[2,:] = front_hemisphere_indices
    xy = np.vstack((xy, angles_horizontal)) 
    xy = np.vstack((xy, angles_vertical)) 
    return xy

if __name__ =="__main__":
    print("**** Running task3 ****")

    data = get_data()

    velo_point_cloud    = data['velodyne']
    velo_mat_T          = data['T_cam2_velo']   # extrinsic camera parameters
    cam_mat_P           = data['P_rect_20']     # intrinsic camera parameters
    cam_image           = data['image_2']

    label_color_map = [[255,0,0], [0,255,0], [0,0,255,], [0,255,255]]

    xy          = projection_3D_2D(velo_point_cloud,velo_mat_T,cam_mat_P)
    filtered_xy = filter_indices_xy(xy, cam_image.shape)
    new_image   = visualize(cam_image, filtered_xy, label_color_map)

    im = Image.fromarray(new_image)
    im.show()

   
