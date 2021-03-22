import os
import cv2
import data_utils
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import interp1d
import numpy as np
import math
from PIL import Image
from load_data import load_data

from data_utils import *

RELATIVE_PATH_TO_DATA = 'data/problem_4'
LIDAR_PATH = RELATIVE_PATH_TO_DATA + '/velodyne_points'
OXTS_PATH = RELATIVE_PATH_TO_DATA + '/oxts'
OXTS_FILE_PATH = OXTS_PATH + '/data'

IMAGE_PATH = RELATIVE_PATH_TO_DATA + '/image_02'
IMAGE_FILE_PATH = IMAGE_PATH + '/data'

TIMESTAMP = '/timestamps.txt'
TIMESTAMP_START = '/timestamps_start.txt'
TIMESTAMP_END = '/timestamps_end.txt'

script_path = './'

def get_laser_id(xy):
    """ Visualize 2D points on image according to their labels and color map.

    Args:
        xy:         3xNumberOfPoints (float) (numpy, list)
                    First row : x in image coordinates and for sure in frame 
                    Second row: y in image coordinates and for sure in frame
                    Third row : index of the point in the original original data source
                    Forth row : vertical angles on the LIDAR frame
                    Fifth row : horizontal angles on the LIDAR frame  
    Returns
        xy but with additional Sixth row: laser ID
    """
    
    laser_id_idx = 5
    xy_with_id = np.vstack((xy, np.zeros(xy[0].shape)))

   
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
        group_inds  = np.argwhere(group).flatten()

        for j in range(len(group_inds)):
            #image = cv2.circle(image, (xy[1,group_inds[j]], xy[0,group_inds[j]]), radius=1, color=color, thickness=-1)
    
            xy_with_id[5,group_inds[j]] = i
        
    return xy_with_id

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


def corrected_projection_3D_2D(points, extrinsic, intrinsic,lidar_ts,lidar_te,lidar_t,imu_rot,imu_vel):
    """ 
        DISCLAIMER: parameter passing is still not documented!

        Projection of 3D points to 2D image and DOES CORRECTION for motion distortion
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
    
    
    #have to process each point separately and correct for distortions

    for i in range(len(XYZ[0,:])):

        #read horizontal angle (yaw) of point
        h_angle = angles_horizontal[i]
        
        #find interpolation coefficient of current angle between starting and stopping angle of lidar scan (360°)
        interp = (h_angle - lidar_end_yaw)/(lidar_start_yaw - lidar_end_yaw)
        
        #linearly interpolate between timestamps
        dt =   interp*lidar_ts + (1-interp)*lidar_te - lidar_t
        
        R = Rot.from_euler('z',imu_rot[2]*dt, degrees=False).as_matrix() #Rs*coeff + Re*(1-coeff)
        T = imu_vel*dt 

        RT = create_homo_trans(R,T)
        XYZ[:,i] = np.matmul(RT,XYZ[:,i])
        
    xy = np.matmul(projection_matrix,XYZ)
    
    xy = xy / xy[2,None]

    xy[2,:] = front_hemisphere_indices
    xy = np.vstack((xy, angles_horizontal)) 
    xy = np.vstack((xy, angles_vertical)) 
    return xy


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


def visualize_task4(cam_image, xy,velo_point_cloud):
    """ Visualize 2D points on image according to their labels and color map.

    Args:
        image:      Type does not matter (numpy, list ...)
                    3D matrix of an RGB image
        xy:         3xNumberOfPoints (fl£oat) (numpy, list)
                    First row : x in image coordinates and for sure in frame 
                    Second row: y in image coordinates and for sure in frame
                    Third row : index of the point in the original original data source
                    Forth row : vertical angles on the LIDAR frame
                    Fifth row : horizontal angles on the LIDAR frame
                    Sixth row : estimated lased id
                    
        velo_point_cloud: 
                    original velo point cloud
        labels:     (numpy.array)
                    object that gives the semantic label of each point within the scene.  

        color_map:  dictionary
                    maps numeric semantic labels to a BGR color for visualization  
    """       

    x =  velo_point_cloud[xy[2,:].astype(np.uint32)][:,0]
    y =  velo_point_cloud[xy[2,:].astype(np.uint32)][:,1]
    z =  velo_point_cloud[xy[2,:].astype(np.uint32)][:,2]
    depth = x #np.sqrt(np.square(x)+np.square(y)+np.square(z))
    hue = depth_color(depth,0,30)
    
    xy = xy.astype(int)
    for i in range(len(xy[0])):
        
        #original index 
        idx = xy[2,i].astype(np.uint32)
        
        color = hsv2rgb(hue[i],1,1)
        #color = hsv2rgb(velo_point_cloud[xy[2,i].astype(np.uint32),0],1,1)
        
        cam_image = cv2.circle(cam_image, (xy[1,i], xy[0,i]), radius=2, color=color, thickness=-1)
        #image[xy[0,i], xy[1,i],:] = [color[2], color[1],color[0]]
        """  
        if idx%10 == 0:
            #im = Image.fromarray(new_image)
            #im.show()
            
            im = np.asarray(cam_image).astype(np.uint8)
    
            cv2.destroyAllWindows()
            cv2.imshow('asd', im)
            k = cv2.waitKey(0)
            
            print("laser id: ",xy[5,i])
            
            if k==113: #q key
                cv2.destroyAllWindows()
                break
    #cv2.destroyAllWindows()
    """

    return np.asarray(cam_image).astype(np.uint8)

def create_homo_trans(R,T):
    """returns homo tranformation"""
    RT = np.zeros((4, 4))
    RT[:3,:3] = R
    if len(T.shape)>1:
        RT[:3,3:4] = T
    else:
        RT[:3,3] = T
    RT[3,3] = 1
    
    return RT
    

def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b


def get_frame_name(frame):
    """Returns frame name as string"""
    return str(frame).zfill(10)


def get_lidar_orientations(lidar_ts,lidar_te,lidar_t):
    """Calculate start and end orietation (yaw) of lidar
    
    Args:
        lidar_ts (float): starting time of lidar scan
        lidar_te (float): ending time of lidar scan
        lidar_t (float): time at which lidar is scanning in positive x direction (forward)

    Return
        lidar_start_yaw (float): starting orientation of lidar 
        lidar_end_yaw (float): ending orientation of lidar 
    """
    
    #at lidar_t the lidar scanning  direction is forward
    #yaw angle is in counter-clockwise direction (angle 0 is in postive x direction)
    #angle on z axis of lidar scanning direction
    #remember lidar scans in counter clockwise direction
    lidar_period = lidar_te-lidar_ts #period that lidar used to complete full rotation
    lidar_rot = math.pi*2/lidar_period
    lidar_start_yaw = lidar_rot*(lidar_t-lidar_ts)
    lidar_end_yaw = lidar_start_yaw - math.pi*2 #giust start + 360° in clockwise direction

    return lidar_start_yaw, lidar_end_yaw



if __name__ =="__main__":
    print("**** Running task4 ****")


    REMOVE_DISTORTION = True

    # LOAD TRANFORMATION MATRICES
    R_velo2cam,T_velo2cam = calib_velo2cam(RELATIVE_PATH_TO_DATA + '/calib_velo_to_cam.txt')
    R_imu2velo,T_imu2velo = calib_velo2cam(RELATIVE_PATH_TO_DATA + '/calib_imu_to_velo.txt')
    P_cam2cam = calib_cam2cam(RELATIVE_PATH_TO_DATA + '/calib_cam_to_cam.txt','02')


    # LOAD DATA FOR specific frame

    #interesting debugging frame = 2
    frame = 37
    frame_interp = [-1,0,1] #list of frame offsets used for imu data interpolation -1 = previus, 0 = current, 1=next etc

    frame_name = get_frame_name(frame)


    #READ TIMESTAMPS

    #ts are start times, te are end times
    lidar_ts = compute_timestamps(LIDAR_PATH + TIMESTAMP_START, frame)
    lidar_te = compute_timestamps(LIDAR_PATH + TIMESTAMP_END, frame)
    lidar_t = compute_timestamps(LIDAR_PATH + TIMESTAMP, frame) #lidar looking forward

    imu_t = compute_timestamps(OXTS_PATH + TIMESTAMP, frame)
    imu_t_interp = [compute_timestamps(OXTS_PATH + TIMESTAMP, frame+i) for i in frame_interp] #list of prev, current and next timestamps
    camera_t = compute_timestamps(IMAGE_PATH + TIMESTAMP, frame)


    #READ IMAGE
    image = cv2.imread(IMAGE_FILE_PATH + '/'+frame_name+ '.png')

    #READ IMU DATA

    #data points for imu vel and angular vel to do interpolation (for previous, current and next frame)
    imu_vel_interp = [load_oxts_velocity(OXTS_FILE_PATH+'/'+get_frame_name(frame+i)+'.txt') for i in frame_interp]
    imu_rot_interp = [load_oxts_angular_rate(OXTS_FILE_PATH+'/'+get_frame_name(frame+i)+'.txt') for i in frame_interp]
    # imu in velo frame
    imu_vel_interp = [np.matmul(vel,R_imu2velo) for vel in imu_vel_interp]
    imu_vel_interp = [vel + T_imu2velo[0,:] for vel in imu_vel_interp]
    imu_rot_interp = [np.matmul(rot,R_imu2velo) for rot in imu_rot_interp]
    #create an interpolation function for imu vel and rot
    imu_vel_f = interp1d(imu_t_interp,imu_vel_interp,axis=0)
    imu_rot_f = interp1d(imu_t_interp,imu_rot_interp,axis=0)
    #imu_vel_f(t) = imu data interpolated at timestamp t

    #imu data at timestep lidar is looking forward (use this as "average" imu data during correction)
    imu_vel = imu_vel_f(lidar_t)
    imu_rot = imu_rot_f(lidar_t)


    #READ POINT CLOUD
    velo_point_cloud = load_from_bin(script_path+LIDAR_PATH+'/data/' +frame_name+'.bin')


    #get starting and ending lidar orientation 
    lidar_start_yaw, lidar_end_yaw = get_lidar_orientations(lidar_ts,lidar_te,lidar_t)


    #create tranformation velo to cam in homogeneous coorindates
    RT_velo2cam = create_homo_trans(R_velo2cam,T_velo2cam)


    #PROJECT lidar point cloud data into image plane

    #switch between corrected and original
    if REMOVE_DISTORTION:
        xy          = corrected_projection_3D_2D(velo_point_cloud,RT_velo2cam,P_cam2cam,
                                                lidar_ts,lidar_te,lidar_t,imu_rot,imu_vel)
    else:
        xy          = projection_3D_2D(velo_point_cloud,RT_velo2cam,P_cam2cam)
    
    
    filtered_xy  = filter_indices_xy(xy, image.shape)
    xy_with_id = get_laser_id(filtered_xy)

    new_image = image.copy()
    new_image = visualize_task4(new_image,xy_with_id,velo_point_cloud)
    im = Image.fromarray(new_image)
    im.show()
