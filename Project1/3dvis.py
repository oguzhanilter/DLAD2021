# Deep Learning for Autonomous Driving
# Material for Problem 2 of Project 1
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
import os
from load_data import load_data

class Visualizer():
    def __init__(self):
        self.canvas = SceneCanvas(keys='interactive', show=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.view = vispy.scene.widgets.ViewBox(border_color='white',
                        parent=self.canvas.scene)
        self.grid.add_widget(self.view, 0, 0)

        # Point Cloud Visualizer
        self.sem_vis = visuals.Markers()
        self.view.camera = vispy.scene.cameras.TurntableCamera(up='z', azimuth=90)
        self.view.add(self.sem_vis)
        visuals.XYZAxis(parent=self.view.scene)
        
        # Object Detection Visualizer
        self.obj_vis = visuals.Line()
        self.view.add(self.obj_vis)
        self.connect = np.asarray([[0,1],[0,2],[0,4],
                                   [3,1],[3,2],[3,7],
                                   [5,1],[5,4],[5,7],
                                   [6,2],[6,4],[6,7]])

        #self.connect = np.asarray([[0,1],[0,3],[0,4],
        #                           [2,1],[2,3],[2,6],
        #                           [5,1],[5,4],[5,6],
        #                           [7,3],[7,4],[7,6]])

    def update(self, points, colors):
        '''
        :param points: point cloud data
                        shape (N, 3)          
        Task 2: Change this function such that each point
        is colored depending on its semantic label
        '''
        colors = [[label_color_map[label][2],label_color_map[label][1],label_color_map[label][0]]
         for label in point_labels[:,0]]
        
        colors_norm = np.array(colors)/256

        self.sem_vis.set_data(points,face_color=colors_norm, size=3)
    
    def update_boxes(self, corners):
        '''
        :param corners: corners of the bounding boxes
                        shape (N, 8, 3) for N boxes
        (8, 3) array of vertices for the 3D box in
        following order:
            1 -------- 0
           /|         /|
          3 -------- 2 .
          | |        | |
          . 5 -------- 4
          |/         |/
          7 -------- 6
        If you plan to use a different order, you can
        change self.connect accordinly.
        '''
        for i in range(corners.shape[0]):
            connect = np.concatenate((connect, self.connect+8*i), axis=0) \
                      if i>0 else self.connect
        self.obj_vis.set_data(corners.reshape(-1,3),
                              connect=connect,
                              width=2,
                              color=[0,1,0,1])


def object_box_points(objects,velo_cam0_mat_T):

    number_of_objects = len(objects)
    #box_edges = np.ones((4, 8*number_of_objects))

    box_edges = []

    #obj_edges = np.ones((8, 4))
    #obj_edges = np.ones((8, 4))


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

        #Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
        Ry = np.array([[np.cos(ty), 0, np.sin(ty),0], [0, 1, 0,0], [-np.sin(ty), 0, np.cos(ty),0],[0,0,0,1]])
        Ry = np.transpose(Ry)

        obj_edges = np.ones((8, 4))

        obj_edges[x1,0] = - object_dim[2]/2 # x
        obj_edges[x2,0] = object_dim[2]/2 # x

        obj_edges[y1,1] = 0
        obj_edges[y2,1] = - object_dim[0]

        obj_edges[z1,2] = object_dim[1]/2
        obj_edges[z2,2] = - object_dim[1]/2

        obj_edges = np.dot(obj_edges,Ry)
        

        obj_edges[x1,0] += object_center[0] 
        obj_edges[x2,0] += object_center[0] 

        obj_edges[y1,1] += object_center[1] 
        obj_edges[y2,1] += object_center[1]

        obj_edges[z1,2] += object_center[2] 
        obj_edges[z2,2] += object_center[2] 
        
        
        trans = velo_cam0_mat_T[:3,3]
        rot = velo_cam0_mat_T[:3,:3]
        rot_inv = np.linalg.inv(rot)
 
        cam0_velo_mat_T = np.zeros((4, 4))
        cam0_velo_mat_T[:3,:3] = rot_inv
        cam0_velo_mat_T[:3,3] = -np.matmul(rot_inv,trans)
        cam0_velo_mat_T[3,3] = 1
 
        obj_edges = np.matmul(obj_edges,np.transpose(cam0_velo_mat_T))
        #obj_edges = obj_edges/obj_edges[None,3]
        obj_edges = obj_edges/np.transpose(np.stack([obj_edges[:,3]]*4))
        
        box_edges.append(obj_edges[:,:3])
        
    box_edges = np.array(box_edges)
        
    return box_edges

if __name__ == '__main__':
    data = load_data('data/demo.p') # Change to data.p for your final submission 
    
    velo_point_cloud    = data['velodyne']
    cam_mat_K           = data['K_cam2']  
    velo_cam0_mat_T     = data['T_cam0_velo']
    velo_mat_T          = data['T_cam2_velo']   # extrinsic camera parameters
    cam_mat_P           = data['P_rect_20']     # intrinsic camera parameters
    cam_image           = data['image_2']
    labels              = data['labels']
    label_color_map     = data['color_map']
    point_labels        = data['sem_label']
    objects             = data['objects']
    

    visualizer = Visualizer()
    visualizer.update(data['velodyne'][:,:3],label_color_map)
    
    '''
    Task 2: Compute all bounding box corners from given
    annotations. You can visualize the bounding boxes using
    visualizer.update_boxes(corners)
    '''


    object_box_points_3D = object_box_points(objects,velo_cam0_mat_T)

    visualizer.update_boxes(object_box_points_3D)
    
    vispy.app.run()




