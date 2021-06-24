import numpy as np

from utils.task1 import get_iou

def nms(pred, score, threshold):
    '''
    Task 5
    Implement NMS to reduce the number of predictions per frame with a threshold
    of 0.1. The IoU should be calculated only on the BEV.
    input
        pred (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
        score (N,) confidence scores
        threshold (float) upper bound threshold for NMS
    output
        s_f (M,7) 3D bounding boxes after NMS
        c_f (M,1) corresopnding confidence scores
    '''

    s_f = np.empty((0,7))
    c_f = np.empty((0,1))
    
    while pred.shape[0] > 0:
        max_ind = np.argmax(score)

        c_f = np.append(c_f, np.array([score[max_ind]]).reshape((1,1)),axis=0)
        score = np.delete(score, max_ind,0)

        print(pred.shape)

        s_f = np.append(s_f, np.array([pred[max_ind]]).reshape((1,7)), axis=0)

        pred = np.delete(pred, max_ind,0) 


        pred_altered = np.copy(pred)

        pred_altered[:,1] = 0
        pred_altered[:,3] = 1

        last_sf = np.copy(s_f[-1]).reshape(1,7)
        
        last_sf[:,1] = 0
        last_sf[:,3] = 1

        iou = get_iou(pred_altered, last_sf)

        indices = np.argwhere(iou>=threshold)[:,0]

        pred = np.delete(pred, indices,0)
        score = np.delete(score, indices,0)


    return s_f, c_f
