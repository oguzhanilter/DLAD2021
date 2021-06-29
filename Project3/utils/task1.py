import numpy as np
from shapely.geometry import MultiPoint
import time

# Adapted from 
# https://github.com/AlienCat-K/3D-IoU-Python/blob/c07df684a31171fa4cbcb8ff0d50caddc9e99a13/3D-IoU-Python.py#L60


def rot_matrix(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])


def label2corners(label):
    '''
    Task 1
    input
        label (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
    output
        corners (N,8,3) corner coordinates in the rectified reference frame
    '''
    N = label.shape[0]
    corners = np.empty((N,8,3))

    i = 0
    for bb in label:
        R = rot_matrix(bb[6])
        h,w,l = bb[3:6]
        
        x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
        y_corners = [0,0,0,0,-h,-h,-h,-h]
        z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]

        corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
        corners_3d[0,:] = corners_3d[0,:] + bb[0]
        corners_3d[1,:] = corners_3d[1,:] + bb[1]
        corners_3d[2,:] = corners_3d[2,:] + bb[2]
        corners_3d = np.transpose(corners_3d)
        corners[i,:,:] = corners_3d
        i += 1

    return corners

def poly_area(x,y):
    # Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates 
    area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    return area

def polygon_clip(subjectPolygon, clipPolygon):

    # Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
   """ Clip a polygon with another polygon
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter_area = MultiPoint(inter_p).convex_hull.area
        return inter_p, hull_inter_area
    else:
        return None, 0.0  

def box3d_vol(corners):
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c


def get_iou(pred, target):
    '''
    Task 1
    input
        pred (N,7) 3D bounding box corners
        target (M,7) 3D bounding box corners
    output
        iou (N,M) pairwise 3D intersection-over-union
    '''

    N = pred.shape[0]
    M = target.shape[0]


    pred_corners = label2corners(pred)
    target_corners = label2corners(target)

    iou = np.empty([N,M])
    
    for p in range(N):
        for t in range(M):
            corners_pred = pred_corners[p]
            corners_target = target_corners[t]

            rect1 = [(corners_pred[i,0], corners_pred[i,2]) for i in range(3,-1,-1)]
            rect2 = [(corners_target[i,0], corners_target[i,2]) for i in range(3,-1,-1)] 

            # area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
            # area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])

            inter, inter_area = convex_hull_intersection(rect1, rect2)

            # iou_2d = inter_area/(area1+area2-inter_area)
            ymax = min(corners_pred[0,1], corners_target[0,1])
            ymin = max(corners_pred[4,1], corners_target[4,1])

            inter_vol = inter_area * max(0.0, ymax-ymin)
            
            vol1 = box3d_vol(corners_pred)
            vol2 = box3d_vol(corners_target)
            iou[p,t] = inter_vol / (vol1 + vol2 - inter_vol)

    return iou

def compute_recall(pred, target, threshold):
    '''
    Task 1
    input
        pred (N,7) proposed 3D bounding box labels
        target (M,7) ground truth 3D bounding box labels
        threshold (float) threshold for positive samples
    output
        recall (float) recall for the scene
    '''

    M = target.shape[0]
    #print(pred.shape)
    #print(target.shape)
    iou = get_iou(pred, target)
    # print(iou)
    Exceed_threshold = iou >= threshold
    # TP = len(np.argwhere(Exceed_threshold))
    TP = np.count_nonzero(np.sum(Exceed_threshold, axis=0))
    FN = M -  np.count_nonzero(np.sum(Exceed_threshold, axis=0))

    return TP / (TP + FN)