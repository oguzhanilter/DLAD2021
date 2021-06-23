import numpy as np
import math

from .task1 import get_iou

def sample_proposals(pred, target, xyz, feat, config, train=False):
    '''
    Task 3
    a. Using the highest IoU, assign each proposal a ground truth annotation. For each assignment also
       return the IoU as this will be required later on.
    b. Sample 64 proposals per scene. If the scene contains at least one foreground and one background
       proposal, of the 64 samples, at most 32 should be foreground proposals. Otherwise, all 64 samples
       can be either foreground or background. If there are less background proposals than 32, existing
       ones can be repeated.
       Furthermore, of the sampled background proposals, 50% should be easy samples and 50% should be
       hard samples when both exist within the scene (again, can be repeated to pad up to equal samples
       each). If only one difficulty class exists, all samples should be of that class.
    input
        pred (N,7) predicted bounding box labels
        target (M,7) ground truth bounding box labels
        xyz (N,512,3) pooled point cloud
        feat (N,512,C) pooled features
        config (dict) data config containing thresholds
        train (string) True if training
    output
        assigned_targets (64,7) target box for each prediction based on highest iou
        xyz (64,512,3) indices 
        feat (64,512,C) indices
        iou (64,) iou of each prediction and its assigned target box
    useful config hyperparameters
        config['t_bg_hard_lb'] threshold background lower bound for hard difficulty
        config['t_bg_up'] threshold background upper bound
        config['t_fg_lb'] threshold foreground lower bound
        config['num_fg_sample'] maximum allowed number of foreground samples
        config['bg_hard_ratio'] background hard difficulty ratio (#hard samples/ #background samples)
    '''
    
    iou_matrix = get_iou(pred, target)

    max_iou_indices = np.argmax(iou_matrix, axis=1)
    max_iou = iou_matrix[:,max_iou_indices]

    if train:
        
        foreground_ind = np.argwhere(max_iou >= config['t_fg_lb'])
        hard_background_ind = np.argwhere( (max_iou >= config['t_bg_hard_lb']) & (max_iou < config['t_bg_up']) )
        easy_background_ind = np.argwhere( (max_iou < config['t_bg_hard_lb']) )

        len_fore = len(foreground_ind)
        len_hard = len(hard_background_ind)
        len_easy = len(easy_background_ind)


        if len_hard + len_easy == 0:

            if len_fore >= 64:
                indices = np.random.choice(foreground_ind, 64, replace=False)
            
            elif len_fore < 64:
                extend = np.random.choice(foreground_ind, 64 - len_fore, replace=True)
                indices = np.append(foreground_ind, extend)
        
        elif len_fore == 0:
            if len_hard == 0:
                if len_easy >= 64:
                    indices = np.random.choice(easy_background_ind, 64, replace=False)
            
                elif len_easy < 64:
                    extend = np.random.choice(easy_background_ind, 64 - len_easy, replace=True)
                    indices = np.append(easy_background_ind, extend)

            if len_easy == 0:
                if len_hard >= 64:
                    indices = np.random.choice(hard_background_ind, 64, replace=False)
            
                elif len_hard < 64:
                    extend = np.random.choice(hard_background_ind, 64 - len_hard, replace=True)
                    indices = np.append(hard_background_ind, extend)

            else:
                if len_hard >= 32:
                    indices1 = np.random.choice(hard_background_ind, 32, replace=False)
            
                elif len_hard < 32:
                    extend = np.random.choice(hard_background_ind, 32 - len_hard, replace=True)
                    indices1 = np.append(hard_background_ind, extend)
                
                if len_easy >= 32:
                    indices2 = np.random.choice(easy_background_ind, 32, replace=False)
            
                elif len_easy < 32:
                    extend = np.random.choice(easy_background_ind, 32 - len_easy, replace=True)
                    indices2 = np.append(easy_background_ind, extend)

                indices = np.append(indices1,indices2)

        else:

            if len_fore >= 32:
                indices1 = np.random.choice(foreground_ind, 32, replace=False)
    
            elif len_fore < 32:
                if len_hard == 0:
                    if len_easy >= 64-len_fore:
                        indices2 = np.random.choice(easy_background_ind, 64-len_fore, replace=False)
                
                    elif len_easy < 64-len_fore:
                        extend = np.random.choice(easy_background_ind, 64-len_fore - len_easy, replace=True)
                        indices2 = np.append(easy_background_ind, extend)

                if len_easy == 0:
                    if len_hard >= 64-len_fore:
                        indices2 = np.random.choice(hard_background_ind, 64-len_fore, replace=False)
                
                    elif len_hard < 64:
                        extend = np.random.choice(hard_background_ind, 64-len_fore - len_hard, replace=True)
                        indices2 = np.append(hard_background_ind, extend)

                else:
                    num_hard = np.ceil((64-len_fore)/2)
                    num_easy = 64 - len_fore - num_hard
                    if len_hard >= num_hard:
                        indices3 = np.random.choice(hard_background_ind, num_hard, replace=False)
                
                    elif len_hard < num_hard:
                        extend = np.random.choice(hard_background_ind, num_hard - len_hard, replace=True)
                        indices3 = np.append(hard_background_ind, extend)
                    
                    if len_easy >= num_easy:
                        indices4 = np.random.choice(easy_background_ind, num_easy, replace=False)
                
                    elif len_easy < num_easy:
                        extend = np.random.choice(easy_background_ind, num_easy - len_easy, replace=True)
                        indices4 = np.append(easy_background_ind, extend)

                    indices2 = np.append(indices3,indices4)

            indices = np.append(indices1,indices2)    

        assigned_targets = target[indices]
        iou = max_iou[indices]
        xyz = xyz[indices]
        feat = feat[indices]


    else: # If training false return everything -> no sampling
        assigned_targets = target[max_iou_indices]
        iou = max_iou

    return assigned_targets, xyz, feat, iou