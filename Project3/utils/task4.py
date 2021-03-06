import torch
import torch.nn as nn

import numpy as np

class RegressionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.SmoothL1Loss()

    def forward(self, pred, target, iou):
        '''
        Task 4.a
        We do not want to define the regression loss over the entire input space.
        While negative samples are necessary for the classification network, we
        only want to train our regression head using positive samples. Use 3D
        IoU ≥ 0.55 to determine positive samples and alter the RegressionLoss
        module such that only positive samples contribute to the loss.
        input
            pred (N,7) predicted bounding boxes (x,y,z,h,w,l,ry)
            target (N,7) target bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_reg_lb'] lower bound for positive samples
        '''

        indices = iou >= self.config['positive_reg_lb']

    
        # Translation
        loss_trans  = self.loss(pred[indices, 0:3], target[indices, 0:3])
        # Size
        loss_size   = self.loss(pred[indices, 3:6], target[indices, 3:6])
        # Rotation 
        loss_rot    = self.loss(pred[indices, 6], target[indices, 6])
        
        if indices.sum() > 0:

            return loss_trans + 3*loss_size + loss_rot
        
        else:
            return 0*(loss_trans + 3*loss_size + loss_rot)


class ClassificationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.BCELoss()

    def forward(self, pred, iou):
        '''
        Task 4.b
        Extract the target scores depending on the IoU. For the training
        of the classification head we want to be more strict as we want to
        avoid incorrect training signals to supervise our network.  A proposal
        is considered as positive (class 1) if its maximum IoU with ground
        truth boxes is ≥ 0.6, and negative (class 0) if its maximum IoU ≤ 0.45.
            pred (N,7) predicted bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_cls_lb'] lower bound for positive samples
            self.config['negative_cls_ub'] upper bound for negative samples
        '''
        
        pos_indices = iou >= self.config['positive_cls_lb']
        neg_indices = iou <= self.config['negative_cls_ub']

        len_pos_indices = pos_indices.sum()
        len_neg_indices = neg_indices.sum()

        predictions = np.zeros((len_pos_indices+len_neg_indices))
        labels = np.zeros(len_pos_indices+len_neg_indices)

        predictions[:len_pos_indices] = pred[pos_indices].cpu().data.reshape(len_pos_indices)
        predictions[len_pos_indices:] = pred[neg_indices].cpu().data.reshape(len_neg_indices)

        labels[:len_pos_indices] = 1

        predictions = torch.tensor(predictions)
        labels = torch.tensor(labels)

        #ones  = np.ones (pos_indices.sum())
        #zeros = np.zeros(neg_indices.sum())

        #predictions = torch.tensor(np.append(pred[pos_indices],pred[neg_indices]).astype(float))
        #labels      = torch.tensor(np.append(ones,zeros))

        loss = self.loss(predictions,labels)

        return loss 


    