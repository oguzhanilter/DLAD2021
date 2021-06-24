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
        # scale = np.array([1,1,1,3,3,3,1])
        indices = np.argwhere(iou >= self.config['positive_reg_lb'])
        indices = indices.reshape(indices.shape[1])

        loss_trans  = self.loss(pred[indices][0:3], target[indices][0:3])
        loss_size   = self.loss(pred[indices][3:6], target[indices][3:6])
        loss_rot    = self.loss(pred[indices][6], target[indices][6])

        return loss_trans + 3*loss_size + loss_rot


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
        s = nn.Sigmoid()
        
        pos_indices = np.argwhere(iou >= self.config['positive_cls_lb'])
        neg_indices = np.argwhere(iou <= self.config['negative_cls_ub'])

        pos_indices = pos_indices.reshape(pos_indices.shape[1])
        neg_indices = neg_indices.reshape(neg_indices.shape[1])

        ones  = np.ones (len(pos_indices))
        zeros = np.zeros(len(neg_indices))

        predictions = torch.tensor(np.append(pred[pos_indices],pred[neg_indices]).astype(float))
        labels      = torch.tensor(np.append(ones,zeros).astype(float))

        loss = self.loss(predictions,labels)

        return loss 


        