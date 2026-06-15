import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, ignore_index=255, eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, outputs, targets):
        # outputs = outputs.contiguous()
        # targets = targets.contiguous()
        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, self.eps, 1. - self.eps)
        targets = torch.clamp(targets, self.eps, 1. - self.eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt) ** self.gamma * torch.log(pt)).mean()



class MulticlassBCELoss():
    def __init__(self,
                 class_weights: list
        ):
        self.class_weights = torch.tensor(class_weights)

    def __call__(self, pred, target):
        target = target.squeeze(1).long()
        if len(target.shape) == 2:
            target = target.unsqueeze(dim=0)
        # loss = F.cross_entropy(pred, target, ignore_index=255, weight=self.class_weights, reduction="mean")
        # do not ignore index
        loss = F.cross_entropy(pred, target, weight=self.class_weights, reduction="mean")
        return loss
    
    def to(self, device):
        self.class_weights = self.class_weights.to(device)
        return self
    
class SingleclassBCELoss():
    def __init__(self, pos_weight=5.):
        self.pos_weight = torch.tensor(pos_weight)
        
    def __call__(self, pred, target):
        target = target.float()
        try:
            loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=self.pos_weight)
        except RuntimeError:
            self.pos_weight = self.pos_weight.to(pred.device)
            loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=self.pos_weight)
        return loss

    def to(self, device):
        self.pos_weight = self.pos_weight.to(device)
        return self
    
class SingleclassBCELossSmooth():
    def __init__(self, smooth=0.1):
        self.smooth = smooth
        
    def __call__(self, pred, target):
        target = target.float()
        # Apply label smoothing
        target = target * (1 - self.smooth) + 0.5 * self.smooth
        loss = F.binary_cross_entropy_with_logits(pred, target)
        return loss
    
    def to(self, device):
        return self
            

class OVRBinaryCrossEntropyLossWithBackground():
    def __init__(self,
                 class_weights: list
        ):
        self.class_weights = torch.tensor(class_weights)
    
    def __call__(self, pred, target, mask):
        # binary cross entropy loss between each class
        # if mask.sum() == 0: 
        #     return None
        # set target to 2 if target is 255
        target = torch.where(target == 255, 2, target)
        target = F.one_hot(target.squeeze(1).long(), num_classes=3)
        # reshape from (batch_size, height, width, 3) to (batch_size, height, width, 2) by dropping bg
        target = target[:, :, :, :2]
        if target.sum() == 0:
            # return loss of 0 if there are no valid values
            return None
        # reshape from (batch_size, height, width, 2) to (batch_size, 2, height, width)
        target = target.permute(0, 3, 1, 2)
        # # make mask from b,1,h,w to b,2,h,w
        # mask = mask.expand(-1, 2, -1, -1)
        # # get the valid values from target and pred
        # target = target[mask == 1]
        # pred = pred[mask == 1]
        # convert target to float
        target = target.float()
        # flatten using reshape
        target = target.reshape(-1)
        pred = pred.view(-1)
        # compute the loss
        loss = F.binary_cross_entropy_with_logits(pred, target)
        return loss
    
    def to(self, device):
        self.class_weights = self.class_weights.to(device)
        return self
    
    
class OVRBinaryCrossEntropyLossWithoutBackground():
    def __init__(self,
                 class_weights: list
        ):
        self.class_weights = torch.tensor(class_weights)
    
    def __call__(self, pred, target, mask):
        # binary cross entropy loss between each class
        if mask.sum() == 0: 
            return None
        # set target to 2 if target is 255
        target = torch.where(target == 255, 0, target)
        target = F.one_hot(target.squeeze(1).long(), num_classes=2)
        # reshape from (batch_size, height, width, 2) to (batch_size, 2, height, width)
        target = target.permute(0, 3, 1, 2)
        # make mask from b,1,h,w to b,2,h,w
        mask = mask.expand(-1, 2, -1, -1)
        # get the valid values from target and pred
        target = target[mask == 1]
        pred = pred[mask == 1]
        # convert target to float
        target = target.float()
        # compute the loss
        loss = F.binary_cross_entropy_with_logits(pred, target)
        return loss
    
    def to(self, device):
        self.class_weights = self.class_weights.to(device)
        return self
