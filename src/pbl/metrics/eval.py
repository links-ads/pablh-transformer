import torch
import torchmetrics

def acc_perclass(pred, target, num_classes=2):
    if pred.shape[1] == 1:
        num_classes = 2
    res_list = []
    acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, ignore_index=255, average='none')
    acc = acc.to(pred.device)
    for i, p in enumerate(pred):
        if pred.shape[1] == 1:
            p = pred[i][0]>0
        else:
            p = p.argmax(dim=0)
        t = target[i][0]
        res_list.append(acc(p, t))
    # average each class
    acc = torch.stack(res_list).mean(dim=0)
    return acc

def iou_perclass(pred, target, num_classes=2):
    if pred.shape[1] == 1:
        num_classes = 2
    res_list = []
    # iou from torchmetrics as jaccard index
    iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes, ignore_index=255, average='none')
    iou = iou.to(pred.device)
    for i, p in enumerate(pred):
        if pred.shape[1] == 1:
            p = pred[i][0]>0
        else:
            p = p.argmax(dim=0)
        t = target[i][0]
        res_list.append(iou(p, t))
    # average each class
    iou = torch.stack(res_list).mean(dim=0)
    return iou