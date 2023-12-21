import numpy as np
import torch

def get_IoU(pt_mask, gt_mask):
    # pred_mask  [N, Image_W, Image_H]
    # gt_mask   [N, Image_W, Image_H]
    SMOOTH = 1e-10
    intersection = (pt_mask & gt_mask).sum((-1, -2)).to(torch.float32) # [N, 1]
    union = (pt_mask | gt_mask).sum((-1, -2)).to(torch.float32) # [N, 1]

    iou = (intersection + SMOOTH) / (union + SMOOTH) # [N, 1]

    return iou

def evaluation_image(frame_pred, frame_label, counts, meta, save_dict=None):
    frame_pred = (frame_pred > 0.5).to(torch.int64)
    frame_label = frame_label.to(torch.int64)
    counts = counts.to(torch.int64)
    vm_no_crop_gt = meta["vm_no_crop_gt"].squeeze().unsqueeze(0).to(torch.int64)
    frame_pred = frame_pred.unsqueeze(0)
    frame_label = frame_label.unsqueeze(0)

    iou_ = get_IoU(frame_pred, frame_label)
    invisible_iou_= iou(frame_pred - vm_no_crop_gt, frame_label - vm_no_crop_gt)
    if (frame_label - vm_no_crop_gt).sum()==0:
        counts-=1
    return iou_.sum(),  invisible_iou_, counts


def iou(pred, labels, average=True, return_num=False):
    e = 1e-6
    pred = (pred>0.5).float()
    labels = (labels>0.5).float()
    intersection = pred * labels
    union = (pred + labels) - intersection
    iou = intersection.sum(-1).sum(-1).sum(-1) / (union.sum(-1).sum(-1).sum(-1) + e)
    if return_num:
        num = (iou!=1.).sum()
        return iou[iou!=1].sum(), num

    if average:
        return iou.mean()
    else:
        return iou.sum()
    
def video_iou(pred, labels):
    e = 1e-6
    pred = (pred>0.5).float()
    labels = (labels>0.5).float()
    intersection = pred * labels
    union = (pred + labels) - intersection
    iou = intersection.sum(-1).sum(-1) / (union.sum(-1).sum(-1)+ e)
    return iou

