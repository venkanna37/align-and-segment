import torch.nn.functional as F
from segmentation_models_pytorch.losses import JaccardLoss, DiceLoss
from segmentation_models_pytorch.losses.constants import BINARY_MODE


def loss_for_seg(pd_mask, gt_mask, wt_mask, loss_type):
    if loss_type == 'cross_entropy':
        loss = F.binary_cross_entropy_with_logits(pd_mask, gt_mask, reduction='none')
        loss = (loss * wt_mask).sum() / ((wt_mask > 0).sum() + 1e-6)
    elif loss_type == 'iou':
        iou_criterion = JaccardLoss(mode=BINARY_MODE, from_logits=True)
        loss = iou_criterion(pd_mask, gt_mask * wt_mask)
    elif loss_type == 'dice':
        dice_criterion = DiceLoss(mode=BINARY_MODE, from_logits=True)
        loss = dice_criterion(pd_mask, gt_mask * wt_mask)
    else:
        raise NotImplementedError(f'Loss type {loss_type} is not implemented.')
    return loss
