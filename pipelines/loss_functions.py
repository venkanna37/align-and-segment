import torch
import torch.nn.functional as F
from segmentation_models_pytorch.losses import JaccardLoss, DiceLoss
from segmentation_models_pytorch.losses.constants import BINARY_MODE

from .utils import add_third_row


def loss_for_align(pred_affine, gt_affine, aligned_mask=None, gt_mask=None, weight_mask=None, device='cpu',
                   loss_type='mse', model_name='TNet'):
    """
    Compute loss for TNet and TNet_FF
    :param pred_affine: Predicted affine matrix of shape (B, 2, 3)            \phi hat
    :param aligned_mask: aligned mask with \phi hat                           I_m \circ \phi hat
    :param gt_mask: True mask (can be augmented or prediction from SNet)      I_m hat
    :param weight_mask: Weight mask for the loss computation                  I_1 \circ \phi hat
    :param device:
    :param loss_type:
    :param model_name:
    :return: loss value
    """

    if model_name == 'TNet':
        mse_criterion = torch.nn.MSELoss()
        iou_criterion = JaccardLoss(mode=BINARY_MODE, from_logits=False)
        if loss_type == 'mse':
            loss = mse_criterion(pred_affine, gt_affine)
        elif loss_type == 'frobenius':
            pred_affine = add_third_row(pred_affine)
            gt_affine = add_third_row(gt_affine)
            identity_matrix = torch.eye(3).unsqueeze(0).repeat(gt_affine.shape[0], 1, 1).to(device)
            matrix = (pred_affine @ gt_affine) - identity_matrix
            loss = torch.sqrt((matrix ** 2).sum(dim=(1, 2)) + 1e-8).mean()
        elif loss_type == 'iou' or loss_type == 'cross_entropy':
            if loss_type == 'iou':
                loss = iou_criterion(aligned_mask, gt_mask * weight_mask)
            elif loss_type == 'cross_entropy':
                loss = F.binary_cross_entropy(aligned_mask, gt_mask, reduction='none')
                loss = (loss * weight_mask).sum() / (weight_mask.sum() + 1e-6)
        else:
            raise NotImplementedError(f'Loss type {loss_type} is not implemented.')
    else:
        raise NotImplementedError(f'Model {model_name} is not implemented for alignment loss.')
    return loss

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
