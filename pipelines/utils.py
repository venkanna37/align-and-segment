import torch
import torch.nn.functional as F


def transform_mask_with_random_affine(batch, device, max_shift=50):
    B, C, H, W = batch.shape
    assert H == W, "Batch must be square (H == W)"

    max_shift = (float(max_shift)/2.0) / H
    tx = torch.rand((B, 1), device=device) *  (max_shift*2) - max_shift
    ty = torch.rand((B, 1), device=device) *  (max_shift*2) - max_shift
    theta = torch.rand((B, 1), device=device) *  (max_shift*2) - max_shift

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Create affine matrices (B, 2, 3)
    row1 = torch.cat([cos_theta, -sin_theta, tx], dim=1)
    row2 = torch.cat([sin_theta, cos_theta, ty], dim=1)
    affine_matrix = torch.stack([row1, row2], dim=1)

    # Generate grid and apply transformation
    grid = F.affine_grid(affine_matrix, size=batch.size(), align_corners=False)
    rand_pred_mask = F.grid_sample(batch, grid, align_corners=False)

    # Check for zero masks per sample in batch and replace with identity where rand_pred_mask is zero
    zero_mask = torch.sum(rand_pred_mask.view(B, -1), dim=1) == 0
    identity = torch.eye(2, 3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    affine_matrix[zero_mask] = identity[zero_mask]

    return rand_pred_mask, affine_matrix


def inverse_affine_matrix(aff_mat):
    """Convert (2x3) affine matrix to inverse affine matrix."""
    ones_row = torch.tensor([0, 0, 1], dtype=aff_mat.dtype, device=aff_mat.device)
    ones_row = ones_row.expand(aff_mat.shape[0], 1, 3)          # (B, 1, 3)
    aff_mat = torch.cat([aff_mat, ones_row], dim=1)
    inv_aff_mat = torch.linalg.inv(aff_mat)                     # (B, 3, 3)
    return inv_aff_mat[:, :2, :]                                # return only (2x3) part


def add_third_row(aff_mat):
    """Add third row"""
    ones_row = torch.tensor([0, 0, 1], dtype=aff_mat.dtype, device=aff_mat.device)
    ones_row = ones_row.expand(aff_mat.shape[0], 1, 3)
    return torch.cat([aff_mat, ones_row], dim=1)


def create_affine_matrix(tx, ty, theta, patch_size):
    tx, ty, theta = float(tx) / patch_size, float(ty) / patch_size, torch.tensor(float(theta) / patch_size)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    affine_matrix = torch.tensor([[cos_theta, -sin_theta, tx], [sin_theta, cos_theta, ty]],
                                 dtype=torch.float32)
    return affine_matrix


def warp_mask_with_affine(mask, affine_matrix):
    if affine_matrix.shape == (2, 3):
        affine_matrix = affine_matrix.unsqueeze(0)
    grid = F.affine_grid(affine_matrix, mask.size(), align_corners=True)
    transformed_mask = F.grid_sample(mask, grid, align_corners=True, mode='nearest').squeeze(0)
    return transformed_mask

