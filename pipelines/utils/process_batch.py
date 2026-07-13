import torch
import itertools
from torch import nn
import numpy as np
import torch.nn.functional as F
from kornia.geometry.transform import get_tps_transform, warp_image_tps, get_perspective_transform, warp_perspective

class dilated_pixels(nn.Module):
    """
    The border pixel width is based on the kernel size.
    For example, a kernel size of 3 will result in a border pixel width of 1, 5-->2, 7-->3, etc.
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    def forward(self, x):
        return self.max_pool(-x) + x


def dilate_boundary(binary_images, kernel_size=3, iterations=1):

    padding = kernel_size // 2
    device = binary_images.device

    # Create a square kernel of ones (like OpenCV's cv2.getStructuringElement)
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device=device)

    # Apply dilation iteratively
    out = binary_images.float()
    out = F.conv2d(out, kernel, padding=padding)
    out = (out > 0).float()  # Threshold to keep binary values

    return out - binary_images.float()


def gaussian_kernel(kernel_size=5, sigma=1.0, channels=1):
    """Creates a Gaussian kernel for convolution."""
    # 1D Gaussian
    x = torch.arange(kernel_size) - kernel_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()

    # 2D kernel
    kernel_2d = gauss[:, None] * gauss[None, :]
    kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    return kernel_2d

def apply_sobel_with_gaussian(images):
    """
    Applies Gaussian filter before Sobel edge detection on grayscale images.
    Args:
        images: Tensor of shape (B, 3, H, W), assumed to be in [0, 1]
    Returns:
        sobel_magnitude: Tensor of shape (B, 1, H, W)
    """
    device = images.device

    # Convert to grayscale
    gray = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]

    # Apply Gaussian blur
    g_kernel = gaussian_kernel(kernel_size=5, sigma=1.0, channels=1).to(device)
    gray_blurred = F.conv2d(gray, g_kernel, padding=2, groups=1)

    # Sobel kernels
    sobel_x = torch.tensor([[[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0).to(device)

    sobel_y = torch.tensor([[[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]]], dtype=torch.float32).unsqueeze(0).to(device)

    # Shape: (1, 1, 3, 3)
    grad_x = F.conv2d(gray_blurred, sobel_x, padding=1)
    grad_y = F.conv2d(gray_blurred, sobel_y, padding=1)

    # Gradient magnitude
    sobel_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

    return sobel_magnitude


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
    grid = F.affine_grid(affine_matrix, size=batch.size(), align_corners=True)
    rand_pred_mask = F.grid_sample(batch, grid, align_corners=True)

    # Check for zero masks per sample in batch and replace with identity where rand_pred_mask is zero
    zero_mask = torch.sum(rand_pred_mask.view(B, -1), dim=1) == 0
    identity = torch.eye(2, 3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    affine_matrix[zero_mask] = identity[zero_mask]

    return rand_pred_mask, affine_matrix

def transform_mask_with_random_tps(batch, grid_size=4, max_shift=5, p=0.3):
    B, C, H, W = batch.shape
    assert H == W, "Batch must be square (H == W)"

    r1, r2, = 0.9, 0.9
    grid_height, grid_width = grid_size, grid_size
    assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
    target_control_points = torch.Tensor(list(itertools.product(
        np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_height - 1)),
        np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_width - 1)),
    )))
    Y, X = target_control_points.split(1, dim=1)
    target_control_points = torch.cat([X, Y], dim=1)

    source_control_points = target_control_points.clone()

    N = target_control_points.shape[0]
    mask = torch.rand(N) < p
    max_disp = max_shift/H
    noise = torch.randn_like(target_control_points) * max_disp
    source_control_points[mask] += noise[mask]

    kernel_weights, affine_weights = get_tps_transform(source_control_points.unsqueeze(0), target_control_points.unsqueeze(0))
    warped_image = warp_image_tps(batch, target_control_points.unsqueeze(0), kernel_weights, affine_weights)

    return warped_image, source_control_points


def transform_mask_with_random_perspective(batch, max_shift=5, p=0.5):
    B, C, H, W = batch.shape
    assert H == W

    device = batch.device
    max_disp = max_shift / H

    # ---- 1. Canonical corner points (normalized coords)
    margin = 0.9
    target_control_points = torch.tensor([
        [-margin, -margin],
        [margin, -margin],
        [-margin, margin],
        [margin, margin],
    ], device=device)

    source_control_points = target_control_points.clone()

    # ---- 2. Apply random perspective with probability p
    if torch.rand(1) < p:
        noise = (torch.rand_like(source_control_points) * 2 - 1) * max_disp
        source_control_points += noise

    # ---- 3. Compute homography
    H_mat = get_perspective_transform(
        source_control_points.unsqueeze(0),
        target_control_points.unsqueeze(0)
    )

    # ---- 4. Warp
    warped = warp_perspective(batch, H_mat, (H, W))

    return warped, source_control_points