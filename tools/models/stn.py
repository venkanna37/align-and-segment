import torch
import torch.nn.functional as F

def spatial_transformer_network(input_fmap, theta):
    """
    Spatial Transformer Network in PyTorch using affine_grid and grid_sample.
    Parameters
    ----------
    input_fmap : torch.Tensor
        Input tensor of shape (B, C, H, W)
    theta : torch.Tensor
        Affine transform matrices of shape (B, 2, 3)

    Returns
    -------
    out_fmap : torch.Tensor
        Transformed feature map of shape (B, C, out_H, out_W)
    """
    B, C, H, W = input_fmap.shape
    if theta.size()[1:] == (3, 3): #assuming third row is just 0,0,1
        theta = theta[:, :2, :]

    # Generate the sampling grid
    grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=True)

    # Sample the input image with bilinear interpolation
    out_fmap = F.grid_sample(input_fmap, grid, align_corners=True)

    return out_fmap