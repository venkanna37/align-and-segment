import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLocalizationNet(nn.Module):
    def __init__(self, input_channels=16):
        super(GlobalLocalizationNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x4l):
        x = torch.cat(x4l, dim=1)
        B = x.size(0)
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        x = F.silu(self.bn3(self.conv3(x)))
        x = F.silu(self.bn4(self.conv4(x)))

        x = self.pool(x)
        x = x.view(B, -1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        params = torch.tanh(self.fc3(x))

        theta = params[:, 0]
        tx = params[:, 1]
        ty = params[:, 2]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        affine_matrix = torch.stack([
            torch.stack([cos_theta, -sin_theta, tx], dim=1),
            torch.stack([sin_theta, cos_theta, ty], dim=1)
        ], dim=1)                                              # (B, 2, 3)

        return affine_matrix


class FlowField(nn.Module):
    def __init__(self, input_channels=16):
        super(GlobalLocalizationNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 3)

        # Initialize affine transform to identity
        self.fc2.weight.data.zero_()
        # self.fc2.bias.data.copy_(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float))

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.conv1(x))   # (B, 32, H/2, W/2)
        x = F.relu(self.conv2(x))   # (B, 64, H/4, W/4)
        x = F.relu(self.conv3(x))   # (B, 128, H/8, W/8)
        x = self.pool(x)            # (B, 128, 4, 4)
        x = x.view(B, -1)           # (B, 128)
        x = F.relu(self.fc1(x))     # (B, 256)
        params = self.fc2(x)         # (B, 3)
        theta = params[:, 0]
        tx = params[:, 1]
        ty = params[:, 2]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        affine_matrix = torch.tensor([[cos_theta, -sin_theta, tx], [sin_theta, cos_theta, ty]],
                                     dtype=torch.float32)
        return affine_matrix

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


def STN_mask(input_fmap, theta):

    B, C, H, W = input_fmap.shape
    grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=True)
    mask = input_fmap.permute(0, 2, 3, 1)
    grid = grid * mask
    # Sample the input image with bilinear interpolation
    out_fmap = F.grid_sample(input_fmap, grid, align_corners=True)

    return out_fmap


def STN_flow(input_fmap, flow):
    """
    Spatial Transformer Network in PyTorch using affine_grid and grid_sample.
    """
    out_fmap = F.grid_sample(input_fmap, flow * input_fmap.permute(0, 2, 3, 1),
                             align_corners=True)

    return out_fmap



class just_affine(nn.Module):
    def __init__(self):
        super(just_affine, self).__init__()
        self.affin_params = torch.nn.Parameter(torch.zeros(3), requires_grad=True)

    def forward(self, x):
        theta = self.affin_params[0]
        tx = self.affin_params[1]
        ty = self.affin_params[2]

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        affine_matrix = torch.stack([
            torch.stack([cos_theta, -sin_theta, tx], dim=1),
            torch.stack([sin_theta, cos_theta, ty], dim=1)
        ], dim=1)  # (B, 2, 3)

        transformed_x = spatial_transformer_network(x, affine_matrix.unsqueeze(0))

        return transformed_x, affine_matrix.unsqueeze(0)  # Return the transformed image and affine matrix