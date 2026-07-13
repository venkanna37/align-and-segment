import torch
import torch.nn as nn


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet11(nn.Module):
    def __init__(self, input_channels, encoder_channels):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.first_block = nn.Sequential(
            nn.Conv2d(input_channels, self.encoder_channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(self.encoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        self.layer0 = self._make_layer(self.encoder_channels[0], self.encoder_channels[0], 1, stride=2)
        self.layer1 = self._make_layer(self.encoder_channels[0], self.encoder_channels[1], 1, stride=2)
        self.layer2 = self._make_layer(self.encoder_channels[1], self.encoder_channels[2], 1, stride=2)
        self.layer3 = self._make_layer(self.encoder_channels[2], self.encoder_channels[3], 1, stride=2)
        self.layer4 = self._make_layer(self.encoder_channels[3], self.encoder_channels[4], 1, stride=2)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.first_block(x)
        x2 = self.layer0(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)

        return x1, x2, x3, x4, x5, x6


class ResNet18(nn.Module):
    def __init__(self, input_channels, encoder_channels):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.first_block = nn.Sequential(
            nn.Conv2d(input_channels, self.encoder_channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(self.encoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        self.layer0 = self._make_layer(self.encoder_channels[0], self.encoder_channels[0], 1, stride=2)
        self.layer1 = self._make_layer(self.encoder_channels[0], self.encoder_channels[1], 2, stride=2)
        self.layer2 = self._make_layer(self.encoder_channels[1], self.encoder_channels[2], 2, stride=2)
        self.layer3 = self._make_layer(self.encoder_channels[2], self.encoder_channels[3], 2, stride=2)
        self.layer4 = self._make_layer(self.encoder_channels[3], self.encoder_channels[4], 2, stride=2)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.first_block(x)
        x2 = self.layer0(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)

        return x1, x2, x3, x4, x5, x6


class ResNet34(nn.Module):
    def __init__(self, input_channels, encoder_channels):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.first_block = nn.Sequential(
            nn.Conv2d(input_channels, self.encoder_channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(self.encoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        self.layer0 = self._make_layer(self.encoder_channels[0], self.encoder_channels[0], 1, stride=2)
        self.layer1 = self._make_layer(self.encoder_channels[0], self.encoder_channels[1], 3, stride=2)
        self.layer2 = self._make_layer(self.encoder_channels[1], self.encoder_channels[2], 4, stride=2)
        self.layer3 = self._make_layer(self.encoder_channels[2], self.encoder_channels[3], 6, stride=2)
        self.layer4 = self._make_layer(self.encoder_channels[3], self.encoder_channels[4], 3, stride=2)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.first_block(x)
        x2 = self.layer0(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)

        return x1, x2, x3, x4, x5, x6

