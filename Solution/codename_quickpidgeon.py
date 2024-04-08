# UNet stolen from the segmentation practice session

import einops
import torch
from torch import nn

from dataset import IMAGE_WIDTH, IMAGE_HEIGHT


class QuickPidgeon(nn.Module):
    def __init__(self, num_kernels=32, num_channels=3, num_classes=2, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
        super(QuickPidgeon, self).__init__()

        self.img_width = img_width
        self.img_height = img_height
        self.num_kernels = num_kernels
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, num_kernels, kernel_size=3, padding='same'),
            nn.ReLU(),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_kernels, 2 * num_kernels, kernel_size=3, padding='same'),
            nn.ReLU(),
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = nn.Conv2d(2 * num_kernels, num_kernels, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = nn.Conv2d(3 * num_kernels, num_kernels, kernel_size=3, padding=1)

        self.classifier = nn.Conv2d(2 * num_kernels, num_classes, kernel_size=1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1_out = self.conv1(x)
        maxpool1_out = self.maxpool1(conv1_out)
        conv2_out = self.conv2(maxpool1_out)
        maxpool2_out = self.maxpool2(conv2_out)

        upconv1_out = self.upsample1(maxpool2_out)  # Upsample
        upconv1_out = self.conv_up1(upconv1_out)  # Then convolve
        upconv2_out = self.upsample2(torch.cat((upconv1_out, conv2_out), dim=1))  # Upsample
        upconv2_out = self.conv_up2(upconv2_out)

        x = self.Sigmoid(self.classifier(torch.cat((upconv2_out, conv1_out), dim=1)))
        x = einops.rearrange(x, 'b c h w -> c b h w')
        return x
