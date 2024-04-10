# Baby bootleg UNet

import einops
import torch
from torch import nn
import torch.nn.functional as F

from dataset import IMAGE_WIDTH, IMAGE_HEIGHT


class QuickPidgeon(nn.Module):
    def __init__(self):
        super(QuickPidgeon, self).__init__()
        num_classes = 2
        num_channels = 3

        def spawn_conv(n_from, n_to):
            return nn.Sequential(
                nn.Conv2d(n_from, n_to, kernel_size=3, padding='same'),
                nn.GELU(),
                nn.Conv2d(n_to, n_to, kernel_size=3, padding='same'),
                nn.GELU(),
            )

        path_downward = [num_channels, 16, 32, 64, 128, 256]
        path_upward = [0, 128, 64, 32, 16, num_classes]

        self.conv_down = nn.Sequential()
        for i in range(len(path_downward) - 1):
            self.conv_down.append(spawn_conv(path_downward[i], path_downward[i + 1]))

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up = nn.Sequential()
        for i in range(len(path_upward) - 1):
            extra = path_downward[0 - 1 - i]
            self.conv_up.append(spawn_conv(extra + path_upward[i], path_upward[i + 1]))

        self.enabling_classifier = nn.Sequential(
            nn.Linear(17 * 27 * 8, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )
        self.classifier = nn.Conv2d(path_upward[-1], num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.pad(x, (0, 20, 0, 24, 0, 0, 0, 0))
        residual = []
        for i in range(len(self.conv_down)):
            x = self.conv_down[i](x)
            residual += [x]
            x = self.maxpool(x)

        enabling = self.enabling_classifier(einops.rearrange(x[..., :8, :, :], 'b c h w -> b (c h w)'))

        x = residual[-1].new_empty([residual[-1].shape[i] if i != 1 else 0 for i in range(len(residual[-1].shape))])
        for i in range(len(self.conv_up)):
            x = self.conv_up[i](torch.cat((residual[0 - 1 - i], x), dim=1))
            if i != len(self.conv_up) - 1:
                x = self.upsample(x)

        x = self.sigmoid(self.classifier(x))
        x = x[:, :, :-24, :-20]

        x = einops.rearrange(x, 'b c h w -> c b h w')
        enabling = einops.rearrange(enabling, 'b c -> c b')
        return x, enabling
