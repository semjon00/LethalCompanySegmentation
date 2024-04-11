import segmentation_models_pytorch as smp
import einops
import torch
from torch import nn
import torch.nn.functional as F


class BraveSnake(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.pad(x, (0, 36, 0, 56, 0, 0, 0, 0))

        x = einops.reduce(x, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=2, w2=2)
        x = self.model(x)
        x = einops.repeat(x, 'b c h w -> b c (h h2) (w w2)', h2=2, w2=2)

        x = x[:, :, :-56, :-36]
        x = self.sigmoid(x)
        seg = x[..., :2, :, :]
        det = torch.mean(torch.mean(x[..., 2:4, :, :], dim=-1), dim=-1)
        return seg.permute((1, 0, 2, 3)), det.permute((1, 0))
