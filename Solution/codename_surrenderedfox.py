import torchvision
import torchvision.models
import torch
from torch import nn
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights


class SurrenderedFox(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
        self.model = model
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)['out']
        x = self.sigmoid(x)
        seg = x[..., :2, :, :]
        det = torch.mean(torch.mean(x[..., 2:4, :, :], dim=-1), dim=-1)
        return seg.permute((1, 0, 2, 3)), det.permute((1, 0))
