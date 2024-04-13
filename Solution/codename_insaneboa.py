import einops
import torch
from torch import nn
import torch.nn.functional as F


class InsaneBoa(nn.Module):
    def __init__(self):
        super().__init__()
        self.snake = torch.load('BraveSnake_v3.pt', map_location=torch.device('cpu'))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        def spawn_conv(n_from, n_to):
            return nn.Sequential(
                nn.Conv2d(n_from, n_to, kernel_size=3, padding='same'),
                nn.BatchNorm2d(n_to),
                nn.ReLU(),
            )
        head = nn.Sequential()
        recipie = [16 + 3, 32, 32, 32, 32]
        for i in range(len(recipie) - 1):
            head.append(spawn_conv(recipie[i], recipie[i + 1]))
        head.append(nn.Conv2d(recipie[-1], 4, kernel_size=1, padding='same'))
        self.head = head
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.pad(x, (0, 36, 0, 56, 0, 0, 0, 0))

        reduced = einops.reduce(x, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=2, w2=2)
        features = self.snake.model.encoder(reduced)
        decoder_output = self.snake.model.decoder(*features)
        decoder_output = self.upsample(decoder_output)

        x = torch.cat((x, decoder_output), dim=1)
        x = self.head(x)

        x = x[:, :, :-56, :-36]
        x = self.sigmoid(x)
        seg = x[..., :2, :, :]
        det = torch.mean(torch.mean(x[..., 2:4, :, :], dim=-1), dim=-1)
        return seg.permute((1, 0, 2, 3)), det.permute((1, 0))
