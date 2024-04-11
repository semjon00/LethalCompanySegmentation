import os
import sys
import einops
import torch
from torch import nn
from torch.utils.data import DataLoader
import datetime
import time

from dataset import LethalDataset, IMAGE_HEIGHT, IMAGE_WIDTH

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
bce_loss = nn.BCELoss()


def torch_2d_any(x):
    return torch.any(torch.any(x, dim=-1), dim=-1)


def describe(model, shape):
    try:
        import torchinfo
        torchinfo.summary(model, shape)
    except ImportError:
        pass


def loss_function(en_truth, lt_truth, seg, det):
    en_seg, lt_seg = seg
    en_det, lt_det = det
    return 0.25 * bce_loss(en_seg, en_truth) + 0.25 * bce_loss(en_det, torch_2d_any(en_truth).float()) + \
           0.25 * bce_loss(lt_seg, lt_truth) + 0.25 * bce_loss(lt_det, torch_2d_any(lt_truth).float())


def visualize(first, second=None):
    from PIL import Image
    import numpy
    if second is None:
        second = first
    img = torch.cat((
        first.unsqueeze(0),
        second.unsqueeze(0),
        second.unsqueeze(0) > 0.5), dim=0) * 253
    img = einops.rearrange(img, 'c h w -> h w c')
    img = img.cpu().numpy().astype(numpy.uint8)
    Image.fromarray(img).show()


def pred_to_masks(seg, det):
    en_seg, lt_seg = seg
    en_det, lt_det = det
    en_seg[en_det < 0.5] = 0
    lt_seg[lt_det < 0.5] = 0
    return en_seg > 0.45, lt_seg > 0.45


def score(en_truth, en_prediction, lt_truth, lt_prediction):
    def compute_iou(prediction, target):
        intersection = torch.logical_and(target, prediction).sum(dim=(-1, -2)).float()
        union = torch.logical_or(target, prediction).sum(dim=(-1, -2)).float()
        union = torch.where(union < 10, torch.tensor(0), union)
        iou = (intersection + 1e-6) / (union + 1e-6)  # Adding epsilon to avoid division by zero
        return iou

    en_importance = (20 - 1) * torch.any(torch.any(en_truth, dim=-1), dim=-1) + 1
    lt_importance = (8 - 1) * torch.any(torch.any(lt_truth, dim=-1), dim=-1) + 1
    en_iou = compute_iou(en_prediction, en_truth > 0.5)
    lt_iou = compute_iou(lt_prediction, lt_truth > 0.5)
    return sum(en_importance * en_iou) + sum(lt_importance * lt_iou)


def eval(model, val_loader):
    with torch.inference_mode():
        loss_total = 0.0
        tot_score = 0.0
        print(f'== {datetime.datetime.now()} ==')
        print('Eval ', end='')
        for batch in val_loader:
            img, en_truth, lt_truth = [x.to(device) for x in batch[1:]]
            seg, det = model(img)
            loss_total += loss_function(en_truth, lt_truth, seg, det)
            en_prediction, lt_prediction = pred_to_masks(seg, det)
            tot_score += score(en_truth, en_prediction, lt_truth, lt_prediction)
            # visualize(lt_truth[0], lt_prediction[0])
        print(f"loss: {loss_total / len(val_loader)}, score: {tot_score}")


def get_model():
    for param in sys.argv[1:]:
        if param.endswith('.pt'):
            print(f'Continuing from {param}')
            return torch.load(param)
    if 'cont.pt' in os.listdir('.'):
        print(f'Continuing from cont.pt')
        return torch.load('cont.pt')
    from codename_bravesnake import BraveSnake as SelectedModel
    print(f'Training new')
    model = SelectedModel().to(device)
    model.train()
    return model


if __name__ == '__main__':
    batch_size = 32
    num_epochs = 10

    model = get_model()
    describe(model, (batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH))

    train_dataset = LethalDataset('../data/train', 0, 96)
    val_dataset = LethalDataset('../data/train', 96, 100)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for t in range(num_epochs):
        b = 0
        for batch in train_loader:
            if b % max(1, len(train_loader) // 10) == 0:
                eval(model, val_loader)
                print(f'Epoch {t}/{num_epochs}, batch {b} / {len(train_loader)}')
            b += 1

            names = batch[0]
            img, en_truth, lt_truth = [x.to(device) for x in batch[1:]]
            optimizer.zero_grad()
            seg, det = model(img)
            loss = loss_function(en_truth, lt_truth, seg, det)
            loss.backward()
            optimizer.step()
        print(f'Saving after epoch {t}')
        torch.save(model, f'./model_{int(time.time())}.pt')
    eval(model, val_loader)
