import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
import datetime
import time

from submission import pred_to_masks, score
from dataset import LethalDataset, IMAGE_HEIGHT, IMAGE_WIDTH

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
bce_loss = nn.BCELoss()


def log(x):
    print(x)
    with open('train.log', 'a') as f:
        f.write(f'{datetime.datetime.now()} | {x}\n')


def describe(model, shape):
    try:
        import torchinfo
        torchinfo.summary(model, shape)
    except ImportError:
        pass


def loss_function(en_truth, lt_truth, seg, det):
    def torch_2d_any(x):
        return torch.any(torch.any(x, dim=-1), dim=-1)

    en_seg, lt_seg = seg
    en_det, lt_det = det
    return 0.4 * bce_loss(en_seg, en_truth) + 0.1 * bce_loss(en_det, torch_2d_any(en_truth).float()) + \
           0.4 * bce_loss(lt_seg, lt_truth) + 0.1 * bce_loss(lt_det, torch_2d_any(lt_truth).float())


def eval(model, val_loader):
    with torch.inference_mode():
        loss_total = 0.0
        tot_score = 0.0
        for batch in val_loader:
            img, en_truth, lt_truth = [x.to(device) for x in batch[1:]]
            seg, det = model(img)
            loss_total += loss_function(en_truth, lt_truth, seg, det)
            en_prediction, lt_prediction = pred_to_masks(seg, det)
            tot_score += score(en_truth, en_prediction, lt_truth, lt_prediction)
            # visualize(lt_truth[0], lt_prediction[0])
        log(f"Eval loss: {loss_total / len(val_loader)}, score: {tot_score}")


def get_model():
    for param in sys.argv[1:]:
        if param.endswith('.pt'):
            log(f'Continuing from {param}')
            return torch.load(param)
    if 'cont.pt' in os.listdir('.'):
        log(f'Continuing from cont.pt')
        return torch.load('cont.pt')
    from codename_insaneboa import InsaneBoa as SelectedModel
    log(f'Training new {SelectedModel}')
    model = SelectedModel().to(device)
    model.train()
    return model


if __name__ == '__main__':
    batch_size = 4
    num_epochs = 15

    model = get_model()
    describe(model, (batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH))

    train_dataset = LethalDataset('../data/train', 0, 96, augmentations=True)
    val_dataset = LethalDataset('../data/train', 96, 100)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    parameters = model.trainable_parameters() if hasattr(model, 'trainable_parameters') else model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_loader) * max(1, num_epochs - 3), eta_min=1e-5)
    train_loss = -1
    train_loss_tot = 1
    for t in range(num_epochs):
        b = 0
        for batch in train_loader:
            if b % max(1, len(train_loader) // 10) == 0:
                log(f'Epoch {t}/{num_epochs}, batch {b} / {len(train_loader)}:')
                eval(model, val_loader)
                log(f'Train loss: {train_loss / train_loss_tot}')
                train_loss, train_loss_tot = 0.0, 0
            names = batch[0]
            img, en_truth, lt_truth = [x.to(device) for x in batch[1:]]
            optimizer.zero_grad()
            seg, det = model(img)
            loss = loss_function(en_truth, lt_truth, seg, det)
            loss.backward()
            train_loss += float(loss)
            train_loss_tot += 1
            optimizer.step()
            scheduler.step()
            b += 1
        log(f'Saving after epoch {t}')
        torch.save(model, f'./model_{int(time.time())}.pt')
    eval(model, val_loader)
