import einops
import torch
from torch import nn
from torch.utils.data import DataLoader
import time

from dataset import LethalDataset, IMAGE_HEIGHT, IMAGE_WIDTH

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
bce_loss = nn.BCELoss()


def describe(model, shape):
    try:
        import torchinfo
        torchinfo.summary(model, shape)
    except ImportError:
        pass


def loss_function(en_truth, lt_truth, prediction):
    def half_loss_function(truth, prediction):
        return bce_loss(prediction, truth)
    en_prediction, lt_prediction = prediction
    lt_k = 0.5
    return (1 - lt_k) * half_loss_function(en_truth, en_prediction) + lt_k * half_loss_function(lt_truth, lt_prediction)


# def predict(model, test_loader):
#     threshold = 0.5
#     result = []
#     with torch.no_grad():
#         for batch in test_loader:
#             name, img = batch
#             en_prediction, lt_prediction = model(img)
#             en_prediction = en_prediction > threshold
#             lt_prediction = lt_prediction > threshold
#             for i in range(len(pred)):
#                 result.append((name[i], en_prediction[i], lt_prediction[i]))
#     return result


def visualize(mask_truth, mask_predicted):
    from PIL import Image
    import numpy
    img = torch.cat((
        mask_truth.unsqueeze(0),
        mask_predicted.unsqueeze(0),
        mask_predicted.unsqueeze(0) > 0.5), dim=0) * 253
    img = einops.rearrange(img, 'c h w -> h w c')
    img = img.cpu().numpy().astype(numpy.uint8)
    Image.fromarray(img).show()


def pred_to_masks(masks):
    en_masks = masks[0]
    lt_masks = masks[1]
    return en_masks > 0.3, lt_masks > 0.3


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
    # TODO: bad
    with torch.no_grad():
        loss_total = 0.0
        tot_score = 0.0
        for batch in val_loader:
            img, en_truth, lt_truth = [x.to(device) for x in batch[1:]]
            pred = model(img)
            loss_total += loss_function(en_truth, lt_truth, pred)
            en_prediction, lt_prediction = pred_to_masks(pred)
            tot_score += score(en_truth, en_prediction, lt_truth, lt_prediction)
            # visualize(lt_truth[0], lt_prediction[0])
        print(f"Eval loss: {loss_total / len(val_loader)}, score: {tot_score}")


if __name__ == '__main__':
    batch_size = 8
    num_epochs = 3

    from codename_quickpidgeon import QuickPidgeon as SelectedModel
    model = SelectedModel().to(device)
    describe(model, (batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH))

    train_dataset = LethalDataset('../data/train', 0, 96)
    val_dataset = LethalDataset('../data/train', 96, 100)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, total_steps=num_epochs * len(train_loader), pct_start=0.2, max_lr=1e-3)

    for t in range(num_epochs):
        b = 0
        for batch in train_loader:
            if b % max(1, len(train_loader) // 20) == 0:
                eval(model, val_loader)
                print(f'Epoch {t}/{num_epochs}, batch {b} / {len(train_loader)}')
            b += 1

            names = batch[0]
            img, en_truth, lt_truth = [x.to(device) for x in batch[1:]]
            optimizer.zero_grad()
            pred = model(img)
            loss = loss_function(en_truth, lt_truth, pred)
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f'Saving after epoch {t}')
        torch.save(model, f'./model_{int(time.time())}.pt')
    eval(model, val_loader)
