import base64
import sys
import zlib
from datetime import datetime
import einops
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import LethalDataset


def visualize(first, second=None, third=None):
    from PIL import Image
    import numpy
    if second is None:
        second = first
    if third is None:
        third = second
    images = [first, second, third]
    for i in range(len(images)):
        if len(images[i].shape) > 2:
            images[i] = einops.reduce(images[i], 'c h w -> h w', 'mean')
        images[i] = images[i].unsqueeze(0)
    img = torch.clamp(torch.cat(images, dim=0), 0.0, 1.0) * 255
    img = einops.rearrange(img, 'c h w -> h w c')
    img = img.cpu().numpy().astype(numpy.uint8)
    Image.fromarray(img).show()


def torch_2d_any(x):
    return torch.any(torch.any(x, dim=-1), dim=-1)


def max_score(en_truth, lt_truth):
    en_importance = (20 - 1) * torch_2d_any(en_truth) + 1
    lt_importance = (8 - 1) * torch_2d_any(lt_truth) + 1
    return (sum(en_importance[en_importance > 2.0]), sum(en_importance[en_importance < 2.0]),
            sum(lt_importance[lt_importance > 2.0]), sum(lt_importance[lt_importance < 2.0]))


def score(en_truth, en_prediction, lt_truth, lt_prediction, detailed=False):
    def compute_iou(prediction, target):
        intersection = torch.logical_and(target, prediction).sum(dim=(-1, -2)).float()
        union = torch.logical_or(target, prediction).sum(dim=(-1, -2)).float()
        union = torch.where(union < 10, torch.tensor(0), union)
        iou = (intersection + 1e-6) / (union + 1e-6)  # Adding epsilon to avoid division by zero
        return iou

    en_importance = (20 - 1) * torch_2d_any(en_truth) + 1
    lt_importance = (8 - 1) * torch_2d_any(lt_truth) + 1
    en_iou = compute_iou(en_prediction, en_truth > 0.5)
    lt_iou = compute_iou(lt_prediction, lt_truth > 0.5)
    if not detailed:
        return sum(en_importance * en_iou) + sum(lt_importance * lt_iou)
    else:
        return (sum(20 * en_iou[en_importance > 2.0]), sum(en_iou[en_importance < 2.0]),
                sum(8 * lt_iou[lt_importance > 2.0]), sum(lt_iou[lt_importance < 2.0]))


def pred_to_masks(seg, det, thresholds=None):
    if thresholds is None:
        thresholds = (0.30, 0.30, 0.5, 0.5)
    en_seg, lt_seg = seg
    en_det, lt_det = det
    en_seg[en_det < thresholds[2]] = 0
    lt_seg[lt_det < thresholds[3]] = 0
    return en_seg > thresholds[0], lt_seg > thresholds[1]


def encode_mask(mask: np.ndarray):
    from pycocotools import mask as coco_mask
    rle = coco_mask.encode(np.asfortranarray(mask))["counts"]
    return base64.b64encode(zlib.compress(rle)).decode('utf-8')


def create_submission():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    dataset = LethalDataset('../data/test')
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    submission_df = pd.DataFrame(columns=['ImageID', 'RleMonsters', 'RleLoot'])
    model = torch.load(sys.argv[1]).to(device)
    with torch.inference_mode():
        print(f'{len(loader)} | ', end='')
        for batch in loader:
            print('.', end='')
            name, img = batch[0], batch[1].to(device)
            seg, det = model(img)

            # # Very lazy visualizer
            # from PIL import Image
            # img_grayscale = einops.repeat(torch.mean(img, dim=1), 'a b c -> a 3 b c')
            # for i in range(len(seg[0])):
            #     tg = img_grayscale[i].clone()
            #     tg[0] = tg[0] + seg[0][i] * (det[0][i] > 0.5)
            #     tg[1] = tg[1] + seg[1][i] * (det[1][i] > 0.5)
            #     tg[2] = tg[2]
            #     tg = tg.clamp(0, 1.0)
            #     tg = Image.fromarray(einops.rearrange(255.0 * tg, 'c a b -> a b c').cpu().numpy().astype(np.uint8))
            #     tg.convert('RGB').save('savedir/' + name[i] + '-0' '.png')

            en_prediction, lt_prediction = pred_to_masks(seg, det)
            for i in range(len(name)):
                en_encoded = encode_mask(en_prediction[i].detach().cpu())
                lt_encoded = encode_mask(lt_prediction[i].detach().cpu())
                submission_df.loc[len(submission_df.index)] = [name[i], en_encoded, lt_encoded]

    submission = pd.DataFrame({'img_id': submission_df['ImageID'],
                               'enemy_rle': submission_df['RleMonsters'],
                               'loot_rle': submission_df['RleLoot']},
                              columns=['img_id', 'enemy_rle', 'loot_rle'])
    fn = str(datetime.now())
    for s in ' -:':
        fn = fn.replace(s, '_')
    submission.to_csv(f'submission_{fn}.csv', index=False)
    print('End')


def thresholds_stats():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = LethalDataset('../data/train', percent_from=96, percent_to=100)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = torch.load(sys.argv[1]).to(device)
    max_scores = [0, 0, 0, 0]
    scores = {}
    levels_seg = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5]
    levels_det = [0.1, 0.3, 0.5, 0.7]
    with torch.inference_mode():
        print(f'{len(loader)} | ', end='')
        for batch in loader:
            print('.', end='')
            img, en_truth, lt_truth = [x.to(device) for x in batch[1:]]
            seg, det = model(img)
            seg.detach()
            det.detach()
            for i in range(4):
                max_add = max_score(en_truth, lt_truth)
                max_scores[i] += max_add[i]
            for level_seg in levels_seg:
                for level_det in levels_det:
                    score_id = f'seg{level_seg}_det{level_det}'
                    if score_id not in scores:
                        scores[score_id] = [0, 0, 0, 0]
                    thres = (level_seg, level_seg, level_det, level_det)
                    en_prediction, lt_prediction = pred_to_masks(seg, det, thres)
                    # visualize(img[0], lt_truth[0], lt_prediction[0])

                    sc = score(en_truth, en_prediction, lt_truth, lt_prediction, detailed=True)
                    for i in range(4):
                        scores[score_id][i] += sc[i]
    print(f'\nMax scores: {max_scores}\n')

    for n, ss in scores.items():
        scores_str = [f'{float(s):.2f}' for s in ss]
        perc = [f'{ss[i] / max_scores[i] * 100:.2f}%' for i in range(4)]
        print(f'{n}: {scores_str} / {perc} / {sum(ss)}')


if __name__ == '__main__':
    # thresholds_stats()
    create_submission()
