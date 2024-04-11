import base64
import sys
import zlib
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from Solution.dataset import LethalDataset
from Solution.training import pred_to_masks
from pycocotools import mask as coco_mask

def encode_mask(mask: np.ndarray):
    rle = coco_mask.encode(np.asfortranarray(mask))["counts"]
    return base64.b64encode(zlib.compress(rle)).decode('utf-8')


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
dataset = LethalDataset('../data/test', 96, 100)
loader = DataLoader(dataset, batch_size=1, shuffle=True)  # !=1 is not supported

submission_df = pd.DataFrame(columns=['ImageID', 'RleMonsters', 'RleLoot'])
model = torch.load(sys.argv[1])
with torch.inference_mode():
    i = 0
    for batch in loader:
        i += 1
        if i % 20 == 0:
            print('.', end='')
            if i % 1000 == 0:
                print('|')
        name, img = batch[0], batch[1].to(device)
        seg, det = model(img)
        en_prediction, lt_prediction = pred_to_masks(seg, det)

        en_encoded = encode_mask(en_prediction[0])[1]
        lt_encoded = encode_mask(lt_prediction[0])[1]
        submission_df.loc[len(submission_df.index)] = [name, en_encoded, lt_encoded]


submission = pd.DataFrame(
    {'img_id': submission_df['ImageID'], 'enemy_rle': submission_df['RleMonsters'], 'loot_rle': submission_df['RleLoot']},
    columns=['img_id', 'enemy_rle', 'loot_rle'])
submission.to_csv(f'submission_{datetime.now()}.txt', index=False)
print('End')


if __name__ == '__main__':
    pass