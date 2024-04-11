import base64
import sys
import zlib
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pycocotools import mask as coco_mask

from dataset import LethalDataset
from training import pred_to_masks

def encode_mask(mask: np.ndarray):
    rle = coco_mask.encode(np.asfortranarray(mask))["counts"]
    return base64.b64encode(zlib.compress(rle)).decode('utf-8')


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
dataset = LethalDataset('../data/test')
loader = DataLoader(dataset, batch_size=4, shuffle=True)

submission_df = pd.DataFrame(columns=['ImageID', 'RleMonsters', 'RleLoot'])
model = torch.load(sys.argv[1]).to(device)
with torch.inference_mode():
    i = 0
    for batch in loader:
        i += 1
        if i % 10 == 0:
            print(f'{i}')
        name, img = batch[0], batch[1].to(device)
        seg, det = model(img)
        en_prediction, lt_prediction = pred_to_masks(seg, det)
        for i in range(len(name)):
            en_encoded = encode_mask(en_prediction[i].detach().cpu())
            lt_encoded = encode_mask(lt_prediction[i].detach().cpu())
            submission_df.loc[len(submission_df.index)] = [name[i], en_encoded, lt_encoded]


submission = pd.DataFrame(
    {'img_id': submission_df['ImageID'], 'enemy_rle': submission_df['RleMonsters'], 'loot_rle': submission_df['RleLoot']},
    columns=['img_id', 'enemy_rle', 'loot_rle'])
submission.to_csv(f'submission_{str(datetime.now()).replace(" ", "")}.txt', index=False)
print('End')


if __name__ == '__main__':
    pass
