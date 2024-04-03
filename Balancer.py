import hashlib
import json
import os
from PIL import Image
import numpy as np

ENEMY_CUTOFF = 40
LOOT_CUTOFF = 20

def load_desc():
    if os.path.isfile('data/descriptions.txt'):
        with open('data/descriptions.txt', 'r') as f:
            t = f.read()
            stats = [x.split('+') for x in t.split('\n')]
            stats = [[x[0], int(x[1]), int(x[2])] for x in stats]
            return stats
    else:
        return None

def calc_desc(names_all):
    def pxcnt(img):
        return int(0.5 + sum(sum(np.array(img, dtype=float))) / 255.0)

    stats = []
    for i in range(len(names_all)):
        if i % 1000 == 0:
            print(f'Loading {i}/{len(names_all)}...')
        name = names_all[i]
        with Image.open(f'data/masks_enemies/{name}') as en:
            with Image.open(f'data/masks_loot/{name}') as lt:
                stats.append([name, pxcnt(en), pxcnt(lt)])
    open('data/descriptions.txt', 'w').write("\n".join([f'{x[0]}+{x[1]}+{x[2]}' for x in stats]))
    return stats


def display_dist(desc, name=None):
    if name is None:
        name = ''
    hash = hashlib.sha1(str(desc).encode()).hexdigest()
    print(f'=== Displaying distribution for desc {name} {hash} (len: {len(desc)}) ===')
    bins_thresholds = [0, 1, 5, 10, 20, 40, 80, 150, 300, 500, 2000, 5000, 10000, 20000, 40000, 100000]
    def bin_i(v):
        for i in range(len(bins_thresholds)):
            if bins_thresholds[i] > v:
                return i - 1
        return len(bins_thresholds) - 1

    en_bins = [0 for _ in range(len(bins_thresholds))]
    lt_bins = [0 for _ in range(len(bins_thresholds))]
    for i in range(len(desc)):
        en_bins[bin_i(desc[i][1])] += 1
        lt_bins[bin_i(desc[i][2])] += 1
    print(f'En bins: {en_bins}')
    print(f'Lt bins: {lt_bins}')

    def stats(s, relevant_frames):
        tot = len(relevant_frames)
        en = len([f for f in relevant_frames if f[1] >= ENEMY_CUTOFF])
        lt = len([f for f in relevant_frames if f[2] >= LOOT_CUTOFF])
        both = len([f for f in relevant_frames if f[1] >= ENEMY_CUTOFF and f[2] >= LOOT_CUTOFF])
        empty = tot - (en + lt - both)
        print(f'S:{s}, total:{tot}, '
              f'has_enemies:{en} ({100*en/tot:.2f}%), has_loot:{lt} ({100*lt/tot:.2f}%), '
              f'has_both:{both} ({100*both/tot:.2f}%), has_nothing:{empty} ({100*empty/tot:.2f}%)')

    print()
    series = sorted(list(set([x[0].split('_')[0] for x in desc])))
    for s in series:
        relevant_frames = [f for f in desc if f[0].startswith(s)]
        stats(s, relevant_frames)
    stats('all', desc)
    print('========================================================================')
    print()


def balancer_core(desc):
    import random
    def info(fdesc):
        # series, enemies_present, loot_present, is_invalid
        return fdesc[0].split('_')[0], fdesc[1] >= ENEMY_CUTOFF, fdesc[2] >= LOOT_CUTOFF, 0 < fdesc[1] < ENEMY_CUTOFF or 0 < fdesc[2] < LOOT_CUTOFF

    train = []
    final_test = []
    preliminary_test = []

    train_target_total = 15000
    train_target_monsters = train_target_total * 20 // 100
    train_target_loot = train_target_total * 40 // 100

    invalid = []
    train_monsters = []
    train_loot = []
    train_empty = []
    for file in desc:
        fi = info(file)
        if fi[3] or fi[0] in ['8540', '1486']:
            invalid.append(file)
            continue
        if fi[0] in ['4062', '3631']:
            pr = 0.7 if fi[0] == '4062' else 0.3
            (final_test if random.random() < pr else preliminary_test).append(file)
            continue
        if fi[1]:
            train_monsters.append(file)
            continue
        if fi[2]:
            train_loot.append(file)
            continue
        train_empty.append(file)
        continue


    random.seed(1337)
    train.extend(random.sample(train_monsters, train_target_monsters))
    train_target_loot -= len([f for f in train if f[2] > 0])
    train.extend(random.sample(train_loot, train_target_loot))
    train.extend(random.sample(train_empty, train_target_total - len(train)))

    display_dist(train, 'train')
    display_dist(preliminary_test, 'preliminary_test')
    display_dist(final_test, 'final_test')
    display_dist(preliminary_test + final_test, 'test_combined')
    return train, preliminary_test, final_test


def move_them(desc, folder):
    def mvinto(filename, into):
        for t in ['screenshots', 'masks_enemies', 'masks_loot']:
            os.makedirs(f'data/{into}/{t}/', exist_ok=True)
            os.rename(f'data/{t}/{filename}', f'data/{into}/{t}/{filename}')

    input('Actually move the files? FOR REAL?')
    for d in desc:
        fn = d[0]
        mvinto(fn, folder)


if __name__ == '__main__':
    names_all = os.listdir('data/screenshots')
    desc = load_desc()
    if desc is None:
       calc_desc(names_all)
       desc = load_desc()

    display_dist(desc, 'unfiltered')
    train, preliminary_test, final_test = balancer_core(desc)
    move_them(train, 'train')
    move_them(preliminary_test, 'preliminary_test')
    move_them(final_test, 'final_test')
