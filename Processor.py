import os
import shutil
from PIL import Image
import PIL
import numpy as np

VERBATIM = 'layer557520895'
SIMPLIFIED = 'layer1631262591'
CLEARED = 'layer1630738239'
LOOT_ONLY = 'layer1073741888'
ENEMIES_ONLY = 'layer1074266112'

def downscale(image: PIL.Image):
    new_width = image.width // 2
    new_height = image.height // 2
    return image.resize((new_width, new_height), Image.BOX)

def difference(a, b, treshold=0.5):
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)
    c = np.sum(np.abs(a - b), axis=2) > treshold
    return c

def blue_screen_mask(image: PIL.Image):
    image = np.array(image).astype(np.float32)
    image[:,:,2] = -image[:,:,2] + 255
    mask = np.sum(image, axis=2) > 2.5
    # Image.fromarray(mask.astype(np.uint8) * 255, mode='L').show()
    return mask

def create_mask(differences_spotted, only):
    potential = blue_screen_mask(only)
    mask = np.logical_and(potential, differences_spotted)
    return np.sum(np.sum(mask)), Image.fromarray(mask.astype(np.uint8) * 255, mode='L')


if __name__ == '__main__':
    for ddir in ['masks_loot', 'masks_enemies', 'fullres', 'downscaled']:
        os.makedirs(f'data/{ddir}', exist_ok=True)
    names_all = os.listdir('data/new')
    name_stubs = set()
    for n in names_all:
        name_stubs.add('_'.join(n.split('_')[:2]))
    name_stubs = sorted(list(name_stubs))

    for n in name_stubs:
        if os.path.exists(f'data/masks_enemies/{n}.png'):
            continue

        shutil.copy(f'data/new/{n}_{VERBATIM}.png', f'data/fullres/{n}.png')
        with Image.open(f'data/new/{n}_{VERBATIM}.png') as im:
            downscale(im).save(f'data/downscaled/{n}.png')

        with Image.open(f'data/new/{n}_{SIMPLIFIED}.png') as simplified:
            with Image.open(f'data/new/{n}_{LOOT_ONLY}.png') as loot_only:
                with Image.open(f'data/new/{n}_{ENEMIES_ONLY}.png') as enemies_only:
                    with Image.open(f'data/new/{n}_{CLEARED}.png') as cleared:
                        differences_spotted = difference(simplified, cleared)
                        pl, il = create_mask(differences_spotted, loot_only)
                        il.save(f"data/masks_loot/{n}.png")
                        pe, ie = create_mask(differences_spotted, enemies_only)
                        ie.save(f"data/masks_enemies/{n}.png")
                        print(n, pl, pe)
