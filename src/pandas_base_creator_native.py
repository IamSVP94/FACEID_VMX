import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from src.utils import recognator

cols = ['path']
cols.extend([i for i in range(0, 512)])
df = pd.DataFrame(columns=cols)

# ------------------ celebrity etalons ------------------
PERSONS_MAIN_DIR = Path('/home/vid/Downloads/datasets/lfw_native/')
etalons_dir_path = list(PERSONS_MAIN_DIR.glob('*'))
p_bar = tqdm(etalons_dir_path, colour='green', leave=False)
for img_idx, dir_path in enumerate(p_bar):
    imgs = list(dir_path.glob('*_best.jpg'))
    if not imgs:
        # continue  # comment this str for full dataset
        imgs = list(dir_path.glob('*.jpg'))
    img_path = imgs[0]

    for_write = [img_path]
    label = img_path.parts[-2]
    p_bar.set_description(f'{label}')
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    emb = recognator.get(img, show=False)

    for_write.extend(emb.tolist()[0])
    df.loc[label, :] = for_write
# ================= /celebrity etalons ==================

# --------------- thispersondoesnotexist ----------------
TPDNE_MAIN_DIR = Path('/home/vid/Downloads/datasets/thispersondoesnotexist')
tpdne_path = list(TPDNE_MAIN_DIR.glob('*.jpg'))
p_bar = tqdm(tpdne_path, colour='yellow', leave=False)
for img_idx, tpdne_path in enumerate(p_bar):
    for_write = [tpdne_path]
    label = tpdne_path.stem
    p_bar.set_description(f'{label}')
    img = cv2.cvtColor(cv2.imread(str(tpdne_path)), cv2.COLOR_BGR2RGB)
    # img = brightness_changer(img, etalon=brightness_etalon)
    emb = recognator.get(img, show=False)
    for_write.extend(emb.tolist()[0])
    df.loc[label, :] = for_write
# =============== /thispersondoesnotexist ===============

# ---------------------- employee -----------------------
OFFICE_PERSONS_MAIN_DIR = Path('/home/vid/hdd/file/project/recog_datasets/LABELED_FACES/LABELED_full/')
etalons_path = list(OFFICE_PERSONS_MAIN_DIR.glob('**/*_best.jpg'))
p_bar = tqdm(etalons_path, colour='blue', leave=False)
for img_idx, img_path in enumerate(p_bar):
    for_write = [img_path]
    label = img_path.parts[-2]
    p_bar.set_description(f'{label}')
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    # img = brightness_changer(img, etalon=brightness_etalon)
    emb = recognator.get(img, show=True)
    for_write.extend(emb.tolist()[0])
    df.loc[label, :] = for_write
# ====================== /employee ======================

df.to_csv(f'n={len(df)}_native.csv')
