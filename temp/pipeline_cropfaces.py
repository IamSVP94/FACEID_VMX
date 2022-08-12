import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from pathlib import Path

from src.constants import PARENT_DIR, det_thresh, recog_tresh
from src.utils import persons_list_from_csv, detector, recognator, Person, plt_show_img

new_output_dir_path = PARENT_DIR / 'temp' / f'recog_tresh={recog_tresh}_det_thresh={det_thresh}'

faces_path = new_output_dir_path.parent / 'who_faces'
new_output_dir_path.mkdir(exist_ok=True, parents=True)
print(f'save to {new_output_dir_path}')

all_persons = persons_list_from_csv(PARENT_DIR / 'src/n=10413_native.csv')

show = False
save = True
if __name__ == '__main__':
    DATASET_DIRS = [
        '/home/vid/Downloads/datasets/face_crop_norm_dataset/datasetv2/good/',
    ]
    imgs = []
    for dir in DATASET_DIRS:
        for format in ['jpg', 'png', 'jpeg']:
            imgs.extend(Path(dir).glob(f'**/*.{format}'))
    # random.seed(2)
    # random.shuffle(imgs)
    # imgs = imgs[:200]

    p_bar = tqdm(sorted(imgs), colour='green', leave=False)
    for img_idx, img_path in enumerate(p_bar):
        p_bar.set_description(f'{img_path}')
        crop_face = cv2.imread(str(img_path))

        embedding = recognator.get(crop_face)
        dists = [cdist(embedding, person.embedding, metric='cosine')[0][0] for person in all_persons]

        who = np.argmin(dists)
        label = all_persons[who].label
        min_dist = round(dists[who], 4)
        title = f'{label}_{min_dist}'

        etalon_img = cv2.imread(all_persons[who].path)
        together = np.concatenate([crop_face, etalon_img], axis=1)

        if show:
            plt_show_img(together, swapRB=True, title=title)

        if save:
            new_path = faces_path / label / f'{img_path.stem}_{min_dist}.jpg'  # len(dirs) == len(persons)
            new_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(new_path), together)
