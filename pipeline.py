import cv2
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
# import albumentations as A
import matplotlib.pyplot as plt
from src.constants import PARENT_DIR, LANDMARKS_COLORS, det_thresh, det_size, recog_tresh, bright_etalon
# from src.utils import persons_list_from_csv
from src.utils import persons_list_from_csv, detector, Person

new_output_dir_path = PARENT_DIR / 'temp' / f'office_turns_0308_recog_tresh={recog_tresh}_det_thresh={det_thresh}_{det_size}'

new_output_dir_path.mkdir(exist_ok=True, parents=True)
print(f'save to {new_output_dir_path}')

all_persons = persons_list_from_csv(PARENT_DIR / 'src/n=10413_native.csv')

img_path = '/home/vid/Downloads/datasets/office_turns/2022-07-19 10:22:18.946607.jpg'
img = cv2.imread(img_path)  # (1080, 1920, 3)
print(img.shape)

faces = detector.get(img, use_roi=None, min_face_size=None)
for face in faces:
    unknown = Person(full_img=img, face=face, change_brightness=False, show=False)

# dimg = detector.draw_on_early(img, faces, plot_roi=True, plot_crop_face=True, plot_etalon=True, show=True)
