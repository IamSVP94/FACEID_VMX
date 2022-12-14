import cv2
from tqdm import tqdm
from pathlib import Path
from src.constants import PARENT_DIR, det_thresh, recog_tresh
from src.utils import persons_list_from_csv, detector, Person

new_output_dir_path = PARENT_DIR / 'temp' / f'recog_tresh={recog_tresh}_det_thresh={det_thresh}'

new_output_dir_path.mkdir(exist_ok=True, parents=True)
print(f'save to {new_output_dir_path}')

all_persons = persons_list_from_csv(PARENT_DIR / 'src/n=10413_native.csv')

show = False  # for easy debugging
if __name__ == '__main__':
    DATASET_DIRS = [
        '/home/vid/hdd/datasets/office_turns',
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
        img = cv2.imread(str(img_path))
        faces = detector.get(img,
                             # use_roi=(45, 0, 20, 20),  # top, bottom, left, right
                             min_face_size=(100, 100),
                             )
        if faces:  # if not empty
            for face in tqdm(faces):
                unknown = Person(full_img=img, face=face, change_brightness=False, show=True)
                near_dist = unknown.get_label(all_persons,
                                              threshold=recog_tresh,
                                              turn_bias=3, limits=(100, 100),
                                              use_nn=False,
                                              show=True,
                                              )
                exit()
                # face.brightness = unknown.brightness
                face.turn = round(unknown.turn, 1)
                face.crop_face = unknown.crop_face

                face.label = unknown.label
                face.color = unknown.color
                face.rec_score = round(near_dist, 2)

                face.etalon_path = unknown.etalon_path
                face.etalon_crop = unknown.etalon_crop

            dimg = detector.draw_on(img, faces, plot_roi=True, plot_crop_face=True, plot_etalon=True, show=show)
            new_suffix = f'{[(face.label, face.rec_score) for face in faces]}.jpg'
        else:
            dimg = img
            new_suffix = f"[('empty')].jpg"

        new_path = new_output_dir_path / f'{img_path.stem}_{new_suffix}'
        cv2.imwrite(str(new_path), dimg)
