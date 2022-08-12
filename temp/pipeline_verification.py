import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.constants import PARENT_DIR, det_thresh, recog_tresh
from src.utils import persons_list_from_csv, detector, Person, get_random_color

new_output_dir_path = PARENT_DIR / 'temp' / f'1208_validation_queue'

new_output_dir_path.mkdir(exist_ok=True, parents=True)
print(f'save to {new_output_dir_path}')

# all_persons = persons_list_from_csv(PARENT_DIR / 'src/n=10413_native.csv')
emb0 = np.ones(shape=(1, 512))
img0 = np.ones(shape=(112, 112, 3))
all_persons = [Person(full_img=img0, label='zero', color=get_random_color(), embedding=emb0)]


def _cv2_add_title(img, title, color, filled=True,
                   text_pos=(5, 40),
                   font=cv2.FONT_HERSHEY_COMPLEX, font_scale=0.7, thickness=1):
    img = img.copy()
    text_pos_x, text_pos_y = text_pos
    if filled:
        (text_h, text_w), _ = cv2.getTextSize(title, font, font_scale, thickness)
        cv2.rectangle(img,
                      (text_pos_x, text_pos_y - text_w - 1),
                      (text_pos_x + text_h, text_pos_y + 4),
                      color, -1)
        cv2.putText(img, title, (text_pos_x, text_pos_y), font, font_scale, (255, 255, 255), thickness)
    else:
        cv2.putText(img, title, (text_pos_x, text_pos_y), font, font_scale, color, thickness)
    return img


show = False  # for easy debugging
if __name__ == '__main__':
    DATASET_DIRS = [
        '/home/vid/Downloads/datasets/queue',
        # '/home/vid/hdd/projects/PycharmProjects/FACEID_VMX/temp/video/faceimg',
    ]
    imgs = []
    for dir in DATASET_DIRS:
        for format in ['jpg', 'png', 'jpeg']:
            imgs.extend(Path(dir).glob(f'**/*.{format}'))
    # random.seed(2)
    # random.shuffle(imgs)
    # imgs = imgs[:200]
    person_idx = 0
    p_bar = tqdm(sorted(imgs), colour='green', leave=False)
    for img_idx, img_path in enumerate(p_bar):
        p_bar.set_description(f'{img_path}')
        img = cv2.imread(str(img_path))
        faces = detector.get(img,
                             use_roi=(35, 0, 20, 10),  # top, bottom, left, right
                             min_face_size=(45, 45),
                             )
        if faces:  # if not empty
            for face in faces:
                unknown = Person(full_img=img, face=face, change_brightness=False, show=show)
                near_dist = unknown.get_label(all_persons,
                                              threshold=recog_tresh,
                                              # turn_bias=3,
                                              limits=(100, 75),
                                              use_nn=True,
                                              show=show,
                                              )

                message = None
                # '''    # for adding to etalons (for reidentification)
                if unknown.label == 'Unknown' and unknown.turn >= 0:
                    unknown.label = f'person_{person_idx}'
                    person_idx += 1
                    unknown.color = get_random_color()
                    all_persons.append(unknown)  # add person for verification
                    message = f'appended "{unknown.label}"!'
                elif unknown.label != 'Unknown':
                    face_box, etalon_box = unknown.face.bbox, unknown.etalon_face.bbox
                    face_size = [int(face_box[2] - face_box[0]), int(face_box[3] - face_box[1])]
                    etalon_size = [int(etalon_box[2] - etalon_box[0]), int(etalon_box[3] - etalon_box[1])]
                    if unknown.turn > unknown.etalon_turn and face_size[0] > etalon_size[0] and face_size[0] > \
                            etalon_size[0]:
                        all_persons[unknown.etalon_who] = unknown
                        message = f'change etalon for "{unknown.label}"! ({img_path.stem})'
                # '''

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

        if message:
            dimg = _cv2_add_title(dimg, message, color=face.color)
        cv2.imwrite(str(new_path), dimg)
        if img_idx > 30:
            exit()
