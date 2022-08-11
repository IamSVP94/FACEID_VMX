import datetime

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from src.constants import PARENT_DIR, det_thresh, recog_tresh
from src.utils import persons_list_from_csv, detector, Person

new_output_dir_path = PARENT_DIR / 'temp' / f'cam_office_recog_tresh={recog_tresh}_det_thresh={det_thresh}'

Path(new_output_dir_path, 'ready').mkdir(exist_ok=True, parents=True)
Path(new_output_dir_path, 'raw').mkdir(exist_ok=True, parents=True)
print(f'save to {new_output_dir_path}')

# -------------------- cam params
# CAM_IP = "10.96.0.96"
CAM_IP = "10.96.1.108"
CAM_LOGIN = "admin"
CAM_PASSWORD = "1ICenter"
cap = cv2.VideoCapture(f"rtsp://{CAM_LOGIN}:{CAM_PASSWORD}@{CAM_IP}")
# cap.set(cv2.CAP_PROP_FPS, 20)

# ==================== /cam params

all_persons = persons_list_from_csv(PARENT_DIR / 'src/n=10413_native.csv')

show = False  # for easy debugging
if __name__ == '__main__':
    every_frame_count = 0
    with tqdm() as pbar:
        while (True):
            ret, img = cap.read()
            if every_frame_count % 13 == 0:
                try:
                    faces = detector.get(
                        img,
                        # use_roi=(45, 0, 20, 20),  # top, bottom, left, right
                        # min_face_size=(45, 45),
                    )
                except AttributeError as e:  # bug!
                    cap = cv2.VideoCapture(f"rtsp://{CAM_LOGIN}:{CAM_PASSWORD}@{CAM_IP}")
                    print(e)
                    continue
                if faces:  # if not empty
                    for face in faces:
                        unknown = Person(full_img=img, face=face, change_brightness=False, show=show)
                        near_dist = unknown.get_label(all_persons,
                                                      threshold=recog_tresh,
                                                      turn_bias=3, limits=(100, 75),
                                                      use_nn=True,
                                                      show=show,
                                                      )
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
                    # continue  # not save empy frames

                pbar.set_description(str(datetime.datetime.now()))
                new_path = new_output_dir_path / 'ready' / f'{every_frame_count}_{datetime.datetime.now()}_{new_suffix}'
                cv2.imwrite(str(new_path), dimg)  # save ready frames
                new_path = new_output_dir_path / 'raw' / f'{every_frame_count}_{datetime.datetime.now()}_{new_suffix}'
                cv2.imwrite(str(new_path), img)  # save raw frames
            every_frame_count += 1
        cap.release()
        cv2.destroyAllWindows()
