import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.constants import PARENT_DIR, det_thresh, recog_tresh
from src.utils import persons_list_from_csv, detector, Person, get_random_color

currentday = datetime.date.today()
new_output_dir_path = f'/home/vid/Downloads/datasets/SCUD/{currentday}'

Path(new_output_dir_path, 'ready').mkdir(exist_ok=True, parents=True)
Path(new_output_dir_path, 'raw').mkdir(exist_ok=True, parents=True)
Path(new_output_dir_path, '_etalons').mkdir(exist_ok=True, parents=True)
print(f'save to {new_output_dir_path}')

# -------------------- cam params
# CAM_IP = "10.96.0.96"
# CAM_IP = "10.96.1.108"
CAM_IP = "10.96.0.64"
CAM_LOGIN = "admin"
CAM_PASSWORD = "1ICenter"
cap = cv2.VideoCapture(f"rtsp://{CAM_LOGIN}:{CAM_PASSWORD}@{CAM_IP}")


# ==================== /cam params


def _cv2_add_title(img, title, color, filled=True, text_pos=(5, 40),
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


cam_fps = cap.get(cv2.CAP_PROP_FPS)
new_fps = 2
print(f'fps = {cam_fps} -> {new_fps}')

# all_persons = persons_list_from_csv(PARENT_DIR / 'src/n=10413_native.csv')
all_persons = [Person(
    full_img=np.ones(shape=(112, 112, 3)),
    label='noperson',
    color=get_random_color(),
    embedding=np.ones(shape=(1, 512)),
)]

show = False  # for easy debugging
if __name__ == '__main__':
    every_frame_count = 0
    person_idx = 0
    messages_idx = 0
    with tqdm() as pbar:
        while (True):
            pbar.set_description(str(datetime.datetime.now()))
            ret, img = cap.read()
            # frame_n = every_frame_count % int(max(cam_fps / new_fps, 1))
            # if frame_n == 0:
            if 1 == 1:  # fake
                try:
                    faces = detector.get(
                        img,
                        use_roi=(25, 0, 0, 15),  # top, bottom, left, right
                        min_face_size=(55, 55),
                    )
                except AttributeError as e:  # bug!
                    cap = cv2.VideoCapture(f"rtsp://{CAM_LOGIN}:{CAM_PASSWORD}@{CAM_IP}")
                    print(e)
                    # continue

                monitor_logs = {'messages': [], 'colors': []}
                if faces:  # if not empty
                    for face in faces:
                        unknown = Person(full_img=img, face=face, change_brightness=False, show=show)
                        near_dist = unknown.get_label(all_persons,
                                                      threshold=recog_tresh,
                                                      turn_bias=3, limits=(100, 75),
                                                      use_nn=True,
                                                      show=show,
                                                      )

                        # '''    # for adding to etalons (for reidentification)
                        if unknown.label == 'Unknown' and unknown.turn >= 0:
                            unknown.label = f'person_{person_idx:03d}'
                            person_idx += 1
                            unknown.color = get_random_color()
                            all_persons.append(unknown)  # add person for verification

                            new_etalon_path = Path(new_output_dir_path, '_etalons', f'{unknown.label}.jpg')
                            cv2.imwrite(str(new_etalon_path), unknown.crop_face)  # save etalon 112x112 img

                            monitor_logs['messages'].append(f'appended "{unknown.label}"! ({every_frame_count})')
                            monitor_logs['colors'].append(unknown.color)
                        # '''  # etalon replacement
                        elif unknown.label != 'Unknown':
                            face_box, etalon_box = unknown.face.bbox, unknown.etalon_face.bbox
                            face_size = [int(face_box[2] - face_box[0]), int(face_box[3] - face_box[1])]
                            etalon_size = [int(etalon_box[2] - etalon_box[0]), int(etalon_box[3] - etalon_box[1])]
                            if unknown.turn > unknown.etalon_turn and face_size[0] > etalon_size[0] and face_size[0] > \
                                    etalon_size[0]:
                                all_persons[unknown.etalon_who] = unknown

                                new_etalon_path = Path(new_output_dir_path, '_etalons', f'{unknown.label}.jpg')
                                cv2.imwrite(str(new_etalon_path), unknown.crop_face)  # save etalon 112x112 img

                                monitor_logs['messages'].append(
                                    f'change etalon for "{unknown.label}"! ({every_frame_count})')
                                monitor_logs['colors'].append(unknown.color)
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

                    start_Ox = 40
                    for i in range(len(monitor_logs['messages'])):
                        message = f'message {messages_idx:03d}: {monitor_logs["messages"][i]}'
                        print()
                        print(message)
                        color = monitor_logs['colors'][i]
                        dimg = _cv2_add_title(dimg, message, color=color, text_pos=(5, start_Ox))
                        messages_idx += 1
                        start_Ox += 20
                else:
                    dimg = img
                    new_suffix = f"[('empty')].jpg"
                    continue  # not save empty frames
                now = str(datetime.datetime.now())  # current time
                new_path = Path(new_output_dir_path, 'ready', f'{now}_{every_frame_count}_{new_suffix}')
                cv2.imwrite(str(new_path), dimg)  # save ready frames
                new_path = Path(new_output_dir_path, 'raw', f'{now}_{every_frame_count}_{new_suffix}')
                cv2.imwrite(str(new_path), img)  # save raw frames
            every_frame_count += 1
        cap.release()
    cv2.destroyAllWindows()
