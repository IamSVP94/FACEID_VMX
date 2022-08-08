import cv2
from src.constants import PARENT_DIR, det_thresh, det_size, recog_tresh
from src.utils import persons_list_from_csv, detector, Person

new_output_dir_path = PARENT_DIR / 'temp' / f'office_turns_0308_recog_tresh={recog_tresh}_det_thresh={det_thresh}_{det_size}'

new_output_dir_path.mkdir(exist_ok=True, parents=True)
print(f'save to {new_output_dir_path}')

all_persons = persons_list_from_csv(PARENT_DIR / 'src/n=10413_native.csv')

img_path = '/home/vid/hdd/projects/PycharmProjects/FACEID_VMX/temp/2022-07-19 10:22:19.431605.jpg'
# img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
img = cv2.imread(str(img_path))

faces = detector.get(img, use_roi=None, min_face_size=None)
for face in faces:
    unknown = Person(full_img=img, face=face, change_brightness=False, show=False)
    near_dist = unknown.get_label(all_persons, threshold=recog_tresh,
                                  turn_bias=3, limits=(100, 75), use_nn=False,
                                  show=False,
                                  )
    # face.brightness = unknown.brightness
    face.turn = round(unknown.turn, 1)
    face.crop_face = unknown.crop_face

    face.label = unknown.label
    face.color = unknown.color
    face.rec_score = round(near_dist, 2)

    face.etalon_path = unknown.etalon_path
    face.etalon_crop = unknown.etalon_crop

dimg = detector.draw_on(img, faces, plot_roi=True, plot_crop_face=False, plot_etalon=False, show=True)
