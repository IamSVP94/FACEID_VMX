import cv2
import numpy as np
import pandas as pd
import requests
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
from src.constants import PARENT_DIR, bright_etalon, LANDMARKS_COLORS, det_nms, det_thresh
from src.detector import Detector_cv2, Detector_ort
from src.face_align import estimate_norm
from src.recognator import Recognator_cv2, Recognator_ort

from src.selector import Selector_ort, Selector_cv2

mpl.rcParams['figure.dpi'] = 200  # plot quality
mpl.rcParams['figure.subplot.left'] = 0.01
mpl.rcParams['figure.subplot.right'] = 1

nn_device = 'cuda'
detector = Detector_ort(PARENT_DIR / 'models/detection/det_1280_1280.onnx', det_thresh=det_thresh, nms_thresh=det_nms,
                        device=nn_device)
recognator = Recognator_ort(PARENT_DIR / 'models/recognition/IResNet100l.onnx', device=nn_device)
selector = Selector_ort(PARENT_DIR / 'models/selection/ConvNext_selector_softmaxv2_R2_15082022_112x112.onnx',
                        device=nn_device)


class Person:
    def __init__(self, path=None,
                 full_img=None,
                 face=None,
                 embedding=None,
                 label='Unknown',
                 color=(0, 0, 255),
                 change_brightness=False,
                 show=False):
        self.color = color
        self.label = label
        self.path = path
        self.crop_face = None
        self.turn = None
        if full_img is not None:
            self.full_img = full_img
        if self.path is not None and full_img is None:
            # self.full_img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            self.full_img = cv2.imread(str(path))
        if embedding is None:
            if face is None:
                face = detector.get(img=self.full_img,
                                    # use_roi=(30, 10, 20, 28),  # how to change?
                                    # min_face_size=(50, 50),  # how to change?
                                    )[0]
            crop_face, face.kps = norm_crop(self.full_img, face.kps)
            if change_brightness:
                self.crop_face = brightness_changer(crop_face, etalon=bright_etalon)
            else:
                self.crop_face = crop_face
        self.embedding = embedding if embedding is not None else recognator.get(self.crop_face, show=show)
        self.face = face

    def _get_turn(self, bias=0, limits=None, show=False):
        countur = np.array([self.face.kps[1], self.face.kps[4], self.face.kps[3], self.face.kps[0]]).astype(np.float32)
        nose = self.face.kps[2]
        nose_inside = cv2.pointPolygonTest(countur, nose.astype(np.float32), measureDist=True)
        nose_color = (0, 255, 0) if nose_inside + bias >= 0 else (255, 0, 0)
        if limits is not None:
            get_middle = lambda p1, p2: [int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)]
            m_right = get_middle(self.face.kps[0], self.face.kps[3])
            m_lelf = get_middle(self.face.kps[1], self.face.kps[4])
            m_eye = get_middle(self.face.kps[0], self.face.kps[1])
            m_mouth = get_middle(self.face.kps[3], self.face.kps[4])
            centroid_face = [int((m_eye[0] + m_mouth[0]) / 2), int((m_lelf[1] + m_right[1]) / 2)]

            difX = int((m_lelf[0] - m_right[0]) / 100 * limits[0] / 2)
            difY = int((m_mouth[1] - m_eye[1]) / 100 * limits[1] / 2)  # /2 for half
            borders = dict()
            borders['x'] = {'min': centroid_face[0] - difX, 'max': centroid_face[0] + difX}
            borders['y'] = {'min': centroid_face[1] - difY, 'max': centroid_face[1] + difY}

            if nose_inside + bias >= 0 and not (
                    borders['x']['min'] <= self.face.kps[2][0] <= borders['x']['max'] and \
                    borders['y']['min'] <= self.face.kps[2][1] <= borders['y']['max']):
                nose_inside = -50
                nose_color = (239, 114, 21)
        if show:
            cimg = cv2.cvtColor(self.crop_face, cv2.COLOR_BGR2RGB)
            for idx_p, p in enumerate(self.face.kps):
                cv2.circle(cimg, p, 1, LANDMARKS_COLORS[idx_p], 1)
            src = np.zeros(cimg.shape).astype(np.uint8)

            if limits is not None:
                borders32 = np.array([
                    (borders['x']['min'], borders['y']['max']),
                    (borders['x']['min'], borders['y']['min']),
                    (borders['x']['max'], borders['y']['min']),
                    (borders['x']['max'], borders['y']['max']),
                ]).astype(np.float32)
                cv2.polylines(src, np.int32([borders32]), True, (255, 255, 0), 1)
                cv2.circle(src, centroid_face, 1, (255, 255, 0), 1)

            cv2.polylines(src, np.int32([countur]), True, 255, 1)
            cv2.circle(src, nose, 1, nose_color, 1)

            concat = np.concatenate((cimg, src.astype(int)), axis=1)
            plt_show_img(concat, title=f'"{nose_inside}" nose_inside face')
        return nose_inside

    def get_label(self, persons,
                  threshold=0.7, metric='cosine',
                  turn_bias=0, use_nn=False, limits=None,
                  show=False):
        dists = []
        for person in persons:
            dist = cdist(self.embedding, person.embedding, metric=metric)[0][0]
            dists.append(dist)
        who = np.argmin(dists)
        min_dist = round(dists[who], 5)

        self.etalon_who = who
        self.etalon_path = persons[who].path
        self.etalon_crop = persons[who].crop_face
        self.etalon_turn = persons[who].turn
        self.etalon_face = persons[who].face
        self.turn = self._get_turn(limits=limits, bias=turn_bias, show=show)
        if use_nn and self.turn >= 0:
            result = selector.get(self.crop_face, show=show)  # 0 - bad, 1 - good
            if result == 0:  # if "bad"
                self.turn = -99
        if dists[who] < threshold and self.turn + turn_bias >= 0:
            self.label = persons[who].label
            self.color = persons[who].color
            # self.etalon_path = persons[who].path
        if show:
            title = f'"{self.label}": turn={self.turn} score={min_dist} (treshold={threshold})'
            plt_show_img(self.crop_face, swapRB=True, title=title)
        return min_dist


def norm_crop(img, landmark, image_size=112, mode='arcface', change_kpss_for_crop=True):
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    if change_kpss_for_crop:
        landmark = np.array(list(map(
            lambda point: [
                M[0][0] * point[0] + M[0][1] * point[1] + M[0][2],  # change Ox
                M[1][0] * point[0] + M[1][1] * point[1] + M[1][2]  # change Oy
            ], landmark)))  # r_eye 0, l_eye 1, nose 2, r_mouth 3, l_mouth 4
    landmark = landmark.astype(np.uint8)
    return warped, landmark


def get_random_color():
    # randomcolor = (random.randint(50, 200), random.randint(50, 200), random.randint(0, 150))
    randomcolor = (random.randint(0, 150), random.randint(50, 200), random.randint(50, 200))
    return randomcolor


def persons_list_from_csv(df_path):
    df_persons = pd.read_csv(df_path, index_col=0)
    persons = []
    for label, line in df_persons.iterrows():
        img_path = line[0]
        emb = np.array([line.to_list()[1:]])
        person = Person(path=img_path, label=label, color=get_random_color(), embedding=emb)
        persons.append(person)
    return persons


def get_imgs_thispersondoesnotexist(n=1, colors='RGB', show=False):
    imgs = []
    for i in range(n):
        img_str = requests.get('https://www.thispersondoesnotexist.com/image?').content
        nparr = np.frombuffer(img_str, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if colors == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if show:
            if colors != 'RGB':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt_show_img(img)
        imgs.append(img)
    return imgs


def brightness_changer(img, etalon=None, diff=None, mode='lab', show=False):  # hsv etalon=150; lab etalon=120?
    assert mode in ['hsv', 'lab']
    assert etalon is not None or diff is not None
    if mode == 'hsv':
        cv2_mode = cv2.COLOR_BGR2HSV
        cv2_mode_invert = cv2.COLOR_HSV2BGR
    elif mode == 'lab':
        cv2_mode = cv2.COLOR_BGR2Lab
        cv2_mode_invert = cv2.COLOR_Lab2BGR
    params = cv2.split(cv2.cvtColor(img, cv2_mode))
    if mode == 'hsv':
        bright_param = params[2]
        orig_br = int(np.mean(bright_param))
    elif mode == 'lab':
        bright_param = params[0]
        orig_br = int(np.mean(bright_param))
    if etalon and diff is None:
        diff = etalon - orig_br
    bright_param = cv2.add(bright_param, diff)
    # bright_param = cv2.normalize(bright_param, None, alpha=0, beta=255)
    if mode == 'hsv':
        params = cv2.merge((params[0], params[1], bright_param))
    elif mode == 'lab':
        params = cv2.merge((bright_param, params[1], params[2]))
    final_img = cv2.cvtColor(params, cv2_mode_invert)
    if show:
        together = np.concatenate((img, final_img), axis=1)
        plt_show_img(together, title=f'before {orig_br}:after ~{etalon}', swapRB=True)
    return final_img


def get_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return int(np.mean(v))


def plt_show_img(img, swapRB: bool = False, title: str = None) -> None:
    img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy() if swapRB else img.copy()
    plt.imshow(img_show)
    if title:
        plt.title(title)
    plt.show()
