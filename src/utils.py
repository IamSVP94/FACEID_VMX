import cv2
import numpy as np
import pandas as pd
import requests
from scipy.spatial.distance import cdist
from pathlib import Path
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
import onnxruntime as ort
from src.constants import PARENT_DIR, bright_etalon, LANDMARKS_COLORS
from src.detector import Detector_cv2
from src.face_align import estimate_norm
from src.recognator import Recognator_cv2

mpl.rcParams['figure.dpi'] = 200  # plot quality
mpl.rcParams['figure.subplot.left'] = 0.01
mpl.rcParams['figure.subplot.right'] = 1

frame_selector_model_path = '/home/vid/hdd/projects/PycharmProjects/insightface/models/ConvNext_selector_softmaxv2.onnx'
frame_selector_model = ort.InferenceSession(frame_selector_model_path, providers=['CUDAExecutionProvider'])
frame_selector_model_input_name = frame_selector_model.get_inputs()[0].name

detector = Detector_cv2(PARENT_DIR / 'models/detection/det_1280_1280.onnx')
recognator = Recognator_cv2(PARENT_DIR / 'models/recognition/IResNet100l.onnx')


class Person:
    def __init__(self, path=None,
                 full_img=None,
                 face=None,
                 embedding=None,
                 label='Unknown',
                 color=(255, 0, 0),
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
            plt.imshow(concat)
            plt.title(f'"{nose_inside}" nose_inside face')
            plt.show()
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
        self.etalon_path = persons[who].path
        self.etalon_crop = persons[who].crop_face
        self.turn = self._get_turn(limits=limits, bias=turn_bias, show=show)
        if use_nn and self.turn >= 0:
            img_for_selector = preprocess_input(self.crop_face, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            selector_out = frame_selector_model.run(None, {frame_selector_model_input_name: img_for_selector})[0]
            good_frame = np.argmax(selector_out)  # bad = 0, good = 1
            if good_frame == 0:  # if "bad"
                self.turn = -99
        if dists[who] < threshold and self.turn + turn_bias >= 0:
            self.label = persons[who].label
            self.color = persons[who].color
            # self.etalon_path = persons[who].path
        if show:
            plt.imshow(cv2.cvtColor(self.crop_face, cv2.COLOR_BGR2RGB))
            plt.title(f'"{self.label}": turn={self.turn} score={min_dist} (treshold={threshold})')
            plt.show()
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
    randomcolor = (random.randint(50, 200), random.randint(50, 200), random.randint(0, 150))
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
            plt.imshow(img)
            plt.show()
        imgs.append(img)
    return imgs


def brightness_changer(img, etalon=None, diff=None, show=False):  # etalon=150
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    orig_br = int(np.mean(v))
    if etalon:
        value = etalon - orig_br
        v = cv2.add(v, value)
        v[v > 255] = 255
        v[v < 0] = 0
        hsv = cv2.merge((h, s, v))
    if diff:
        v = cv2.add(v, diff)
        v[v > 255] = 255
        v[v < 0] = 0
        hsv = cv2.merge((h, s, v))
    final_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if show:
        vis = np.concatenate((img, final_img), axis=1)
        plt.imshow(vis)
        plt.title(f'before {orig_br}:after ~{etalon}')
        plt.show()
    return final_img


def get_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return int(np.mean(v))


def preprocess_input(img, mean=None, std=None, input_space="RGB", size=(112, 112)):
    max_pixel_value = 255.0
    if input_space == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resizeimg = cv2.resize(img, size)

    img = resizeimg.astype(np.float32)
    if mean is not None:
        mean = np.array(mean, dtype=np.float32)
        mean *= max_pixel_value
        img -= mean

    if std is not None:
        std = np.array(std, dtype=np.float32)
        std *= max_pixel_value

        denominator = np.reciprocal(std, dtype=np.float32)
        img *= denominator

    img = np.moveaxis(img, -1, 0)
    img = img[np.newaxis, :, :, :]
    return img
