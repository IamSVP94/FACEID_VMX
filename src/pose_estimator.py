import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2 import dnn, cuda
import math
import onnxruntime as ort

from src import face_align


class Poser:
    def __init__(self, path, device='cuda'):
        self.path = path
        self.device = device
        self.lmk_dim = 3
        self.lmk_num = 68
        with open('/home/vid/hdd/projects/PycharmProjects/FACEID_VMX/models/pose/meanshape_68.pkl', 'rb') as f:
            self.mean_lmk = pickle.load(f)

    def _run(self, img, input_size=None):
        '''
        Poser_ort or Poser_cv2 method
        :param img:
        :param input_size:
        :return:
        '''
        pass

    @staticmethod
    def _get_blob(img, std=1.0, mean=0.0, input_size=(192, 192), swapRB=True):
        if input_size is None:
            input_size = img.shape[:2]
        blob = cv2.dnn.blobFromImage(image=img,
                                     scalefactor=1.0 / std,
                                     size=input_size,
                                     mean=(mean, mean, mean),
                                     swapRB=swapRB,
                                     crop=None, ddepth=None)
        return blob

    def get(self, img, face, show=False):
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.input_size[0] / (max(w, h) * 1.5)
        aimg, M = face_align.transform(img, center, self.input_size[0], _scale, rotate)

        input_size = tuple(aimg.shape[0:2][::-1])  # why this? cause _scale?
        blob = self._get_blob(aimg, input_size=input_size)
        pose = self._run(blob)[0]
        if pose.shape[0] >= 3000:
            pose = pose.reshape((-1, 3))
        else:
            pose = pose.reshape((-1, 2))
        if self.lmk_num < pose.shape[0]:
            pose = pose[self.lmk_num * -1:, :]
        pose[:, 0:2] += 1
        pose[:, 0:2] *= (self.input_size[0] // 2)
        if pose.shape[1] == 3:
            pose[:, 2] *= (self.input_size[0] // 2)

        IM = cv2.invertAffineTransform(M)
        pose = face_align.trans_points(pose, IM)
        P = face_align.estimate_affine_matrix_3d23d(self.mean_lmk, pose)
        s, R, t = face_align.P2sRt(P)
        rx, ry, rz = face_align.matrix2angle(R)
        pose = np.array([rx, ry, rz], dtype=np.float32)
        if show:
            from src.utils import plt_show_img
            plt_show_img(img, swapRB=True, title=f'Pitch Yaw Roll: {str(pose)}')
        return pose


class Poser_ort(Poser):
    def __init__(self, *args, **kwargs):
        super(Poser_ort, self).__init__(*args, **kwargs)
        providers = ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(self.path), providers=providers)

        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.input_shape = input_cfg.shape
        self.input_size = tuple(self.input_shape[2:4][::-1])

        self.output_names = [o.name for o in self.session.get_outputs()]

    def _run(self, blob, input_size=None):
        results = self.session.run(self.output_names, {self.input_name: blob})[0]
        return results


class Poser_cv2(Poser):
    def __init__(self, *args, **kwargs):
        super(Poser_cv2, self).__init__(*args, **kwargs)
        net = cv2.dnn.readNetFromONNX(str(self.path))
        self.output_names = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()]
        self.input_size = (192, 192)
        if self.device == 'cuda' and cv2.cuda.getCudaEnabledDeviceCount():
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)  # fp16 here
        self.net = net

    def _run(self, blob, input_size=None):
        self.net.setInput(blob)
        results = self.net.forward(self.output_names)[0]
        return results
