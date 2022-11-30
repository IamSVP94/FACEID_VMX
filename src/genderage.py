import random
from pathlib import Path

import cv2
import numpy as np
import onnx
from cv2 import dnn, cuda
import onnxruntime as ort
from tqdm import tqdm


# 0 - bad, 1 - good


class GenderAge:
    def __init__(self, path, device='cuda'):
        self.path = str(path)
        self.device = device

    def _run(self, img, input_size=None):
        '''
        Selector_ort or Selector_cv2 method
        :param img:
        :param input_size:
        :return:
        '''
        pass

    @staticmethod
    def _get_blob(img, std=1, mean=0, input_size=(96, 96), swapRB=True):
        if input_size is None:
            input_size = img.shape[:2]
        blob = cv2.dnn.blobFromImage(image=img,
                                     scalefactor=1.0 / std,
                                     size=input_size,
                                     mean=(mean, mean, mean),
                                     swapRB=swapRB,
                                     crop=None, ddepth=None)
        return blob

    def get(self, img, show=False):
        blob = self._get_blob(img)
        output = self._run(blob)
        gender = np.argmax(output[:2])
        age = int(np.round(output[2] * 100))
        if show:
            from src.utils import plt_show_img
            plt_show_img(img, swapRB=True, title=f'gender={gender}, age={age}')
        return gender, age


class GenderAge_ort(GenderAge):
    def __init__(self, *args, **kwargs):
        super(GenderAge_ort, self).__init__(*args, **kwargs)
        providers = ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(self.path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _run(self, blob, input_size=None):
        results = self.session.run(self.output_names, {self.input_name: blob})
        return results[0][0]


class GenderAge_cv2(GenderAge):
    def __init__(self, *args, **kwargs):
        super(GenderAge_cv2, self).__init__(*args, **kwargs)
        net = cv2.dnn.readNetFromONNX(str(self.path))
        self.output_names = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()]
        if self.device == 'cuda' and cv2.cuda.getCudaEnabledDeviceCount():
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)  # fp16 here
        self.net = net

    def _run(self, blob, input_size=None):
        self.net.setInput(blob)
        results = self.net.forward(self.output_names)[0]
        return results
