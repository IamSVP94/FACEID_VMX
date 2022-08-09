import cv2
import numpy as np
from cv2 import dnn, cuda
import onnxruntime as ort
from matplotlib import pyplot as plt

from src.constants import LANDMARKS_COLORS, det_window_size


class Face(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None


class Detector:
    def __init__(self, path, det_thresh=0.7, nms_thresh=0.4, device='cuda'):
        self.path = path
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh
        self.device = device
        self.output_names = None
        self.use_roi = False
        self.color = (0, 255, 0)
        self.thickness = 1

    def _run(self, img, input_size=None):
        '''
        Detector_ort or Detector_cv2 method
        :param img:
        :param input_size:
        :return:
        '''
        pass

    @staticmethod
    def _get_blob(img, std=128.0, mean=127.5, input_size=None, swapRB=True):
        if input_size is None:
            input_size = img.shape[:2]
        blob = cv2.dnn.blobFromImage(image=img,
                                     scalefactor=1.0 / std,
                                     size=input_size,
                                     mean=(mean, mean, mean),
                                     swapRB=swapRB,
                                     crop=None, ddepth=None)
        return blob

    @staticmethod
    def _distance2bbox(points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def _distance2kps(points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = px.clamp(min=0, max=max_shape[1])
                py = py.clamp(min=0, max=max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    @staticmethod
    def _nms(dets, nms_thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def forward(self, img, threshold, input_size=None):
        blob = self._get_blob(img, input_size=input_size)
        net_outs = self._run(blob, input_size=input_size)

        _feat_stride_fpn = [8, 16, 32]  # for Feature Pyramid Network
        fmc = len(_feat_stride_fpn)
        _num_anchors = 2

        _, _, input_height, input_width = blob.shape
        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(_feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc]
            bbox_preds = bbox_preds * stride
            kps_preds = net_outs[idx + fmc * 2] * stride
            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)

            center_cache = dict()
            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                anchor_centers = np.zeros((height, width, 2), dtype=np.float32)
                for i in range(height):
                    anchor_centers[i, :, 1] = i
                for i in range(width):
                    anchor_centers[:, i, 0] = i

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                anchor_centers = np.stack([anchor_centers] * _num_anchors, axis=1).reshape((-1, 2))

                if len(center_cache) < 100:  # on the image
                    center_cache[key] = anchor_centers
            pos_inds = np.where(scores >= threshold)[0]
            bboxes = self._distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            kpss = self._distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, input_size=None):
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self._nms(pre_det, self.nms_thresh)
        det = pre_det[keep, :]
        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]
        return det, kpss

    def get(self, img, use_roi=None, det_window_size=det_window_size, min_face_size=None):
        bboxes, kpss = self.detect(img, input_size=det_window_size)
        if bboxes.shape[0] == 0:
            return []
        faces = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            xmin_box, ymin_box, xmax_box, ymax_box = bbox
            if min_face_size is not None:
                if (xmax_box - xmin_box < min_face_size[0]) or (ymax_box - ymin_box < min_face_size[1]):
                    continue  # skip small faces
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            if use_roi is not None:
                self.use_roi = True  # for correct plot_roi
                top_proc, bottom_proc, left_proc, right_proc = use_roi
                bbox_centroid_y, bbox_centroid_x = (xmin_box + xmax_box) / 2, (ymin_box + ymax_box) / 2
                orig_h, orig_w, _ = img.shape
                if top_proc + bottom_proc >= 100:
                    top_proc, bottom_proc = 0, 0
                if left_proc + right_proc >= 100:
                    left_proc, right_proc = 0, 0
                x_min_roi = max(0, int(orig_h / 100 * top_proc))  # for correct crop
                x_max_roi = min(orig_h, int(orig_h / 100 * (100 - bottom_proc)))  # for correct crop
                y_min_roi = max(0, int(orig_w / 100 * left_proc))  # for correct crop
                y_max_roi = min(orig_w, int(orig_w / 100 * (100 - right_proc)))  # for correct crop
                self.roi_points = {'x_min': x_min_roi, 'y_min': y_min_roi,
                                   'x_max': x_max_roi, 'y_max': y_max_roi}  # for draw roi in img

                if not x_min_roi <= bbox_centroid_x <= x_max_roi or not y_min_roi <= bbox_centroid_y <= y_max_roi:
                    continue  # centroid not in roi
            faces.append(face)
        return faces

    def draw_on(self, img, faces, plot_roi=False, plot_crop_face=False, plot_etalon=False, show=False):
        def _cv2_add_title(img, title, filled=True, font=cv2.FONT_HERSHEY_COMPLEX, font_scale=0.7, thickness=2):
            img = img.copy()
            text_pos_x, text_pos_y = box[0] - 1, box[1] - 4
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

        dimg = img.copy()
        if plot_roi and self.use_roi:
            dimg = cv2.rectangle(img.copy(),
                                 (self.roi_points['y_min'], self.roi_points['x_min']),
                                 (self.roi_points['y_max'], self.roi_points['x_max']),
                                 self.color, self.thickness)
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)  # xmin, ymin, xmax, ymax
            face.size = [box[2] - box[0], box[3] - box[1]]
            color = face['color'] if face.get('color') else (0, 255, 0)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 1)
            title = f'"{face.label}", ({round(float(face.det_score), 4)}, {round(float(face.rec_score), 4)}) turn={face.turn}, size={face.size}'
            dimg = _cv2_add_title(dimg, title)
        if plot_crop_face:
            crops = [face.crop_face for face in faces]
            # draw landmarsks on crops
            for crop_idx, crop in enumerate(crops):
                for idx_p, p in enumerate(faces[crop_idx].kps):
                    cv2.circle(crop, p, 1, LANDMARKS_COLORS[idx_p], 1)
            # /draw landmarsks on crops

            crops_together = self._get_coll_imgs(crops, dimg.shape)
            dimg = np.concatenate([dimg, crops_together], axis=1)

        if plot_etalon:
            etalons = []
            for face in faces:
                if face.etalon_crop is not None:
                    etalon = face.etalon_crop
                elif face.etalon_path is not None:
                    etalon = cv2.imread(str(face.etalon_path))
                else:
                    etalon = np.full(shape=(112, 112, 3), fill_value=face.color, dtype=np.uint8)  # empties
                etalons.append(etalon)

            etalons_together = self._get_coll_imgs(etalons, dimg.shape)
            dimg = np.concatenate([dimg, etalons_together], axis=1)
        if show:
            plt.imshow(cv2.cvtColor(dimg, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.show()
        return dimg

    def _get_coll_imgs(self, imgs_list, size, top=1, left=1, right=1):  # top=10, left=5, right=5
        max_w = max([i.shape[1] for i in imgs_list])
        top_one, bottom_one = 1, 1
        good_size_ready = []
        for i in imgs_list:
            curr_w = i.shape[1]
            left = int((max_w - curr_w) / 2)
            right = max_w - curr_w - left
            vis_part_img = cv2.copyMakeBorder(i, top_one, bottom_one, left, right, cv2.BORDER_CONSTANT)
            good_size_ready.append(vis_part_img)
        ready_together = np.concatenate(good_size_ready, axis=0)

        bottom = size[0] - ready_together.shape[0] - top
        ready_together = cv2.copyMakeBorder(ready_together, top, bottom, left, right, cv2.BORDER_CONSTANT)
        return ready_together


class Detector_ort(Detector):
    def __init__(self, *args, **kwargs):
        super(Detector_ort, self).__init__(*args, **kwargs)
        providers = ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(self.path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _run(self, blob, input_size=None):
        results = self.session.run(self.output_names, {self.input_name: blob})
        return results


class Detector_cv2(Detector):
    def __init__(self, *args, **kwargs):
        super(Detector_cv2, self).__init__(*args, **kwargs)
        net = cv2.dnn.readNetFromONNX(str(self.path))
        output_names = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()]  # wrong sorting
        self.output_names = [layer for layer in net.getLayerNames() if layer in output_names]
        if self.device == 'cuda' and cv2.cuda.getCudaEnabledDeviceCount():
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)  # fp16 here
        self.net = net

    def _run(self, blob, input_size=None):
        self.net.setInput(blob)
        results = self.net.forward(self.output_names)
        return results
