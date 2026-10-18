"""Детектор YOLOX (Apache 2.0) через onnxruntime. Без PyTorch, работает на Mac/Windows/Linux.
Веса: https://github.com/Megvii-BaseDetection/YOLOX/releases (yolox_s.onnx / yolox_m.onnx)."""
from pathlib import Path

import cv2
import numpy as np

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"]

# Нужные нам классы (остальные выбрасываем)
KEEP = {"person", "cell phone", "handbag", "backpack", "suitcase", "cup", "bottle", "wine glass",
        "laptop", "tv", "dining table", "chair", "book", "bowl", "sandwich", "donut", "cake", "banana",
        "apple", "keyboard", "mouse", "remote"}


class YoloxOnnx:
    def __init__(self, model_path, conf=0.25, nms=0.45, size=640):
        import onnxruntime as ort
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(str(model_path), providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.size, self.conf, self.nms = size, conf, nms
        self.name = "yolox:" + Path(model_path).stem
        # сетка для декодирования выходов
        grids, strides = [], []
        for s in (8, 16, 32):
            hs, ws = size // s, size // s
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((hs * ws, 1), s))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def _pre(self, img):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[: int(h * r), : int(w * r)] = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        return canvas.transpose(2, 0, 1)[None].astype(np.float32), r

    def raw(self, img):
        x, r = self._pre(img)
        out = self.sess.run(None, {self.inp: x})[0][0]
        out[:, :2] = (out[:, :2] + self.grids) * self.strides
        out[:, 2:4] = np.exp(out[:, 2:4]) * self.strides
        boxes = np.empty((len(out), 4), np.float32)
        boxes[:, 0] = out[:, 0] - out[:, 2] / 2
        boxes[:, 1] = out[:, 1] - out[:, 3] / 2
        boxes[:, 2] = out[:, 0] + out[:, 2] / 2
        boxes[:, 3] = out[:, 1] + out[:, 3] / 2
        boxes /= r
        scores = out[:, 4:5] * out[:, 5:]
        cls = scores.argmax(1)
        sc = scores[np.arange(len(cls)), cls]
        m = sc > self.conf
        boxes, sc, cls = boxes[m], sc[m], cls[m]
        res = []
        for c in np.unique(cls):
            name = COCO[c]
            if name not in KEEP:
                continue
            idx = np.where(cls == c)[0]
            b = boxes[idx]
            keep = cv2.dnn.NMSBoxes([[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in b],
                                    sc[idx].tolist(), self.conf, self.nms)
            for k in np.array(keep).flatten():
                res.append((name, float(sc[idx][k]), tuple(float(v) for v in b[k])))
        return res
