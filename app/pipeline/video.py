"""Чтение видео: равномерная выборка кадров и кодирование кадров-доказательств."""
import base64
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoInfo:
    fps: float
    n_frames: int
    duration: float
    width: int
    height: int


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видео")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoInfo(fps=fps, n_frames=n, duration=n / fps if fps else 0.0, width=w, height=h)


def iter_frames(path: str, sample_fps: float):
    """Отдаёт (время_сек, кадр BGR) с частотой sample_fps.
    Читаем последовательно (grab без декодирования лишних кадров) — это быстрее, чем seek."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                yield idx / fps, frame
        idx += 1
    cap.release()


def frame_at(path: str, t: float):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def to_jpeg_b64(frame: np.ndarray, max_side: int = 720, quality: int = 85) -> str:
    h, w = frame.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")
