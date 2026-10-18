"""Детекторы объектов. Этап 1: заглушка. Этап 2: реальная модель (RT-DETR / RTMDet + поза).

Контракт: detect(frame) -> list[Detection]. Классы, которые нам нужны:
person, cell_phone, handbag/backpack, cup/bottle, laptop, banknotes, cash_bag, safe_door_open.
"""
from dataclasses import dataclass, field

from .. import config


@dataclass
class Detection:
    cls: str
    conf: float
    box: tuple  # x1, y1, x2, y2 в пикселях
    track_id: int | None = None
    extra: dict = field(default_factory=dict)


class NullDetector:
    name = "none"

    def detect(self, frame):
        return []


def get_detector():
    backend = config.DETECTOR_BACKEND
    if backend == "none":
        return NullDetector()
    raise ValueError(f"Неизвестный DETECTOR_BACKEND={backend}")
