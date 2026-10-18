"""Генератор моментов-кандидатов по детекциям людей и предметов.

Идея: детектор не решает «нарушение или нет». Он находит МОМЕНТЫ, где что-то меняется
(кассир встал/ушёл, кто-то вошёл, у стола два человека, появился телефон/стакан/сумка).
Решение по каждому моменту принимает VLM по кадрам до/во время/после (vlm.py).
Без VLM работают только простые предметные правила.
"""
from dataclasses import dataclass

import numpy as np

PERSON_CONF = 0.45
OBJ_CONF = {"cell phone": 0.35, "cup": 0.35, "bottle": 0.35, "wine glass": 0.35,
            "handbag": 0.35, "backpack": 0.35, "suitcase": 0.35}


@dataclass
class Moment:
    t: float
    kind: str        # leave / enter / crowd / object:<cls>
    score: float
    info: str = ""


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def persons(frame_dets, H):
    """Люди в комнате. Отбрасываем отражения и людей за стеклом: маленькие и далеко (низ рамки высоко)."""
    out = []
    for name, conf, b in frame_dets:
        if name != "person" or conf < PERSON_CONF:
            continue
        h = b[3] - b[1]
        if b[3] < 0.30 * H or h < 0.12 * H:
            continue
        out.append(b)
    return out


def smooth_counts(ts, counts, win=3):
    """Медианное сглаживание числа людей (детектор иногда мигает)."""
    c = np.array(counts, float)
    out = c.copy()
    for i in range(len(c)):
        out[i] = np.median(c[max(0, i - win): i + win + 1])
    return out


def find_moments(timeline, H, W):
    """timeline: list of (t, dets) с dets = [(cls, conf, box), ...]."""
    if not timeline:
        return []
    ts = [t for t, _ in timeline]
    P = [persons(d, H) for _, d in timeline]
    counts = smooth_counts(ts, [len(p) for p in P])
    moments = []

    # 1. Изменение числа людей: вход (рост) и уход (падение)
    for i in range(1, len(counts)):
        if counts[i] > counts[i - 1]:
            moments.append(Moment(ts[i], "enter", 0.6, f"людей {int(counts[i-1])}→{int(counts[i])}"))
        elif counts[i] < counts[i - 1]:
            moments.append(Moment(ts[i], "leave", 0.6, f"людей {int(counts[i-1])}→{int(counts[i])}"))

    # 2. Кассир встал и отошёл: сильное смещение «главного» человека от его начальной позиции
    anchor = None
    for p in P[: max(1, len(P) // 6)]:
        if p:
            anchor = _center(max(p, key=lambda b: b[3]))  # ближайший к камере в начале
            break
    if anchor is not None:
        away_prev = False
        for t, p in zip(ts, P):
            d = min((np.hypot(*(np.subtract(_center(b), anchor))) for b in p), default=1e9)
            away = d > 0.10 * W
            if away and not away_prev:
                moments.append(Moment(t, "leave", 0.5, "человек у рабочего места сместился"))
            away_prev = away

    # 3. Трое и больше людей у стола или двое там, где обычно один
    for i in range(len(counts)):
        if counts[i] >= 3 and (i == 0 or counts[i - 1] < 3):
            moments.append(Moment(ts[i], "crowd", 0.6, f"в кадре {int(counts[i])} человек"))

    # 4. Предметы: телефон, напиток, сумка (держатся ≥1 с)
    for cls, thr in OBJ_CONF.items():
        seen = [any(n == cls and c >= thr for n, c, _ in d) for _, d in timeline]
        run = 0
        for i, s in enumerate(seen):
            run = run + 1 if s else 0
            if run == 3:
                moments.append(Moment(ts[i - 2], "object:" + cls, 0.5, cls))

    return merge(moments)


def merge(moments, gap=1.5):
    moments = sorted(moments, key=lambda m: m.t)
    out = []
    for m in moments:
        if out and m.t - out[-1].t <= gap:
            if m.kind not in out[-1].kind:
                out[-1].kind += "+" + m.kind
                out[-1].info += "; " + m.info
            out[-1].score = max(out[-1].score, m.score)
            continue
        out.append(m)
    return out
