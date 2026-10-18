"""Просмотр видео окнами через VLM.

Каждое окно = 4 кадра за ~3 секунды, склеенные в одну картинку с подписями 1-4.
Этап A: VLM отвечает, какие нарушения из каталога видны и на каком кадре.
Этап B: для найденного — короткое описание того, что видно (по-русски).
"""
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Коды для модели -> type_id реестра
CHECKLIST = [
    ("A", "unlocked_pc", "the cashier walks away from the desk while the laptop screen still shows an open program (not a lock screen)"),
    ("B", "valuables_unattended", "the cashier walks away from the desk leaving banknotes, a cash bag, documents or other valuables on the desk"),
    ("C", "safe_open", "the safe/metal cabinet door is left open while the cashier is away from it or sitting at the desk"),
    ("D", "door_open", "the cashier enters the room and does not close the glass door behind her"),
    ("E", "personal_items", "the cashier holds or uses a mobile phone, or puts a personal bag/coat on the work desk"),
    ("F", "food_drink", "the cashier eats or drinks (cup, bottle, food) at the desk"),
    ("G", "outsider_in_cash_zone", "a second person comes behind the desk to the cashier's side / into the cashier's working area"),
    ("H", "collector_not_verified", "a cash collector hands over a green cash bag and leaves without showing any document or the bag being checked"),
    ("I", "second_client", "a second visitor approaches the desk while another client is still being served, or a new client sits down while the previous client's money/documents are still on the desk"),
    ("J", "count_without_client", "the client walks away while the cashier is still counting his money (money still in the counting machine)"),
    ("K", "suspicious_cash_handling", "the cashier hides banknotes in clothing, under the desk or keeps part of the money instead of giving it to the client"),
]
CODE2TYPE = {c: t for c, t, _ in CHECKLIST}

PROMPT_A = (
    "You are checking CCTV of a bank cashier's workplace. The image shows 4 consecutive frames "
    "(labelled 1-4, about 1 second apart). The cashier sits at a desk with a laptop, a banknote "
    "counting machine and a grey safe; clients sit across the desk.\n"
    "Which of these violations are clearly visible in these frames?\n"
    + "\n".join(f"{c}) {d}" for c, _, d in CHECKLIST)
    + "\nNormal work (serving a client, counting money, typing, a client sitting down, the cashier "
    "locking the laptop and the safe before leaving) is NOT a violation.\n"
    "Answer in exactly this format and nothing else:\n"
    "VIOLATIONS: <letters separated by commas, or NONE>\nFRAME: <1-4, the frame where it is best seen>"
)

PROMPT_B = (
    "This is CCTV of a bank cashier's workplace, 4 frames labelled 1-4. A possible violation was flagged: "
    "\"{what}\". In 1-2 short sentences in Russian, describe only what is actually visible "
    "(who does what, which objects). Do not guess about things that are not visible. "
    "If on closer look this is clearly normal work, answer only: НЕТ"
)


CROP_BOTTOM = 0.75  # нижняя часть кадра обычно занята передним столом


def strip_image(frames, ts, tile_h=480):
    tiles = []
    for i, (f, t) in enumerate(zip(frames, ts)):
        f = f[: int(f.shape[0] * CROP_BOTTOM)]
        h, w = f.shape[:2]
        s = tile_h / h
        tile = cv2.resize(f, (int(w * s), tile_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (110, 34), (0, 0, 0), -1)
        cv2.putText(tile, f"{i + 1}  {t:.0f}s", (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        tiles.append(tile)
    return np.hstack(tiles)


def parse_a(text):
    m = re.search(r"VIOLATIONS\s*:\s*([^\n]*)", text, re.I)
    codes = []
    if m and "NONE" not in m.group(1).upper():
        codes = [c for c in re.findall(r"\b([A-K])\b", m.group(1).upper())]
    f = re.search(r"FRAME\s*:\s*([1-4])", text, re.I)
    return codes, (int(f.group(1)) if f else 2)


def scan(video_path, vlm, info, deadline, stride=3.0, span=3.0, progress=None):
    """Возвращает список находок: dict(type_id, t, ts(список кадров), desc, raw)."""
    import time
    cap = cv2.VideoCapture(video_path)

    def grab(t):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, min(t, info.duration - 0.05)) * 1000)
        ok, fr = cap.read()
        return fr if ok else None

    tmp = Path(tempfile.mkdtemp(prefix="scan_"))
    windows = []
    t = 0.0
    while t < max(info.duration - 0.5, 0.1):
        windows.append(t)
        t += stride
    hits, log = [], []
    for wi, t0 in enumerate(windows):
        if time.time() > deadline:
            log.append({"t": t0, "skipped": "deadline"})
            break
        ts = [t0 + span * k / 3 for k in range(4)]
        frames = [grab(x) for x in ts]
        if any(f is None for f in frames):
            ts = [x for x, f in zip(ts, frames) if f is not None]
            frames = [f for f in frames if f is not None]
            if not frames:
                continue
        img = strip_image(frames, ts)
        p = tmp / f"w{wi:03d}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        raw = vlm.ask([str(p)], PROMPT_A, max_tokens=24)
        codes, fi = parse_a(raw)
        log.append({"t": round(t0, 1), "raw": raw})
        for c in codes:
            hits.append({"type_id": CODE2TYPE[c], "code": c, "t": ts[min(fi, len(ts)) - 1],
                         "ts": ts, "img": str(p), "win": wi})
        if progress:
            progress(0.9 * (wi + 1) / len(windows))
    cap.release()
    return merge_hits(hits), log


def merge_hits(hits, gap=4.0):
    """Склеиваем соседние окна с одним типом в одно событие (берём первое окно)."""
    hits = sorted(hits, key=lambda h: (h["type_id"], h["t"]))
    out = []
    for h in hits:
        if out and out[-1]["type_id"] == h["type_id"] and h["t"] - out[-1]["t_last"] <= gap:
            out[-1]["t_last"] = h["t"]
            out[-1]["n"] += 1
            continue
        out.append(dict(h, t_last=h["t"], n=1))
    return sorted(out, key=lambda h: h["t"])


def describe(vlm, hit):
    what = next(d for c, _, d in CHECKLIST if c == hit["code"])
    text = vlm.ask([hit["img"]], PROMPT_B.format(what=what), max_tokens=90)
    return text.strip()
