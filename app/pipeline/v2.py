"""Анализ v2: детектор находит рабочую зону и моменты, VLM отвечает на точечные вопросы
по увеличенным фрагментам кадра.

Почему так: 7B-модель плохо выбирает из длинного списка нарушений, но хорошо отвечает
«да/нет» на простой вопрос по крупному фрагменту (проверено на 22 кадрах: ошибки были
только там, где объект мелкий — их и убирает увеличение зоны).
"""
import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .. import config
from .video import frame_at, iter_frames

MODELS_DIR = config.BASE_DIR / "models"
YOLOX_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"

DET_FPS = 3.0
PERSON_CONF = 0.45

# ---------- вопросы к VLM ----------
Q_MONEY = ("This is a close-up of a bank cashier's desk. Are there banknotes, money, a cash bag, "
           "a tablet, documents or papers lying on the desk? Ignore the laptop, the phone handset and "
           "the counting machines. Answer YES or NO.")
Q_SCREEN = ("Look at the laptop screen. Is it showing a lock screen / desktop wallpaper picture (LOCKED), "
            "or an open program window with text, tables or forms (OPEN), or is it closed/off (OFF)? "
            "Answer exactly one word: LOCKED, OPEN or OFF.")
Q_SAFE = ("Look at the grey metal safe/cabinet next to the desk. Is its door standing open? Answer YES or NO.")
Q_DESK_LIST = ("Close-up of the top of a bank cashier's desk. List every separate object lying on the desk surface, "
               "in a few words each, comma-separated (for example: laptop, phone handset, counting machine, banknotes, papers).")
# слова «cash»/«money» не берём: модель так называет обычный cash drawer / money tray
DESK_ALERT = ("banknote", "bills", "bag", "pouch", "document", "paper", "tablet", "canister",
              "envelope", "folder", "wallet")
Q_PHONE = ("Close-up of a bank cashier at her desk. Is the cashier herself holding or using a mobile phone (smartphone) "
           "in her hand right now? The desk phone handset does not count. Answer YES or NO.")
Q_DRINK = ("Close-up of a bank cashier at her desk. Is the cashier drinking or eating right now, or holding a cup, glass, "
           "bottle or food in her hand? Answer YES or NO.")
Q_BAG = ("Close-up of a bank cashier's desk. Is there a personal handbag, backpack, purse or a coat/jacket lying ON the desk? "
         "The black counting machines, the laptop and a green cash-collection bag do not count. Answer YES or NO.")
Q_HABITS = ("Close-up of a bank cashier at her desk. Which of these is clearly visible: "
            "PHONE (she holds or uses a mobile phone), DRINK (she drinks or eats, a cup/bottle/food in hand or at her mouth), "
            "BAG (a personal handbag, backpack or coat on the desk). "
            "Answer with the matching words separated by commas, or NONE.")
Q_BEHIND = ("Is there a person other than the cashier standing BEHIND the desk on the cashier's side, "
            "inside the cashier's working area? Answer YES or NO.")
Q_TWO_VISITORS = ("Bank cashier CCTV. Count only the visitors (not the cashier) who are at the desk or "
                  "approaching it right now. Are there two or more visitors at the same time? Answer YES or NO.")
Q_COLLECTOR = ("Is a man carrying or handing over a green cash bag? Answer YES or NO.")
Q_COLLECTOR_DOC = ("A man is handing over a green cash bag. Is he also showing or handing over a paper document "
                   "or an ID card? Answer YES or NO.")


RU = {"banknote": "купюры", "bills": "купюры", "bag": "сумка", "pouch": "сумка", "document": "документы",
      "paper": "бумаги/документы", "tablet": "планшет", "canister": "предметы (ёмкости)", "envelope": "конверт",
      "folder": "папка", "wallet": "кошелёк"}


def ru_items(keys):
    out = []
    for k in keys:
        if RU[k] not in out:
            out.append(RU[k])
    return ", ".join(out)


def _yes(ans):
    return ans.strip().upper().startswith("YES")


# ---------- детектор ----------
_det = None


def detector():
    global _det
    if _det is None:
        from .yolox_onnx import YoloxOnnx
        MODELS_DIR.mkdir(exist_ok=True)
        p = MODELS_DIR / "yolox_s.onnx"
        if not p.exists():
            print("Скачиваю веса детектора YOLOX-s (36 МБ)...")
            urllib.request.urlretrieve(YOLOX_URL, p)
        _det = YoloxOnnx(p, conf=0.15)
    return _det


def persons(dets, H):
    out = []
    for name, conf, b in dets:
        if name == "person" and conf >= PERSON_CONF and b[3] >= 0.30 * H and (b[3] - b[1]) >= 0.12 * H:
            out.append(b)
    return out


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def inside(pt, z):
    return z[0] <= pt[0] <= z[2] and z[1] <= pt[1] <= z[3]


# ---------- рабочая зона кассира ----------
def find_zone(tl, H, W):
    """Зона = ноутбук кассира, расширенный на стол и сейф. Если ноутбук не найден — вокруг кассира."""
    laps = [b for _, d in tl for n, s, b in d if n == "laptop" and s > 0.5]
    if len(laps) >= 3:
        lb = np.median(np.array(laps), 0)
        w, h = lb[2] - lb[0], lb[3] - lb[1]
        z = [lb[0] - 3.0 * w, lb[1] - 2.0 * h, lb[2] + 1.5 * w, lb[3] + 1.6 * h]
        src = "laptop"
    else:
        lb = None
        first = [persons(d, H) for _, d in tl[: max(3, len(tl) // 5)]]
        cand = [max(p, key=lambda b: b[3]) for p in first if p]
        if not cand:
            return [0, 0, W, int(H * 0.75)], None, "full"
        pb = np.median(np.array(cand), 0)
        w, h = pb[2] - pb[0], pb[3] - pb[1]
        z = [pb[0] - 1.2 * w, pb[1] - 0.4 * h, pb[2] + 1.2 * w, pb[3] + 0.5 * h]
        src = "person"
    z = [int(max(0, z[0])), int(max(0, z[1])), int(min(W, z[2])), int(min(H, z[3]))]
    return z, lb, src


def desk_box(laptop, zone):
    if laptop is None:
        return zone
    lb = laptop
    w, h = lb[2] - lb[0], lb[3] - lb[1]
    return [lb[0] - 3.5 * w, lb[1] - 0.3 * h, lb[2] + 0.5 * w, lb[3] + 1.6 * h]


def crop(frame, box, min_w=640):
    x1, y1, x2, y2 = [int(v) for v in box]
    c = frame[max(0, y1):y2, max(0, x1):x2]
    if c.size == 0:
        return frame
    s = max(1.0, min_w / c.shape[1])
    return cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


# ---------- моменты ----------
def seat_zone(tl, H, W, laptop):
    """Место кассира: человек у ноутбука в начале видео (или ближайший к камере), слегка расширенный."""
    first = [persons(d, H) for _, d in tl[: max(4, len(tl) // 4)]]
    cand = []
    for p in first:
        if not p:
            continue
        if laptop is not None:
            lc = center(laptop)
            cand.append(min(p, key=lambda b: np.hypot(*np.subtract(center(b), lc))))
        else:
            cand.append(max(p, key=lambda b: b[3]))
    if not cand:
        return None
    pb = np.median(np.array(cand), 0)
    w, h = pb[2] - pb[0], pb[3] - pb[1]
    # 5-й элемент — рост сидящего кассира: стоящий человек выше, значит встал
    return [pb[0] - 0.45 * w, pb[1] - 0.25 * h, pb[2] + 0.45 * w, pb[3] + 0.25 * h, h]


def occupancy(tl, zone, H):
    """Кассир сидит на месте: человек в зоне места и не выше сидящего (если рост известен)."""
    hmax = zone[4] * 1.18 if len(zone) > 4 else 1e9
    return [any(inside(center(b), zone) and (b[3] - b[1]) <= hmax for b in persons(d, H)) for _, d in tl]


def runs(flags, ts, min_len=1.5):
    """Отрезки, где flags=True держится не меньше min_len секунд: [(t_start, t_end)]."""
    out, start = [], None
    for i, f in enumerate(flags + [False]):
        t = ts[i] if i < len(ts) else ts[-1] + 1
        if f and start is None:
            start = t
        if not f and start is not None:
            if t - start >= min_len:
                out.append((start, t))
            start = None
    return out


def leave_events(tl, zone, H):
    """Кассир покинул зону: занято → пусто ≥2 с."""
    ts = [t for t, _ in tl]
    occ = occupancy(tl, zone, H)
    empty = runs([not o for o in occ], ts, min_len=2.0)
    ev = []
    for a, b in empty:
        if any(o for t, o in zip(ts, occ) if t < a):  # до этого кассир был на месте
            ev.append((a, b))
    return ev


def enter_events(tl, H):
    """Число людей в комнате выросло и держится ≥1 с."""
    ts = [t for t, _ in tl]
    cnt = np.array([len(persons(d, H)) for _, d in tl], float)
    sm = np.array([np.median(cnt[max(0, i - 2): i + 3]) for i in range(len(cnt))])
    ev = []
    for i in range(1, len(sm)):
        if sm[i] > sm[i - 1]:
            ev.append((ts[i], int(sm[i])))
    return ev


# ---------- главный анализ ----------
def analyze_v2(video_path, vlm, info, deadline, progress=None):
    t0 = time.time()
    det = detector()
    tl = []
    H = W = None
    for t, f in iter_frames(video_path, DET_FPS):
        H, W = f.shape[:2]
        tl.append((t, det.raw(f)))
        if time.time() > deadline - 15:  # очередь съела время — отвечаем по тому, что успели
            break
        if progress and info.duration:
            progress(0.3 * t / info.duration)
    if not tl:
        return [], {"error": "нет кадров"}
    zone, laptop, zsrc = find_zone(tl, H, W)
    log = {"zone": zone, "zone_src": zsrc, "det_sec": round(time.time() - t0, 1), "calls": []}
    findings = []  # dict(type_id, t, ev_ts, desc, score)

    def ask(frame, box, q, tag, t, max_tokens=8):
        img = crop(frame, box, min_w=896 if tag == "desk" else 640) if box else frame
        ans = vlm.ask([_tmp(img)], q, max_tokens=max_tokens)
        log["calls"].append({"t": round(t, 1), "q": tag, "a": ans})
        return ans

    seat = seat_zone(tl, H, W, laptop)
    log["seat"] = [int(v) for v in seat[:4]] if seat else None
    seat_or_zone = seat or zone

    # 1. Кассир ушёл: что осталось
    for a, b in leave_events(tl, seat_or_zone, H):
        if time.time() > deadline:
            break
        # проверяем стол, когда в комнате уже никого нет (кассир ушла, всё заперто или нет);
        # если кассир не выходила из комнаты — через 3 с после ухода с места
        t_chk = None
        for (tt, d) in tl:
            if a <= tt <= b and not persons(d, H):
                t_chk = tt + 0.7
                break
        # таймкод: середина между «встала с места» и «вышла из комнаты» (на разметке это ближе всего)
        t_mid = (a + t_chk - 0.7) / 2 if t_chk is not None else a
        if t_chk is None:
            t_chk = a + 3.0
        t_chk = min(t_chk, info.duration - 0.1)
        f = frame_at(video_path, t_chk)
        if f is None:
            continue
        t_leave = max(0.0, t_mid - 0.3)
        ev = [max(0, a - 0.5), t_mid, t_chk]
        desk = ask(f, desk_box(laptop, zone), Q_DESK_LIST, "desk", t_chk, max_tokens=48).lower()
        alert = [k for k in DESK_ALERT if k in desk]
        if alert:
            findings.append(dict(type_id="valuables_unattended", t=t_leave, ev=ev,
                                 desc="Кассир покидает рабочее место, на столе остаются: " + ru_items(alert), score=0.8))
        if laptop is not None:
            lb = laptop
            w, h = lb[2] - lb[0], lb[3] - lb[1]
            sbox = [lb[0] - 0.3 * w, lb[1] - 0.3 * h, lb[2] + 0.3 * w, lb[3] + 0.3 * h]
            if "OPEN" in ask(f, sbox, Q_SCREEN, "screen", t_chk).upper():
                findings.append(dict(type_id="unlocked_pc", t=t_leave, ev=ev,
                                     desc="Кассир покидает рабочее место, на экране ноутбука остаётся открытая программа (компьютер не заблокирован)", score=0.75))
        g = frame_at(video_path, min(t_chk + 1.0, info.duration - 0.05))
        if _yes(ask(f, zone, Q_SAFE, "safe", t_chk)) and g is not None and _yes(ask(g, zone, Q_SAFE, "safe2", t_chk + 1)):
            findings.append(dict(type_id="safe_open", t=t_leave, ev=ev,
                                 desc="Кассир отходит, дверца сейфа остаётся открытой", score=0.7))

    # 3. Кто-то вошёл: посторонний за спиной, второй посетитель, инкассатор
    for t_in, n in enter_events(tl, H):
        if time.time() > deadline:
            break
        t_chk = min(t_in + 1.5, info.duration - 0.1)
        f = frame_at(video_path, t_chk)
        if f is None:
            continue
        full = [0, 0, W, int(H * 0.75)]
        ev = [t_in, t_chk, min(t_chk + 1.5, info.duration - 0.1)]
        f2 = frame_at(video_path, min(t_chk + 2.0, info.duration - 0.1))
        if _yes(ask(f, full, Q_BEHIND, "behind", t_chk)) or (f2 is not None and _yes(ask(f2, full, Q_BEHIND, "behind+2", t_chk + 2))):
            findings.append(dict(type_id="outsider_in_cash_zone", t=t_in, ev=ev,
                                 desc="В рабочую зону кассира (за стол со стороны кассира) заходит постороннее лицо", score=0.7))
        elif n >= 3 and (_yes(ask(f, full, Q_TWO_VISITORS, "two_visitors", t_chk)) or
                         (f2 is not None and _yes(ask(f2, full, Q_TWO_VISITORS, "two_visitors+2", t_chk + 2)))):
            findings.append(dict(type_id="second_client", t=t_in, ev=ev,
                                 desc="К столу подходит второй посетитель, пока обслуживание предыдущего не завершено", score=0.6))
        if _yes(ask(f, full, Q_COLLECTOR, "collector", t_chk)):
            ok_doc = False
            for dt in (0.0, 1.5, 3.0, 4.5, 6.0):
                g = frame_at(video_path, min(t_chk + dt, info.duration - 0.1))
                if g is None:
                    continue
                pp = persons(det.raw(g), H)
                box = full
                if len(pp) >= 2:
                    box = [min(b[0] for b in pp), min(b[1] for b in pp), max(b[2] for b in pp), max(b[3] for b in pp)]
                if _yes(ask(g, box, Q_COLLECTOR_DOC, "collector_doc", t_chk + dt)):
                    ok_doc = True
                    break
            if not ok_doc:
                findings.append(dict(type_id="collector_not_verified", t=t_in + 1, ev=ev,
                                     desc="Инкассатор передаёт сумку; передача и проверка препроводительного документа на видео не видны", score=0.6))

    # 2. Привычки на рабочем месте: каждые 4 с (сумка — каждые 12 с), пока кассир на месте
    occ = occupancy(tl, seat_or_zone, H)
    ts = [t for t, _ in tl]
    t = 1.0
    while t < info.duration - 0.5 and time.time() < deadline:
        i = min(range(len(ts)), key=lambda k: abs(ts[k] - t))
        if occ[i]:
            f = frame_at(video_path, t)
            if f is not None:
                # крупный фрагмент вокруг самого кассира: телефон и стакан в руках мелкие
                cbox = zone
                pp = persons(tl[i][1], H)
                if pp and seat:
                    sc = ((seat[0] + seat[2]) / 2, (seat[1] + seat[3]) / 2)
                    b = min(pp, key=lambda b: np.hypot(*np.subtract(center(b), sc)))
                    bw, bh = b[2] - b[0], b[3] - b[1]
                    cbox = [b[0] - 0.35 * bw, b[1] - 0.1 * bh, b[2] + 0.35 * bw, b[3] + 0.15 * bh]
                for q, tag, type_id, text in (
                        (Q_PHONE, "phone", "personal_items", "Кассир держит в руках мобильный телефон на рабочем месте"),
                        (Q_DRINK, "drink", "food_drink", "Кассир пьёт или ест за рабочим столом"),
                        (Q_BAG, "bag", "personal_items", "На рабочем столе кассира лежит личная сумка или одежда")):
                    if tag == "bag" and info.duration > 40 and int(t) % 12 not in (1, 2, 3, 4, 5, 6):
                        continue
                    qbox = zone if tag == "bag" else cbox
                    if _yes(ask(f, qbox, q, tag, t)):
                        need = (1.5, 3.0) if tag == "phone" else (1.5,)
                        ok = True
                        for dt in need:
                            g = frame_at(video_path, min(t + dt, info.duration - 0.05))
                            if g is None or not _yes(ask(g, qbox, q, f"{tag}+{dt}", t + dt)):
                                ok = False
                                break
                        if ok:
                            findings.append(dict(type_id=type_id, t=t, ev=[t, t + 0.7, t + 1.5], desc=text, score=0.65))
        t += 2.0 if info.duration <= 25 else (4.0 if info.duration <= 40 else 6.0)
        if progress:
            progress(0.3 + 0.4 * t / max(info.duration, 1))

    log["vlm_sec"] = round(time.time() - t0 - log["det_sec"], 1)
    return merge(findings), log


def merge(fs, gap=6.0):
    """Один тип в пределах gap секунд — одна находка (берём самую раннюю)."""
    fs = sorted(fs, key=lambda x: (x["type_id"], x["t"]))
    out = []
    for f in fs:
        if out and out[-1]["type_id"] == f["type_id"] and f["t"] - out[-1]["t"] <= gap:
            out[-1]["score"] = min(1.0, out[-1]["score"] + 0.1)
            continue
        out.append(dict(f))
    return sorted(out, key=lambda x: x["t"])


_tmpdir = None


def _tmp(img):
    global _tmpdir
    import tempfile
    if _tmpdir is None:
        _tmpdir = Path(tempfile.mkdtemp(prefix="v2_"))
    p = _tmpdir / f"{time.time_ns()}.jpg"
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return str(p)
