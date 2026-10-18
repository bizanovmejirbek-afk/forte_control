"""Главная функция анализа: analyze(video_path) -> dict в формате раздела 10 ТЗ."""
import json
import time
from pathlib import Path

from .. import config
from .detectors import get_detector
from .events import FrameObs, run_rules
from .video import frame_at, iter_frames, probe, to_jpeg_b64
from .vlm import get_vlm

RULES_PATH = Path(__file__).resolve().parent.parent / "rules.json"
_detector = None
_vlm = None


def load_rules():
    with open(RULES_PATH, encoding="utf-8") as f:
        return json.load(f)


def _models():
    global _detector, _vlm
    if _detector is None:
        _detector = get_detector()
    if _vlm is None:
        _vlm = get_vlm()
    return _detector, _vlm


def rule_ref(rules, type_id):
    """Ссылка на норму строго из реестра. Непроверенный пункт не показываем."""
    t = rules["types"].get(type_id, rules["types"]["other"])
    doc = rules["documents"][t["doc"]]
    ref = {"document": doc["title"], "url": doc["url"], "point": "", "quote": ""}
    if t.get("verified"):
        ref["point"] = t["point"]
        ref["quote"] = t["quote"]
    return ref


def make_violation(rules, type_id, t, desc, ev_ts, frames, score):
    t_rule = rules["types"].get(type_id, rules["types"]["other"])
    ref = rule_ref(rules, type_id)
    basis = ref["document"] + (f", {ref['point']}" if ref["point"] else "")
    return {
        "type": t_rule["name"],
        "timecode_sec": round(float(t), 1),
        "description": f"{desc.rstrip('.')}. Основание: {basis}.",
        "evidence_frames": [
            {"timestamp_sec": round(float(ts), 1), "image_base64": to_jpeg_b64(f)}
            for ts, f in list(zip(ev_ts, frames))[:3] if f is not None
        ],
        # дополнительные поля для интерфейса (не мешают схеме жюри)
        "type_id": type_id,
        "confidence": round(float(score), 2),
        "rule": ref,
    }


def analyze(video_path: str, progress=None, deadline=None) -> dict:
    """deadline — абсолютное время (time.time()), к которому нужен ответ. По умолчанию TIME_BUDGET_SEC от старта."""
    t0 = time.time()
    if deadline is None:
        deadline = t0 + config.TIME_BUDGET_SEC
    rules = load_rules()
    detector, vlm = _models()
    info = probe(video_path)
    meta = {"duration_sec": round(info.duration, 1), "detector": detector.name, "vlm": vlm.name}
    violations = []

    if config.VLM_BACKEND != "none":
        # Основной режим: детектор находит зону и моменты, VLM отвечает на точечные вопросы
        from .v2 import analyze_v2
        found, log = analyze_v2(video_path, vlm, info, deadline - 20, progress=progress)
        meta["v2"] = log
        for h in found:
            ev = [max(0.0, min(x, info.duration - 0.1)) for x in h["ev"]]
            frames = [frame_at(video_path, x) for x in ev]
            violations.append(make_violation(rules, h["type_id"], h["t"], h["desc"], ev, frames, h["score"]))
    else:
        # Режим без VLM: детектор + правила (этап разработки)
        timeline = []
        for t, frame in iter_frames(video_path, config.SAMPLE_FPS):
            timeline.append(FrameObs(t=t, detections=detector.detect(frame)))
            if progress and info.duration:
                progress(min(0.7, 0.7 * t / info.duration))
            if time.time() > deadline - 60:
                break
        meta["frames_analyzed"] = len(timeline)
        for c in run_rules(timeline):
            ev = c.evidence_ts or [c.t]
            frames = [frame_at(video_path, x) for x in ev]
            ok, type_id, desc = vlm.review(c, frames, rules)
            if ok:
                violations.append(make_violation(rules, type_id, c.t, desc, ev, frames, c.score))

    meta["processing_sec"] = round(time.time() - t0, 1)
    return {"verdict": "violation" if violations else "no_violation", "violations": violations, "meta": meta}
