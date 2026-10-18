"""Движок событий: из покадровых детекций делает кандидатов на нарушения с таймкодом.

Каждое правило — функция (timeline) -> list[Candidate]. Новые типы добавляются
сюда и в rules.json, остальной код не меняется.
"""
from dataclasses import dataclass, field


@dataclass
class Candidate:
    type_id: str
    t: float                 # главный таймкод (сек)
    score: float             # уверенность 0..1
    t_start: float = 0.0
    t_end: float = 0.0
    objects: list = field(default_factory=list)   # что видно ("наличные", "сумка")
    evidence_ts: list = field(default_factory=list)  # 1-3 таймкода кадров-доказательств
    note: str = ""


@dataclass
class FrameObs:
    t: float
    detections: list


RULES = []


def rule(fn):
    RULES.append(fn)
    return fn


@rule
def cashier_left_workplace(timeline):
    """Типы unlocked_pc / valuables_unattended / safe_open.
    Этап 2: найти зону рабочего места (стол+ноутбук), трек кассира,
    момент выхода кассира из зоны и состояние стола/экрана/сейфа в этот момент."""
    return []


@rule
def extra_person_near_cashier(timeline):
    """Типы outsider_in_cash_zone / second_client / collector_not_verified."""
    return []


@rule
def forbidden_objects(timeline):
    """Типы personal_items / food_drink: телефон, сумка, стакан у кассира."""
    return []


def run_rules(timeline):
    out = []
    for fn in RULES:
        try:
            out.extend(fn(timeline))
        except Exception as e:  # одно сломанное правило не должно ронять весь анализ
            print(f"[events] rule {fn.__name__} failed: {e}")
    return merge_close(out)


def merge_close(cands, gap=3.0):
    """Склеиваем кандидатов одного типа, если они ближе gap секунд."""
    cands = sorted(cands, key=lambda c: (c.type_id, c.t))
    merged = []
    for c in cands:
        if merged and merged[-1].type_id == c.type_id and c.t - merged[-1].t <= gap:
            if c.score > merged[-1].score:
                merged[-1] = c
            continue
        merged.append(c)
    return sorted(merged, key=lambda c: c.t)


# Режим проверки интерфейса до подключения детекторов: DEMO_FAKE=1
import os as _os
if _os.getenv("DEMO_FAKE") == "1":
    @rule
    def _demo_fake(timeline):
        if not timeline:
            return []
        t = timeline[len(timeline) // 2].t
        return [Candidate("valuables_unattended", t, 0.5, objects=["наличные"],
                          evidence_ts=[max(0, t - 1), t, t + 1], note="ДЕМО-РЕЖИМ")]
