"""Оценка качества на размеченных роликах (как будет считать жюри).

Запуск:  python -m scripts.evaluate --videos ~/Downloads/open/videos --labels ~/Downloads/open/labels_annotated.csv
"""
import argparse
import csv
import json
import re
import time
import unicodedata
from pathlib import Path

from app.pipeline import analyze


def norm(name: str) -> str:
    # X латинская и Х кириллическая -> одна форма
    name = unicodedata.normalize("NFC", Path(name).stem).upper().replace("Х", "X")
    return name


def parse_tc(s):
    if not s:
        return None
    s = str(s)
    m = re.findall(r"\d+(?:[.,]\d+)?", s)
    if not m:
        return None
    if "мин" in s and len(m) >= 2:
        return float(m[0]) * 60 + float(m[1].replace(",", "."))
    return float(m[0].replace(",", "."))


def load_labels(path):
    labels = {}
    p = Path(path)
    if p.suffix.lower() == ".csv":
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                labels[norm(r["file"])] = {
                    "violation": r["label"] == "violation",
                    "t": parse_tc(r.get("timecode_sec")),
                    "type": r.get("type_hypothesis", ""),
                }
    else:  # исходный open_labels.xlsx
        import openpyxl
        ws = openpyxl.load_workbook(p).active
        for row in list(ws.iter_rows(values_only=True))[1:]:
            if not row[0]:
                continue
            labels[norm(row[0])] = {"violation": str(row[2]).strip() == "нарушение",
                                    "t": parse_tc(row[1]), "type": ""}
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(Path(__file__).resolve().parent.parent.parent / "videos"))
    ap.add_argument("--labels", default=str(Path(__file__).resolve().parent.parent.parent / "labels_annotated.csv"))
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--only", nargs="*", help="только эти ролики, например X-22 X-23")
    a = ap.parse_args()

    labels = load_labels(a.labels)
    vids = sorted(Path(a.videos).glob("*.mov"))
    if a.only:
        want = {norm(x) for x in a.only}
        vids = [v for v in vids if norm(v.name) in want]

    rows, tp, tn, fp, fn, tc_err, slow = [], 0, 0, 0, 0, [], 0
    for v in vids:
        lab = labels.get(norm(v.name))
        if lab is None:
            continue
        t0 = time.time()
        res = analyze(str(v))
        dt = time.time() - t0
        slow += dt > 300
        pred = res["verdict"] == "violation"
        if pred and lab["violation"]:
            tp += 1
        elif pred:
            fp += 1
        elif lab["violation"]:
            fn += 1
        else:
            tn += 1
        err = None
        if pred and lab["violation"] and lab["t"] is not None:
            err = min(abs(x["timecode_sec"] - lab["t"]) for x in res["violations"])
            tc_err.append(err)
        rows.append({"file": v.name, "true": lab["violation"], "pred_v": pred, "true_t": lab["t"],
                     "true_type": lab["type"],
                     "pred": [(x["timecode_sec"], x.get("type_id"), x["description"][:200]) for x in res["violations"]],
                     "tc_err": err, "sec": round(dt, 1), "meta": res.get("meta", {})})
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        mark = "OK " if pred == lab["violation"] else "ERR"
        print(f"{mark} {v.name:10s} true={'V' if lab['violation'] else '-'} "
              f"pred={[x['timecode_sec'] for x in res['violations']]} t_true={lab['t']} {dt:.1f}s")

    n = tp + tn + fp + fn
    print("\n=== ИТОГ ===")
    print(f"Роликов: {n}  точность вердикта: {(tp + tn) / max(1, n):.0%}")
    print(f"Пропуски (FN): {fn}   ложные тревоги (FP): {fp}")
    if tc_err:
        print(f"Ошибка таймкода: медиана {sorted(tc_err)[len(tc_err)//2]:.1f} с, "
              f"в пределах 2 с: {sum(e <= 2 for e in tc_err)}/{len(tc_err)}")
    print(f"Дольше 5 минут: {slow}")
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
