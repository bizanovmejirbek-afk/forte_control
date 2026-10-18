"""Проверка как у жюри: отправить видео в API, дождаться ответа, проверить формат.

Запуск: python -m scripts.jury_check --url http://localhost:8000 --key change-me --video ~/Downloads/open/videos/X-1.mov
"""
import argparse
import time

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--video", required=True)
    a = ap.parse_args()
    h = {"X-API-Key": a.key}

    assert requests.post(f"{a.url}/api/v1/analyze", files={"file": open(a.video, "rb")}).status_code == 401, \
        "запрос без ключа должен отклоняться"
    t0 = time.time()
    r = requests.post(f"{a.url}/api/v1/analyze", headers=h, files={"file": open(a.video, "rb")})
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print("job_id:", job_id)
    while True:
        j = requests.get(f"{a.url}/api/v1/jobs/{job_id}", headers=h).json()
        if j["status"] != "processing":
            break
        if time.time() - t0 > 300:
            raise SystemExit("ПРОВАЛ: дольше 5 минут")
        time.sleep(2)
    dt = time.time() - t0
    assert j["status"] == "done", j
    assert j["verdict"] in ("violation", "no_violation")
    assert isinstance(j["violations"], list)
    assert (j["verdict"] == "violation") == bool(j["violations"])
    for v in j["violations"]:
        for k in ("type", "timecode_sec", "description", "evidence_frames"):
            assert k in v, f"нет поля {k}"
        assert 1 <= len(v["evidence_frames"]) <= 3, "нужно 1-3 кадра"
        for f in v["evidence_frames"]:
            assert f["image_base64"][:4] == "/9j/", "кадр должен быть JPEG"
    print(f"OK за {dt:.1f} c: verdict={j['verdict']}, нарушений={len(j['violations'])}")
    for v in j["violations"]:
        print(f"  {v['timecode_sec']:>6} c  {v['type']}")


if __name__ == "__main__":
    main()
