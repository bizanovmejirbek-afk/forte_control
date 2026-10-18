"""Проверка: правильно ли VLM отвечает на точечные вопросы по кадрам (нарушение vs норма)."""
import sys, time
from pathlib import Path
import cv2

from app.pipeline.vlm import MlxVLM
from app.pipeline.video import frame_at

V = Path(__file__).resolve().parent.parent.parent / "videos"
Q = {
 "screen": "Look at the laptop on the desk. Is its screen showing a lock screen / desktop wallpaper picture, or an open program window with text/tables? Answer exactly one word: LOCKED or OPEN.",
 "money": "Look at the top of the wooden desk. Are there banknotes, money, a cash bag, documents or papers lying on the desk (ignore the laptop, phone handset and counting machine)? Answer YES or NO.",
 "phone": "Is the woman holding a mobile phone in her hands? Answer YES or NO.",
 "cup": "Is the woman drinking or holding a cup/glass/bottle or food? Answer YES or NO.",
 "bag": "Is there a personal bag, handbag or coat lying on the desk? Answer YES or NO.",
 "safe": "Look at the grey metal safe (cabinet) next to the desk. Is its door open? Answer YES or NO.",
 "doc": "A man is handing over a green bag. Is he also showing or handing over a paper document or ID card? Answer YES or NO.",
 "behind": "Is there any person standing BEHIND the desk on the same side as the cashier's chair (in the cashier's working area), other than the cashier? Answer YES or NO.",
 "door": "Look at the glass door of the room. Is the door open (not closed)? Answer YES or NO.",
}
# (видео, секунда, вопрос, правильный ответ)
CASES = [
 ("Х-22", 16, "screen", "OPEN"), ("Х-23", 13, "screen", "LOCKED"), ("Х-28", 9, "screen", "OPEN"), ("Х-29", 11, "screen", "LOCKED"),
 ("Х-26", 12, "money", "YES"), ("Х-27", 22, "money", "NO"), ("Х-15", 16, "money", "YES"), ("Х-24", 10, "money", "YES"),
 ("Х-20", 6.5, "phone", "YES"), ("Х-10", 8, "phone", "NO"),
 ("Х-46", 13.5, "cup", "YES"), ("Х-43", 10, "cup", "NO"),
 ("Х-40", 19, "bag", "YES"), ("Х-44", 10, "bag", "NO"),
 ("Х-20", 6.5, "safe", "YES"), ("Х-21", 14, "safe", "NO"),
 ("Х-16", 8.7, "doc", "YES"), ("Х-17", 7, "doc", "NO"),
 ("Х-30", 14.5, "behind", "YES"), ("Х-31", 14, "behind", "NO"),
 ("Х-32", 7, "door", "YES"), ("Х-31", 11, "door", "NO"),
]

def main():
    vlm = MlxVLM("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
    tmp = Path("/tmp/probe"); tmp.mkdir(exist_ok=True)
    ok = 0
    for vid, t, q, want in CASES:
        f = frame_at(str(V / f"{vid}.mov"), t)
        f = f[: int(f.shape[0] * 0.75)]
        p = tmp / f"{vid}_{t}.jpg"; cv2.imwrite(str(p), f)
        t0 = time.time()
        ans = vlm.ask([str(p)], Q[q], max_tokens=5).upper()
        good = want in ans
        ok += good
        print(f"{'OK ' if good else 'ERR'} {vid} {t:>5}s {q:7s} want={want:6s} got={ans!r:12s} {time.time()-t0:.1f}s", flush=True)
    print(f"\nПравильно: {ok}/{len(CASES)}")

main()
