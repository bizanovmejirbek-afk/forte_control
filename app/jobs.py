"""Очередь задач. Одна задача за раз (одна видеокарта), состояние дублируется на диск."""
import json
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from . import config
from .pipeline import analyze

_executor = ThreadPoolExecutor(max_workers=1)
_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def _path(job_id):
    return config.JOBS_DIR / job_id


def _save(job):
    d = _path(job["job_id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.json").write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")


def get(job_id: str):
    with _lock:
        if job_id in _jobs:
            return dict(_jobs[job_id])
    f = _path(job_id) / "job.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return None


def video_path(job_id: str):
    return _path(job_id) / "video.mov"


def create(file_bytes_iter, filename: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    d = _path(job_id)
    d.mkdir(parents=True, exist_ok=True)
    size = 0
    with open(d / "video.mov", "wb") as f:
        for chunk in file_bytes_iter:
            size += len(chunk)
            if size > config.MAX_UPLOAD_MB * 1024 * 1024:
                raise ValueError("Файл слишком большой")
            f.write(chunk)
    job = {"job_id": job_id, "status": "processing", "filename": filename, "progress": 0.0,
           "submitted_at": time.time()}
    with _lock:
        _jobs[job_id] = job
    _save(job)
    _executor.submit(_run, job_id)
    return job_id


def _update(job_id, **kw):
    with _lock:
        _jobs[job_id].update(kw)
        job = dict(_jobs[job_id])
    _save(job)


def _run(job_id):
    try:
        deadline = get(job_id)["submitted_at"] + config.TIME_BUDGET_SEC  # очередь тоже считается
        res = analyze(str(video_path(job_id)),
                      progress=lambda p: _update(job_id, progress=round(p, 2)), deadline=deadline)
        _update(job_id, status="done", progress=1.0, **res)
    except Exception as e:
        traceback.print_exc()
        # Видео, которое не открывается, честно помечаем failed.
        _update(job_id, status="failed", error=str(e))
