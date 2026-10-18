"""HTTP-сервис: веб-интерфейс + API для жюри (раздел 10 ТЗ)."""
import hmac
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import config, jobs

app = FastAPI(title="Контроль кассовой дисциплины", version="0.1")
STATIC = Path(__file__).resolve().parent / "static"

PUBLIC_FIELDS = ("job_id", "status", "verdict", "violations")


@app.on_event("startup")
def warmup():
    """Загружаем модели при старте, чтобы первое видео жюри не ждало загрузку."""
    if config.VLM_BACKEND != "none":
        from .pipeline import _models
        from .pipeline.v2 import detector
        _models()
        detector()
        print("Модели загружены, сервис готов")


def _check_key(key):
    if not key or not hmac.compare_digest(key, config.API_KEY):
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-API-Key")


def _chunks(upload: UploadFile, size=1024 * 1024):
    while True:
        b = upload.file.read(size)
        if not b:
            break
        yield b


def _submit(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Нет файла (поле file)")
    try:
        job_id = jobs.create(_chunks(file), file.filename)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
    return {"job_id": job_id, "status": "processing"}


def _public(job, full=False):
    if job is None:
        raise HTTPException(status_code=404, detail="job_id не найден")
    if full:
        return job
    out = {k: job[k] for k in PUBLIC_FIELDS if k in job}
    if job["status"] == "failed":
        out["error"] = job.get("error", "")
    return out


# ---------- API для жюри (по ключу) ----------
@app.post("/api/v1/analyze")
def api_analyze(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    _check_key(x_api_key)
    return JSONResponse(_submit(file), status_code=202)


@app.get("/api/v1/jobs/{job_id}")
def api_job(job_id: str, x_api_key: str | None = Header(None)):
    _check_key(x_api_key)
    return _public(jobs.get(job_id))


# ---------- Для веб-интерфейса ----------
@app.post("/ui/analyze")
def ui_analyze(file: UploadFile = File(...)):
    return _submit(file)


@app.get("/ui/jobs/{job_id}")
def ui_job(job_id: str):
    return _public(jobs.get(job_id), full=True)


@app.get("/ui/video/{job_id}")
def ui_video(job_id: str):
    p = jobs.video_path(job_id)
    if not p.exists():
        raise HTTPException(status_code=404)
    return FileResponse(p, media_type="video/quicktime")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
