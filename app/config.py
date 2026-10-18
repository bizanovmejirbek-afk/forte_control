"""Настройки сервиса. Всё задаётся переменными окружения."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Ключ для жюри. Обязательно поменять перед сдачей: export API_KEY=...
API_KEY = os.getenv("API_KEY", "change-me")

# Сколько кадров в секунду анализируем
SAMPLE_FPS = float(os.getenv("SAMPLE_FPS", "5"))

# Жёсткий бюджет времени на одно видео (с запасом до лимита жюри в 5 минут)
# Бюджет от момента ЗАГРУЗКИ видео: жюри ждёт не больше 5 минут, включая очередь
TIME_BUDGET_SEC = float(os.getenv("TIME_BUDGET_SEC", "250"))

# Какие модели подключать: "none" = заглушка (этап 1)
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "none")
VLM_BACKEND = os.getenv("VLM_BACKEND", "none")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "300"))
