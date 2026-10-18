#!/bin/bash
# Запуск сервиса. Ключ для жюри задаётся здесь или заранее: export API_KEY=...
cd "$(dirname "$0")"
source .venv/bin/activate
export API_KEY="${API_KEY:-change-me}"
uvicorn app.main:app --host 0.0.0.0 --port 8000
