#!/bin/bash
# Боевой запуск для жюри: сервис с моделью + публичный адрес через ngrok.
# Перед первым запуском:
#   1) положите ключ для жюри в файл .api_key (одной строкой)
#   2) положите адрес ngrok-домена в файл .ngrok_domain (например forte-cash.ngrok-free.app)
cd "$(dirname "$0")"
source .venv/bin/activate
export API_KEY="$(cat .api_key)"
export VLM_BACKEND=mlx
DOMAIN="$(cat .ngrok_domain)"

# сервис (caffeinate не даёт Mac уснуть, пока он работает)
lsof -ti:8000 | xargs kill 2>/dev/null; sleep 1
export PYTHONUNBUFFERED=1
caffeinate -dimsu uvicorn app.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
SERVER_PID=$!
echo "Сервис запускается, загружаю модели..."
until grep -q "Application startup complete" server.log 2>/dev/null; do sleep 2; if ! kill -0 $SERVER_PID 2>/dev/null; then echo "Сервис упал, см. server.log"; exit 1; fi; done
echo "Модели загружены. Открываю публичный адрес https://$DOMAIN"
trap "kill $SERVER_PID" EXIT
ngrok http --url="$DOMAIN" 8000 || ngrok http --domain="$DOMAIN" 8000
