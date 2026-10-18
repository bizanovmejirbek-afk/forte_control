#!/bin/bash
# Автопилот проверки: гоняет оценку на всех роликах, ждёт новую версию кода (файл RUN_VERSION) и повторяет.
# caffeinate не даёт Mac уснуть, пока скрипт работает.
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p runs
[ -f RUN_VERSION ] || echo 1 > RUN_VERSION
while true; do
  v=$(cat RUN_VERSION)
  echo "=== $(date '+%H:%M') запуск версии $v ==="
  VLM_BACKEND=mlx caffeinate -dimsu python -m scripts.evaluate --out "runs/eval_v$v.json" 2>&1 | tee "runs/log_v$v.txt"
  echo "$(date '+%H:%M')" > "runs/DONE_v$v"
  echo "=== версия $v готова, жду новую версию от Claude ==="
  while [ "$(cat RUN_VERSION)" = "$v" ]; do caffeinate -dimsu -t 60; done
done
