#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
PID_DIR="${PROJECT_DIR}/.pids"
QUERY_PORT="${QUERY_PORT:-8080}"
IMPORT_PORT="${IMPORT_PORT:-8081}"

mkdir -p "${LOG_DIR}" "${PID_DIR}"

QUERY_PID_FILE="${PID_DIR}/query_server.pid"
IMPORT_PID_FILE="${PID_DIR}/import_server.pid"
QUERY_LOG="${LOG_DIR}/query_server.log"
IMPORT_LOG="${LOG_DIR}/import_server.log"

cd "${PROJECT_DIR}"

activate_env() {
  if [[ -f "${PROJECT_DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/.venv/bin/activate"
  fi
}

is_running() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null
}

start_one() {
  local name="$1"
  local app_path="$2"
  local port="$3"
  local log_file="$4"
  local pid_file="$5"

  if is_running "${pid_file}"; then
    echo "${name} already running: pid=$(cat "${pid_file}"), port=${port}"
    return
  fi

  echo "Starting ${name} on port ${port} ..."
  nohup python -m uvicorn "${app_path}" --host 0.0.0.0 --port "${port}" --reload \
    > "${log_file}" 2>&1 &
  echo "$!" > "${pid_file}"
  sleep 1

  if is_running "${pid_file}"; then
    echo "${name} started: pid=$(cat "${pid_file}"), log=${log_file}"
  else
    echo "${name} failed to start. Check log: ${log_file}" >&2
    return 1
  fi
}

stop_one() {
  local name="$1"
  local pid_file="$2"

  if ! [[ -f "${pid_file}" ]]; then
    echo "${name} not running: pid file not found"
    return
  fi

  local pid
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping ${name}: pid=${pid}"
    kill "${pid}" 2>/dev/null || true
    sleep 1
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Force stopping ${name}: pid=${pid}"
      kill -9 "${pid}" 2>/dev/null || true
    fi
  else
    echo "${name} already stopped"
  fi
  rm -f "${pid_file}"
}

status_one() {
  local name="$1"
  local port="$2"
  local pid_file="$3"

  if is_running "${pid_file}"; then
    echo "${name}: running, pid=$(cat "${pid_file}"), port=${port}"
  else
    echo "${name}: stopped, port=${port}"
  fi
}

start_all() {
  activate_env
  start_one "query_server" "app.query_process.api.query_server:app" "${QUERY_PORT}" "${QUERY_LOG}" "${QUERY_PID_FILE}"
  start_one "import_server" "app.import_process.api.import_server:app" "${IMPORT_PORT}" "${IMPORT_LOG}" "${IMPORT_PID_FILE}"
  echo
  echo "Query health: http://127.0.0.1:${QUERY_PORT}/health"
  echo "Query page:   http://127.0.0.1:${QUERY_PORT}/chat.html"
  echo "Import page:  http://127.0.0.1:${IMPORT_PORT}/import"
}

stop_all() {
  stop_one "query_server" "${QUERY_PID_FILE}"
  stop_one "import_server" "${IMPORT_PID_FILE}"
}

status_all() {
  status_one "query_server" "${QUERY_PORT}" "${QUERY_PID_FILE}"
  status_one "import_server" "${IMPORT_PORT}" "${IMPORT_PID_FILE}"
}

case "${1:-start}" in
  start)
    start_all
    ;;
  stop)
    stop_all
    ;;
  restart)
    stop_all
    start_all
    ;;
  status)
    status_all
    ;;
  logs)
    echo "Query log:  ${QUERY_LOG}"
    echo "Import log: ${IMPORT_LOG}"
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}"
    echo "Optional env: QUERY_PORT=8080 IMPORT_PORT=8081"
    exit 1
    ;;
esac
