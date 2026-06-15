#!/usr/bin/env bash
# Build + start the webapp in production mode. For local development use
# `npm run dev` directly (faster, with HMR).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/webapp.log"
PIDFILE="$LOG_DIR/webapp.pid"

if lsof -ti:3000 >/dev/null 2>&1; then
  echo "Port 3000 already in use. Run scripts/stop.sh first." >&2
  exit 1
fi

if [ ! -f .env.local ]; then
  echo "Missing .env.local — copy .env.example and fill in REPLICATE_API_TOKEN" >&2
  exit 1
fi

if [ ! -d node_modules ]; then
  echo "[start] installing deps…"
  npm install
fi

echo "[start] building…"
npm run build >> "$LOG" 2>&1

echo "[start] launching…"
nohup npm start >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"

# Wait for /api/jobs/probe (404 is fine — proves the route handler is up)
for i in $(seq 1 30); do
  if curl -fsS -o /dev/null -w "%{http_code}" http://localhost:3000/ 2>/dev/null | grep -qE '^(200|404|500)$'; then
    LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 'localhost')"
    echo "[start] up — pid $(cat "$PIDFILE")"
    echo "          Web    http://localhost:3000"
    echo "          LAN    http://$LAN_IP:3000"
    echo "          Logs   $LOG"
    exit 0
  fi
  sleep 1
done

echo "[start] FAILED — no response from :3000 within 30s. Last log lines:" >&2
tail -30 "$LOG" >&2
exit 1
