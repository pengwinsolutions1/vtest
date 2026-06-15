#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="$ROOT/logs/webapp.pid"
if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE")"
  kill "$PID" 2>/dev/null || true
  sleep 1
  kill -9 "$PID" 2>/dev/null || true
  rm -f "$PIDFILE"
fi
STRAYS="$(lsof -ti:3000 2>/dev/null || true)"
if [ -n "$STRAYS" ]; then
  echo "$STRAYS" | xargs kill -9 2>/dev/null || true
fi
echo "[stop] done"
