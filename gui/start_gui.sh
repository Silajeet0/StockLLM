#!/bin/bash
# start_gui.sh — Launch the Stock Market LLM Django GUI
# Usage: bash start_gui.sh [port]  (default port: 8000)

PORT=${1:-8000}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        Stock Market LLM — Django GUI Launcher        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  GUI Directory : $SCRIPT_DIR"
echo "  Port          : $PORT"
echo ""

# Check Python
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
  echo "❌  Python not found. Please install Python 3.9+"
  exit 1
fi

PYTHON=$(command -v python3 || command -v python)

# Install requirements if needed
echo "📦  Checking dependencies..."
$PYTHON -m pip install -q django>=4.2 yfinance pandas numpy 2>/dev/null && echo "✅  Dependencies OK"

echo ""
echo "🚀  Starting server at http://127.0.0.1:$PORT"
echo "    Press Ctrl+C to stop."
echo ""

cd "$SCRIPT_DIR"
$PYTHON manage.py runserver "0.0.0.0:$PORT" --noreload
