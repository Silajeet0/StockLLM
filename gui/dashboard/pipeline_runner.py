"""
dashboard/pipeline_runner.py
============================
Manages background prediction pipeline execution with real-time progress
tracking.  Spawns a subprocess for each run.

Changes vs. original:
  - Single ticker/company: RELIANCE.NS / Reliance Industries Limited
  - Only one mode: load saved models + predict (no training flags)
  - Simplified STAGES to match the predict-only workflow
"""

import os
import sys
import uuid
import json
import subprocess
import threading
from datetime import datetime

# Task store: task_id -> task_dict
_tasks = {}
_tasks_lock = threading.Lock()

# Fixed company / ticker
TICKER  = "RELIANCE.NS"
COMPANY = "Reliance Industries Limited"

# Progress stages for the predict pipeline
STAGES = {
    "data_download":  {"label": "Data Download (15y)",        "weight": 20},
    "feature_eng":    {"label": "Feature Engineering",        "weight": 10},
    "itransformer":   {"label": "iTransformer Inference",     "weight": 25},
    "tft":            {"label": "TFT Inference",              "weight": 25},
    "news":           {"label": "News Sentiment Analysis",    "weight": 10},
    "orchestrator":   {"label": "Final Orchestration",        "weight": 10},
}


def _make_task():
    task_id = uuid.uuid4().hex[:8].upper()
    task = {
        "id":          task_id,
        "ticker":      TICKER,
        "company":     COMPANY,
        "status":      "pending",
        "stages":      {k: {"status": "pending", "pct": 0, "message": ""} for k in STAGES},
        "log_lines":   [],
        "result":      None,
        "error":       None,
        "started_at":  None,
        "finished_at": None,
    }
    return task_id, task


def _parse_progress(line, task):
    """Parse a stdout line and update stage progress."""
    line = line.strip()
    if not line:
        return

    stages = task["stages"]

    # ── Data download ──────────────────────────────────────────────────────
    if "Probing latest available date" in line or "Download range" in line:
        stages["data_download"]["status"]  = "running"
        stages["data_download"]["pct"]     = 20
        stages["data_download"]["message"] = "Connecting to Yahoo Finance…"
    elif "[DataLoader] Downloading" in line:
        stages["data_download"]["status"]  = "running"
        stages["data_download"]["pct"]     = 50
        stages["data_download"]["message"] = "Downloading 15-year OHLCV data…"
    elif "[DataLoader] Saved" in line or "Downloaded" in line and "rows" in line:
        stages["data_download"]["status"]  = "complete"
        stages["data_download"]["pct"]     = 100
        stages["data_download"]["message"] = "Download complete"

    # ── Feature engineering ────────────────────────────────────────────────
    elif "[DataPipeline]" in line and "feature" in line.lower():
        stages["feature_eng"]["status"]  = "running"
        stages["feature_eng"]["pct"]     = 50
        stages["feature_eng"]["message"] = "Computing technical indicators…"
    elif "[DataPipeline] ✅ Features saved" in line:
        stages["data_download"]["status"] = "complete"
        stages["data_download"]["pct"]    = 100
        stages["feature_eng"]["status"]   = "complete"
        stages["feature_eng"]["pct"]      = 100
        stages["feature_eng"]["message"]  = "Features ready"

    # ── iTransformer ───────────────────────────────────────────────────────
    elif "iTransformer Agent" in line and "inference" in line.lower():
        stages["itransformer"]["status"]  = "running"
        stages["itransformer"]["pct"]     = 20
        stages["itransformer"]["message"] = "Loading iTransformer checkpoint…"
    elif "[iTransformer]" in line and ("loaded" in line.lower() or "checkpoint" in line.lower()):
        stages["itransformer"]["pct"]     = 60
        stages["itransformer"]["message"] = "Running inference…"
    elif "[iTransformer]" in line and ("MAE" in line or "RMSE" in line or "Prediction" in line or "predict" in line.lower()):
        stages["itransformer"]["status"]  = "complete"
        stages["itransformer"]["pct"]     = 100
        stages["itransformer"]["message"] = "Predictions generated"

    # ── TFT ────────────────────────────────────────────────────────────────
    elif "TFT Agent" in line and "inference" in line.lower():
        stages["tft"]["status"]  = "running"
        stages["tft"]["pct"]     = 20
        stages["tft"]["message"] = "Loading TFT checkpoint…"
    elif "[TFT]" in line and ("loaded" in line.lower() or "checkpoint" in line.lower()):
        stages["tft"]["pct"]     = 60
        stages["tft"]["message"] = "Running TFT inference…"
    elif "TFT" in line and ("✅" in line or "Prediction" in line or "complete" in line.lower()):
        stages["tft"]["status"]  = "complete"
        stages["tft"]["pct"]     = 100
        stages["tft"]["message"] = "TFT complete"

    # ── News ───────────────────────────────────────────────────────────────
    elif "News Agent" in line:
        stages["news"]["status"]  = "running"
        stages["news"]["pct"]     = 20
        stages["news"]["message"] = "Scraping news headlines…"
    elif "News cache hit" in line:
        stages["news"]["status"]  = "complete"
        stages["news"]["pct"]     = 100
        stages["news"]["message"] = "Loaded from cache"
    elif "News sentiment cached" in line:
        stages["news"]["status"]  = "complete"
        stages["news"]["pct"]     = 100
        stages["news"]["message"] = "Sentiment analysed"

    # ── Orchestrator ───────────────────────────────────────────────────────
    elif "Orchestrat" in line or "ORCHESTRATOR" in line:
        stages["orchestrator"]["status"]  = "running"
        stages["orchestrator"]["pct"]     = 30
        stages["orchestrator"]["message"] = "Synthesising predictions…"
    elif "PREDICTION COMPLETE" in line or "Final output saved" in line:
        stages["orchestrator"]["status"]  = "complete"
        stages["orchestrator"]["pct"]     = 100
        stages["orchestrator"]["message"] = "Complete!"
        for s in stages.values():
            if s["status"] != "complete":
                s["status"] = "complete"
                s["pct"]    = 100


def run_pipeline(project_root: str, log_runs: bool = True) -> str:
    """
    Start a predict-only pipeline run in a background thread.
    Returns task_id immediately.
    """
    task_id, task = _make_task()
    with _tasks_lock:
        _tasks[task_id] = task

    def _worker():
        task["status"]     = "running"
        task["started_at"] = datetime.now().isoformat()

        env = os.environ.copy()
        env["STOCK_TICKER"]    = TICKER
        env["STOCK_COMPANY"]   = COMPANY
        env["PYTHONUNBUFFERED"] = "1"

        python_exe = sys.executable
        main_py    = os.path.join(project_root, "main.py")

        cmd = [python_exe, main_py]
        if not log_runs:
            cmd.append("--no-log")

        task["stages"]["data_download"]["status"]  = "running"
        task["stages"]["data_download"]["pct"]     = 5
        task["stages"]["data_download"]["message"] = "Initialising pipeline…"

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=project_root,
                bufsize=1,
            )

            for line in proc.stdout:
                task["log_lines"].append(line.rstrip())
                if len(task["log_lines"]) > 500:
                    task["log_lines"] = task["log_lines"][-400:]
                _parse_progress(line, task)

            proc.wait()

            if proc.returncode == 0:
                # Load result JSON (saved by OrchestratorAgent.save())
                _slug = TICKER.replace(".", "_").replace("-", "_").lower()
                result_path = os.path.join(
                    project_root, "saved_models", f"{_slug}_final_prediction.json"
                )
                if os.path.exists(result_path):
                    with open(result_path) as f:
                        task["result"] = json.load(f)
                task["status"] = "complete"
            else:
                task["status"] = "error"
                task["error"]  = f"Pipeline exited with code {proc.returncode}"
                for s in task["stages"].values():
                    if s["status"] == "running":
                        s["status"] = "error"

        except Exception as e:
            task["status"] = "error"
            task["error"]  = str(e)

        task["finished_at"] = datetime.now().isoformat()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return task_id


def get_task(task_id: str):
    with _tasks_lock:
        return _tasks.get(task_id)


def get_ohlc_data(project_root: str, days: int = 60) -> list:
    """
    Read the cached raw CSV for RELIANCE.NS and return last `days` rows.
    Falls back to a live yfinance download if no cache exists.
    RAW_CSV is derived directly from project_root — no config import needed
    so this works inside the Django process without sys.path manipulation.
    """
    RAW_CSV = os.path.join(project_root, "data", "raw", "reliance_ns_raw.csv")
    data = None

    if os.path.exists(RAW_CSV):
        try:
            import pandas as pd
            df = pd.read_csv(RAW_CSV, parse_dates=["date"])
            df = df.sort_values("date").tail(days).reset_index(drop=True)
            data = df.to_dict(orient="records")
        except Exception:
            pass

    if data is None:
        try:
            import yfinance as yf
            import pandas as pd
            df = yf.download(TICKER, period="3mo", interval="1d",
                             auto_adjust=False, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                df.index.name = "date"
                df = df.reset_index()
                df["date"] = pd.to_datetime(df["date"])
                df = df.rename(columns={"adj close": "adj_close"})
                cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
                df = df[cols].tail(days)
                data = df.to_dict(orient="records")
        except Exception:
            data = []

    # Serialise dates to strings and floats
    for row in (data or []):
        if hasattr(row.get("date"), "isoformat"):
            row["date"] = row["date"].isoformat()[:10]
        for k in ["open", "high", "low", "close", "adj_close", "volume"]:
            if k in row and row[k] is not None:
                try:
                    row[k] = float(row[k])
                except (TypeError, ValueError):
                    row[k] = None
    return data or []
