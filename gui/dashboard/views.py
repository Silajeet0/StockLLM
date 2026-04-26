"""
dashboard/views.py
==================
Django views for the Stock Market LLM Dashboard (predict-only).
"""

import json
import os
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from .pipeline_runner import STAGES, run_pipeline, get_task, get_ohlc_data

PROJECT_ROOT = settings.STOCK_PROJECT_ROOT


def index(request):
    """Main dashboard page."""
    return render(request, "dashboard/index.html", {
        "stages": list(STAGES.items()),
    })


@csrf_exempt
def api_run(request):
    """POST /api/run/ — start a predict pipeline run."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        body     = json.loads(request.body) if request.body else {}
        log_runs = body.get("log_runs", True)
        task_id  = run_pipeline(PROJECT_ROOT, log_runs=log_runs)
        return JsonResponse({"task_id": task_id, "status": "started"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def api_progress(request, task_id):
    """GET /api/progress/<task_id>/ — poll task progress."""
    task = get_task(task_id)
    if not task:
        return JsonResponse({"error": "Task not found"}, status=404)

    total_w  = sum(s["weight"] for s in STAGES.values())
    achieved = 0
    for stage_key, stage_meta in STAGES.items():
        pct = task["stages"][stage_key]["pct"]
        achieved += (pct / 100) * stage_meta["weight"]
    overall_pct = int(achieved / total_w * 100)

    return JsonResponse({
        "task_id":     task["id"],
        "status":      task["status"],
        "overall_pct": overall_pct,
        "stages":      task["stages"],
        "log_tail":    task["log_lines"][-30:],
        "error":       task["error"],
        "started_at":  task["started_at"],
        "finished_at": task["finished_at"],
    })


def api_results(request, task_id):
    """GET /api/results/<task_id>/ — get final results."""
    task = get_task(task_id)
    if not task:
        return JsonResponse({"error": "Task not found"}, status=404)
    if task["status"] != "complete":
        return JsonResponse({"error": "Not complete yet", "status": task["status"]}, status=202)
    return JsonResponse({"result": task["result"], "task_id": task_id})


def api_ohlc(request):
    """GET /api/ohlc/ — OHLC data for RELIANCE.NS candlestick chart."""
    data = get_ohlc_data(PROJECT_ROOT, days=60)
    return JsonResponse({"ticker": "RELIANCE.NS", "data": data})
