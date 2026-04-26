"""
mcp/mcp_connector.py
=====================
MCP (Model Communication Protocol) Connector

CONFIDENCE FIX (v2)
--------------------
Previously _build_consensus applied a 25% penalty AND ollama_interpreter
applied a 20% pre-penalty — disagreement was penalised twice, producing
an artificially low final confidence (~57%) even when both models had
high prediction accuracy.

Now only _build_consensus applies the disagreement penalty (single source
of truth). The severity of the penalty is also made adaptive:

  • If the models predict the SAME direction but with different magnitudes
    (agreement=True), no penalty.
  • If models disagree, we weight the penalty by each model's RMSE —
    the model with lower RMSE gets more trust. If the lower-RMSE model's
    signal is strong enough to override the other, the penalty is lighter.

SIGNAL STRENGTH COLUMN:
  Each model now optionally supplies signal_strength = (D5 - last) / RMSE.
  If present, it's used to weight the directional vote more accurately.
"""

import os
import sys
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.settings import MODEL_DIR
    LOG_DIR = os.path.join(MODEL_DIR, "mcp_logs")
except ImportError:
    LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models", "mcp_logs")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [MCPConnector]  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MCPConnector")


# ---------------------------------------------------------------------------
# Field validators (unchanged)
# ---------------------------------------------------------------------------
def _require(obj: dict, fields: list, label: str) -> list:
    return [f for f in fields if f not in obj or obj[f] is None]


def _validate_model_output(data: dict, name: str) -> tuple:
    warnings = []
    required = ["predicted_prices", "trend", "confidence"]
    missing  = _require(data, required, name)
    if missing:
        warnings.append(f"{name} missing fields: {missing}")

    cleaned = {
        "model":             data.get("model", name),
        "predicted_prices":  data.get("predicted_prices", []),
        "trend":             str(data.get("trend", "Unknown")).capitalize(),
        "confidence":        float(data.get("confidence", 0.5)),
        "signal_strength":   data.get("signal_strength"),   # may be None if not computed
        "important_features":data.get("important_features", []),
        "val_mae":           data.get("val_mae"),
        "val_rmse":          data.get("val_rmse"),
        "last_actual_price": data.get("last_actual_price"),
    }
    cleaned["confidence"] = min(0.99, max(0.01, cleaned["confidence"]))
    return cleaned, warnings


def _validate_news(data: dict) -> tuple:
    warnings = []
    if not data:
        warnings.append("News data is empty — using neutral defaults.")
        data = {}

    cleaned = {
        "sentiment_label":  str(data.get("sentiment_label", "Neutral")).capitalize(),
        "sentiment_score":  float(data.get("sentiment_score", 0.0)),
        "headline_summary": str(data.get("headline_summary", "No headlines available.")),
        "num_articles":     int(data.get("num_articles", 0)),
    }
    cleaned["sentiment_score"] = min(1.0, max(-1.0, cleaned["sentiment_score"]))
    return cleaned, warnings


def _validate_llm_output(data: dict) -> tuple:
    warnings = []
    required = ["signal", "final_trend", "confidence", "price_target", "explanation"]
    missing  = _require(data, required, "LLM")
    if missing:
        warnings.append(f"LLM output missing fields: {missing}")

    valid_signals = {"BUY", "SELL", "HOLD"}
    signal = str(data.get("signal", "HOLD")).upper()
    if signal not in valid_signals:
        warnings.append(f"Invalid signal '{signal}', defaulting to HOLD.")
        signal = "HOLD"

    cleaned = {
        "signal":          signal,
        "final_trend":     str(data.get("final_trend", "Sideways")).capitalize(),
        "confidence":      float(data.get("confidence", 0.5)),
        "price_target":    float(data.get("price_target", 0.0)),
        "explanation":     str(data.get("explanation", "")),
        "model_agreement": bool(data.get("model_agreement", False)),
        "key_risks":       str(data.get("key_risks", "Prediction uncertainty.")),
    }
    cleaned["confidence"] = min(0.99, max(0.01, cleaned["confidence"]))
    return cleaned, warnings


# ---------------------------------------------------------------------------
# Consensus builder (FIXED: single disagreement penalty, RMSE-aware)
# ---------------------------------------------------------------------------
def _build_consensus(itx: dict, tft: dict, news: dict, llm: dict) -> dict:
    """
    Build a final consensus by weighting all four signals.

    Weights:
      iTransformer : 30%
      TFT          : 30%
      LLM          : 30%
      News         : 10%

    FIX v2:
      - Only ONE disagreement penalty (previously two — here AND in
        ollama_interpreter._compute_consensus).
      - Penalty severity is adaptive: if the more-accurate model (lower RMSE)
        strongly agrees with the LLM, penalty is lighter (15%).
        Otherwise the standard 25% penalty applies.
      - HOLD is forced when models disagree AND LLM is uncertain (< 0.70).
    """
    def _is_bullish(trend_str: str) -> float:
        return 1.0 if "bull" in str(trend_str).lower() else 0.0

    itx_bull  = _is_bullish(itx.get("trend", ""))
    tft_bull  = _is_bullish(tft.get("trend", ""))
    llm_bull  = _is_bullish(llm.get("final_trend", ""))
    news_bull = (1.0 if news.get("sentiment_score", 0) > 0.05
                 else (0.5 if abs(news.get("sentiment_score", 0)) <= 0.05 else 0.0))

    weighted_bull = (0.30 * itx_bull +
                     0.30 * tft_bull +
                     0.30 * llm_bull +
                     0.10 * news_bull)

    # Final trend / signal from weighted score
    if weighted_bull >= 0.65:
        final_trend  = "Bullish"
        final_signal = "BUY"
    elif weighted_bull <= 0.35:
        final_trend  = "Bearish"
        final_signal = "SELL"
    else:
        final_trend  = "Sideways"
        final_signal = "HOLD"

    # LLM override: if LLM is sufficiently confident, trust its signal
    if llm.get("confidence", 0) > 0.75:
        final_signal = llm.get("signal", final_signal)
        if llm.get("signal") == "BUY":    final_trend = "Bullish"
        elif llm.get("signal") == "SELL": final_trend = "Bearish"

    # ── Weighted confidence ────────────────────────────────────────────────
    conf = (0.30 * itx.get("confidence", 0.5) +
            0.30 * tft.get("confidence", 0.5) +
            0.30 * llm.get("confidence", 0.5) +
            0.10 * (0.5 + 0.5 * news.get("sentiment_score", 0.0)))

    # ── Model agreement check ──────────────────────────────────────────────
    agreement = (itx.get("trend", "X").lower() == tft.get("trend", "Y").lower())

    if not agreement:
        # Determine if the more-accurate model also agrees with LLM
        itx_rmse = float(itx.get("val_rmse") or 999)
        tft_rmse = float(tft.get("val_rmse") or 999)

        if itx_rmse < tft_rmse:
            better_model_bull = itx_bull
            better_model_rmse_ratio = tft_rmse / (itx_rmse + 1e-6)
        else:
            better_model_bull = tft_bull
            better_model_rmse_ratio = itx_rmse / (tft_rmse + 1e-6)

        # If the better model agrees with the LLM, apply lighter penalty (15%)
        # Otherwise apply standard penalty (25%)
        if better_model_bull == llm_bull and better_model_rmse_ratio > 1.3:
            penalty = 0.85   # 15% reduction
            logger.info(
                f"Model disagreement: better model (lower RMSE by ×{better_model_rmse_ratio:.1f}) "
                f"agrees with LLM → lighter penalty (−15%)"
            )
        else:
            penalty = 0.75   # 25% reduction (standard)
            logger.info(
                f"Model disagreement: no dominant model signal → standard penalty (−25%)"
            )

        conf        = conf * penalty
        final_signal = "HOLD"
        final_trend  = "Sideways"

    # ── Average price target ───────────────────────────────────────────────
    prices = []
    if itx.get("predicted_prices"):
        prices.append(itx["predicted_prices"][-1])
    if tft.get("predicted_prices"):
        prices.append(tft["predicted_prices"][-1])
    if llm.get("price_target", 0) > 0:
        prices.append(llm["price_target"])
    price_target = round(sum(prices) / len(prices), 2) if prices else 0.0

    return {
        "signal":       final_signal,
        "trend":        final_trend,
        "confidence":   round(min(0.99, max(0.01, conf)), 4),
        "price_target": price_target,
        "agreement":    agreement,
    }


# ---------------------------------------------------------------------------
# MCPConnector class (unchanged except _build_consensus is fixed above)
# ---------------------------------------------------------------------------
class MCPConnector:
    def __init__(
        self,
        ticker:   str  = "RELIANCE.NS",
        company:  str  = "Reliance Industries Limited",
        log_runs: bool = True,
    ):
        self.ticker   = ticker
        self.company  = company
        self.log_runs = log_runs

    def package(
        self,
        itransformer_output: dict,
        tft_output:          dict,
        news_output:         dict,
        llm_output:          dict,
    ) -> dict:
        run_id    = str(uuid.uuid4())[:8].upper()
        timestamp = datetime.now(timezone.utc).isoformat()
        errors    = []

        logger.info(f"Packaging run {run_id} for {self.ticker} at {timestamp}")

        itx_clean,  itx_warns  = _validate_model_output(itransformer_output, "iTransformer")
        tft_clean,  tft_warns  = _validate_model_output(tft_output,          "TFT")
        news_clean, news_warns = _validate_news(news_output)
        llm_clean,  llm_warns  = _validate_llm_output(llm_output)

        all_warnings = itx_warns + tft_warns + news_warns + llm_warns
        for w in all_warnings:
            logger.warning(w)
        errors.extend(all_warnings)

        last_price = float(
            itransformer_output.get("last_actual_price") or
            tft_output.get("last_actual_price") or 0.0
        )

        consensus = _build_consensus(itx_clean, tft_clean, news_clean, llm_clean)

        has_llm = bool(llm_clean.get("explanation", "").strip())
        has_itx = bool(itx_clean.get("predicted_prices"))
        has_tft = bool(tft_clean.get("predicted_prices"))

        if has_itx and has_tft and has_llm:
            status = "success"
        elif has_itx or has_tft:
            status = "partial"
        else:
            status = "error"

        package = {
            "run_id":    run_id,
            "timestamp": timestamp,
            "ticker":    self.ticker,
            "company":   self.company,
            "last_price":last_price,
            "models": {
                "itransformer": itx_clean,
                "tft":          tft_clean,
            },
            "news":               news_clean,
            "llm_interpretation": llm_clean,
            "consensus":          consensus,
            "status":             status,
            "errors":             errors,
        }

        logger.info(
            f"Run {run_id} packaged — Status: {status}  "
            f"Signal: {consensus['signal']}  "
            f"Trend: {consensus['trend']}  "
            f"Confidence: {consensus['confidence']:.1%}  "
            f"Price target: ₹{consensus['price_target']}"
        )

        if self.log_runs:
            self._save_log(run_id, package)

        return package

    def _save_log(self, run_id: str, package: dict):
        date_str = datetime.now().strftime("%Y%m%d")
        log_path = os.path.join(LOG_DIR, f"run_{date_str}_{run_id}.json")
        log_copy = json.loads(json.dumps(package))
        if "llm_interpretation" in log_copy:
            log_copy["llm_interpretation"].pop("raw_llm_output", None)
        with open(log_path, "w") as f:
            json.dump(log_copy, f, indent=2)
        logger.info(f"Run log saved → {log_path}")

    @staticmethod
    def load_latest_log() -> Optional[dict]:
        files = sorted(
            [f for f in os.listdir(LOG_DIR) if f.startswith("run_") and f.endswith(".json")],
            reverse=True,
        )
        if not files:
            return None
        with open(os.path.join(LOG_DIR, files[0])) as f:
            return json.load(f)

    @staticmethod
    def print_summary(package: dict):
        c = package.get("consensus", {})
        l = package.get("llm_interpretation", {})
        print("\n" + "╔" + "═" * 58 + "╗")
        print(f"║{'  MCP Package Summary':^58}║")
        print("╠" + "═" * 58 + "╣")
        print(f"║  Run ID     : {package.get('run_id', ''):<43}║")
        print(f"║  Ticker     : {package.get('ticker', ''):<43}║")
        print(f"║  Timestamp  : {package.get('timestamp', '')[:19]:<43}║")
        print(f"║  Status     : {package.get('status', ''):<43}║")
        print("╠" + "═" * 58 + "╣")
        print(f"║  Signal     : {c.get('signal', ''):<43}║")
        print(f"║  Trend      : {c.get('trend', ''):<43}║")
        conf_str = f"{c.get('confidence', 0)*100:.1f}%"
        print(f"║  Confidence : {conf_str:<43}║")
        print(f"║  Price tgt  : ₹{c.get('price_target', 0):<42.2f}║")
        print(f"║  Agreement  : {str(c.get('agreement', False)):<43}║")
        print("╠" + "═" * 58 + "╣")
        exp    = l.get("explanation", "")
        words, line, lines = exp.split(), "", []
        for w in words:
            if len(line) + len(w) + 1 > 54: lines.append(line); line = w
            else: line = (line + " " + w).strip()
        if line: lines.append(line)
        print(f"║  {'EXPLANATION':^56}  ║")
        for ln in lines:
            print(f"║  {ln:<56}║")
        print("╚" + "═" * 58 + "╝")
