"""
llm/ollama_interpreter.py
==========================
LLM Interpretation Agent — uses a locally running Ollama model via LangChain
to synthesise predictions from iTransformer, TFT, and the News Sentiment agent
into a human-readable investment interpretation.

CONFIDENCE DESIGN (important to understand)
--------------------------------------------
There are TWO distinct confidence metrics in this system:

  1. model.confidence  (per-agent)
       = 1 - 2 × (val_MAE / price_range)
       Meaning: "how accurately does this model track historical prices?"
       High value (88%, 93%) means low prediction error — NOT that the model
       is certain about tomorrow's direction.

  2. system.confidence  (final signal)
       Reflects how strongly all signals agree on a BUY/SELL/HOLD direction.
       Computed here and in mcp_connector._build_consensus.
       Naturally lower when models predict opposite directions.

BUG FIX (v2):
  Previously _compute_consensus applied a 20% confidence penalty for
  model disagreement, and mcp_connector._build_consensus applied a FURTHER
  25% penalty — the disagreement was punished twice, producing an
  artificially low final confidence.

  Fix: _compute_consensus no longer modifies combined_conf.
  It still force-HOLD the pre_signal and informs the LLM prompt clearly.
  mcp_connector applies the single penalty that matters.

SIGNAL STRENGTH:
  signal_strength = (predicted_D5 − last_price) / val_rmse
  Normalises the model's directional bet by its own error margin.
  A model predicting +₹30 with RMSE ₹100 is less convincing than one
  predicting +₹80 with RMSE ₹60.
"""

import os
import sys
import json
import re
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from langchain_community.llms import Ollama
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    _LANGCHAIN_OK = True
except ImportError:
    _LANGCHAIN_OK = False
    print("[LLMInterpreter] langchain-community not found. "
          "Install: pip install langchain-community")


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_SYSTEM_CONTEXT = """You are an expert quantitative financial analyst specialising \
in Indian equity markets. You have access to two deep learning model predictions \
and a news sentiment analysis for a stock. Synthesise all signals and provide a \
clear, structured investment interpretation. Be precise and data-driven. \
Do NOT hallucinate prices or facts not given to you."""

_ANALYSIS_TEMPLATE = """{system_context}

══════════════════════════════════════════
STOCK ANALYSIS REQUEST
══════════════════════════════════════════
Ticker   : {ticker}
Company  : {company}
Date     : {date}
Current  : ₹{last_price:.2f}

──────────────────────────────────────────
MODEL AGREEMENT STATUS
──────────────────────────────────────────
Models agree on direction : {models_agree}
Combined rule-based confidence (pre-penalty) : {combined_conf_pct}

IMPORTANT:
  • When models DISAGREE, default to HOLD unless one model has
    a substantially lower validation error (MAE/RMSE) than the other.
  • signal_strength normalises each model's directional bet by its
    own RMSE — values >0.5 are convincing, <0.3 are marginal.

──────────────────────────────────────────
MODEL 1 — iTransformer
──────────────────────────────────────────
Predicted 5-day prices : {itx_prices}
Predicted trend        : {itx_trend}
Prediction accuracy    : {itx_conf:.1%}  (historical MAE-based)
Signal strength (±σ)   : {itx_signal_strength:+.2f}
Key driving features   : {itx_features}
Validation MAE         : ₹{itx_mae:.2f}
Validation RMSE        : ₹{itx_rmse:.2f}

──────────────────────────────────────────
MODEL 2 — Temporal Fusion Transformer (TFT)
──────────────────────────────────────────
Predicted 5-day prices : {tft_prices}
Predicted trend        : {tft_trend}
Prediction accuracy    : {tft_conf:.1%}  (historical MAE-based)
Signal strength (±σ)   : {tft_signal_strength:+.2f}
Key driving features   : {tft_features}
Validation MAE         : ₹{tft_mae}
Validation RMSE        : ₹{tft_rmse}

──────────────────────────────────────────
NEWS SENTIMENT ANALYSIS
──────────────────────────────────────────
Overall sentiment : {news_label} (score: {news_score:+.3f}, range -1 to +1)
Articles analysed : {news_count}
Recent headlines  : {news_summary}

══════════════════════════════════════════
INSTRUCTIONS
══════════════════════════════════════════
Based on ALL signals above, provide your analysis in EXACTLY this JSON format.
Return ONLY the JSON — no markdown, no preamble, no extra text.

{{
  "signal": "<BUY | SELL | HOLD>",
  "final_trend": "<Bullish | Bearish | Sideways>",
  "confidence": <float between 0.0 and 1.0 — your own estimate of signal reliability>,
  "price_target": <predicted price on day 5, float>,
  "explanation": "<2-3 sentence professional explanation of your recommendation>",
  "model_agreement": <true | false>,
  "key_risks": "<one sentence about the main risk to this prediction>"
}}"""


# ---------------------------------------------------------------------------
# Signal strength helper
# ---------------------------------------------------------------------------
def _signal_strength(predicted_prices: list, last_price: float, val_rmse: float) -> float:
    """
    Compute normalised directional signal strength.
    Returns (predicted_D5 - last_price) / val_rmse.
    Positive = bullish, Negative = bearish.
    |value| > 0.5 → convincing,  |value| < 0.3 → marginal.
    Returns 0.0 if data is missing or invalid.
    """
    if not predicted_prices or last_price <= 0 or val_rmse <= 0:
        return 0.0
    try:
        d5 = float(predicted_prices[-1])
        return round((d5 - float(last_price)) / float(val_rmse), 3)
    except (TypeError, ZeroDivisionError):
        return 0.0


# ---------------------------------------------------------------------------
# Rule-based consensus (BUG-FIXED: no double penalty)
# ---------------------------------------------------------------------------
def _compute_consensus(payload: dict) -> dict:
    """
    Rule-based pre-processing: compute model agreement, weighted confidence,
    and a consensus price target BEFORE sending to the LLM.

    CHANGE FROM v1: The 20% disagreement penalty has been REMOVED from
    combined_conf. The penalty now lives only in mcp_connector._build_consensus
    so disagreement is penalised exactly ONCE across the whole pipeline.

    The pre_signal is still forced to HOLD on disagreement — this is the
    correct conservative default and does not affect the numeric confidence.
    """
    itx  = payload.get("itransformer", {})
    tft  = payload.get("tft", {})
    news = payload.get("news", {})

    # ── Model agreement ────────────────────────────────────────────────────
    itx_bullish  = itx.get("trend", "").lower() == "bullish"
    tft_bullish  = tft.get("trend", "").lower() == "bullish"
    models_agree = (itx_bullish == tft_bullish)

    # ── Signal strength per model ──────────────────────────────────────────
    last_price = float(itx.get("last_actual_price") or tft.get("last_actual_price") or 0)
    itx_ss = _signal_strength(
        itx.get("predicted_prices", []), last_price, itx.get("val_rmse") or 1.0
    )
    tft_ss = _signal_strength(
        tft.get("predicted_prices", []), last_price, tft.get("val_rmse") or 1.0
    )

    # ── Prediction-accuracy-based confidence (no penalty here) ────────────
    itx_conf = float(itx.get("confidence", 0.5))
    tft_conf = float(tft.get("confidence", 0.5))
    model_avg = (itx_conf + tft_conf) / 2.0

    # News sentiment modifies confidence ±5%
    news_modifier = 0.05 * float(news.get("sentiment_score", 0.0))
    combined_conf = float(min(0.99, max(0.05, model_avg + news_modifier)))

    # ── Consensus price target (day-5 average of both models) ─────────────
    prices = [p for p in [
        itx.get("predicted_prices", [None])[-1],
        tft.get("predicted_prices", [None])[-1],
    ] if p is not None]
    price_target = round(sum(prices) / len(prices), 2) if prices else 0.0

    # ── Pre-signal (conservative HOLD when models disagree) ───────────────
    # Note: We count bullish signals from both models and news
    bullish_votes = sum([
        int(itx_bullish),
        int(tft_bullish),
        int(news.get("sentiment_score", 0.0) > 0.05),
    ])

    if not models_agree:
        pre_signal = "HOLD"   # Force HOLD, do NOT modify combined_conf
    elif bullish_votes >= 2:
        pre_signal = "BUY"
    elif bullish_votes == 0:
        pre_signal = "SELL"
    else:
        pre_signal = "HOLD"

    return {
        "models_agree":   models_agree,
        "combined_conf":  combined_conf,   # unpenalised — penalty is in MCP
        "pre_signal":     pre_signal,
        "price_target":   price_target,
        "bullish_votes":  bullish_votes,
        "itx_signal_strength": itx_ss,
        "tft_signal_strength": tft_ss,
    }


# ---------------------------------------------------------------------------
# OllamaInterpreter
# ---------------------------------------------------------------------------
class OllamaInterpreter:
    """
    Sends all agent outputs to a local Ollama model and returns a structured
    investment interpretation.
    """

    def __init__(
        self,
        model_name:       str  = "llama3.2",
        base_url:         str  = "http://localhost:11434",
        temperature:      float = 0.1,
        max_retries:      int  = 3,
        fallback_on_error:bool = True,
    ):
        self.model_name       = model_name
        self.base_url         = base_url
        self.temperature      = temperature
        self.max_retries      = max_retries
        self.fallback_on_error = fallback_on_error
        self._chain           = None

        if _LANGCHAIN_OK:
            try:
                llm = Ollama(
                    model       = model_name,
                    base_url    = base_url,
                    temperature = temperature,
                )
                prompt  = PromptTemplate.from_template(_ANALYSIS_TEMPLATE)
                self._chain = prompt | llm | StrOutputParser()
                print(f"[LLMInterpreter] Chain built with model='{model_name}' "
                      f"at {base_url}")
            except Exception as e:
                print(f"[LLMInterpreter] Chain build failed: {e}")

    # ── Build prompt variables ─────────────────────────────────────────────
    @staticmethod
    def _build_prompt_vars(payload: dict, consensus: dict) -> dict:
        itx  = payload.get("itransformer", {})
        tft  = payload.get("tft", {})
        news = payload.get("news", {})

        last_price = float(
            itx.get("last_actual_price") or tft.get("last_actual_price") or 0.0
        )

        def _rmse_str(val):
            return f"{val:.2f}" if isinstance(val, (int, float)) and val > 0 else "N/A"
        def _mae_str(val):
            return f"{val:.2f}" if isinstance(val, (int, float)) and val > 0 else "N/A"

        return {
            "system_context":       _SYSTEM_CONTEXT,
            "ticker":               payload.get("ticker",  "UNKNOWN"),
            "company":              payload.get("company", "Unknown"),
            "date":                 payload.get("date",    ""),
            "last_price":           last_price,
            "models_agree":         str(consensus["models_agree"]),
            "combined_conf_pct":    f"{consensus['combined_conf']:.1%}",
            # iTransformer
            "itx_prices":           str(itx.get("predicted_prices", [])),
            "itx_trend":            itx.get("trend", "Unknown"),
            "itx_conf":             float(itx.get("confidence", 0.5)),
            "itx_signal_strength":  consensus["itx_signal_strength"],
            "itx_features":         str(itx.get("important_features", [])),
            "itx_mae":              float(itx.get("val_mae") or 0.0),
            "itx_rmse":             float(itx.get("val_rmse") or 0.0),
            # TFT
            "tft_prices":           str(tft.get("predicted_prices", [])),
            "tft_trend":            tft.get("trend", "Unknown"),
            "tft_conf":             float(tft.get("confidence", 0.5)),
            "tft_signal_strength":  consensus["tft_signal_strength"],
            "tft_features":         str(tft.get("important_features", [])),
            "tft_mae":              _mae_str(tft.get("val_mae")),
            "tft_rmse":             _rmse_str(tft.get("val_rmse")),
            # News
            "news_label":           news.get("sentiment_label",  "Neutral"),
            "news_score":           float(news.get("sentiment_score", 0.0)),
            "news_count":           news.get("num_articles", 0),
            "news_summary":         news.get("headline_summary", "No headlines."),
        }

    # ── Parse LLM JSON response ────────────────────────────────────────────
    @staticmethod
    def _parse_llm_response(raw: str) -> Optional[dict]:
        patterns = [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
            r"(\{.*?\})",
        ]
        text = raw.strip()
        for pat in patterns:
            match = re.search(pat, text, re.DOTALL)
            if match:
                text = match.group(1).strip()
                break

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(raw.strip())
            except json.JSONDecodeError:
                return None

        required = {"signal", "final_trend", "confidence",
                    "price_target", "explanation", "model_agreement"}
        if not required.issubset(parsed.keys()):
            return None

        parsed["signal"]          = str(parsed["signal"]).upper()
        parsed["final_trend"]     = str(parsed["final_trend"]).capitalize()
        parsed["confidence"]      = float(parsed.get("confidence", 0.5))
        parsed["price_target"]    = float(parsed.get("price_target", 0.0))
        parsed["model_agreement"] = bool(parsed.get("model_agreement", False))
        return parsed

    # ── Rule-based fallback ────────────────────────────────────────────────
    @staticmethod
    def _fallback(payload: dict, consensus: dict) -> dict:
        itx  = payload.get("itransformer", {})
        tft  = payload.get("tft", {})
        news = payload.get("news", {})

        trend_map   = {3: "Bullish", 2: "Bullish", 1: "Sideways", 0: "Bearish"}
        final_trend = trend_map.get(consensus["bullish_votes"], "Sideways")
        if not consensus["models_agree"]:
            final_trend = "Sideways"

        itx_ss = consensus["itx_signal_strength"]
        tft_ss = consensus["tft_signal_strength"]
        explanation = (
            f"iTransformer predicts a {itx.get('trend','unknown')} trend "
            f"(signal strength {itx_ss:+.2f}σ) while TFT predicts "
            f"{tft.get('trend','unknown')} (signal strength {tft_ss:+.2f}σ). "
            f"{'Model disagreement warrants caution — HOLD is prudent. ' if not consensus['models_agree'] else ''}"
            f"News sentiment is {news.get('sentiment_label','Neutral').lower()} "
            f"(score {news.get('sentiment_score',0):+.2f})."
        )

        return {
            "signal":          consensus["pre_signal"],
            "final_trend":     final_trend,
            "confidence":      consensus["combined_conf"],
            "price_target":    consensus["price_target"],
            "explanation":     explanation,
            "model_agreement": consensus["models_agree"],
            "key_risks":       "Model disagreement and market volatility are the primary risks.",
            "raw_llm_output":  "[FALLBACK — LLM not available]",
        }

    # ── Main interpret method ──────────────────────────────────────────────
    def interpret(self, payload: dict) -> dict:
        """
        Synthesise all agent outputs into a final LLM interpretation.
        """
        print("[LLMInterpreter] Computing rule-based consensus …")
        consensus   = _compute_consensus(payload)
        prompt_vars = self._build_prompt_vars(payload, consensus)

        print(f"[LLMInterpreter] Pre-signal: {consensus['pre_signal']}  "
              f"Models agree: {consensus['models_agree']}  "
              f"Combined accuracy: {consensus['combined_conf']:.1%}")
        print(f"[LLMInterpreter] Signal strength — "
              f"iTx: {consensus['itx_signal_strength']:+.2f}σ  "
              f"TFT: {consensus['tft_signal_strength']:+.2f}σ")

        if not consensus["models_agree"]:
            print("[LLMInterpreter] ⚠️  Models DISAGREE — pre-signal forced to HOLD. "
                  "(Single disagreement penalty will be applied by MCP connector.)")

        if self._chain is not None:
            for attempt in range(1, self.max_retries + 1):
                try:
                    print(f"[LLMInterpreter] Calling Ollama '{self.model_name}' "
                          f"(attempt {attempt}/{self.max_retries}) …")
                    t0      = time.time()
                    raw_out = self._chain.invoke(prompt_vars)
                    elapsed = time.time() - t0
                    print(f"[LLMInterpreter] LLM responded in {elapsed:.1f}s")

                    parsed = self._parse_llm_response(raw_out)
                    if parsed is not None:
                        parsed["raw_llm_output"] = raw_out
                        # Blend LLM's own confidence with rule-based average
                        # (70% LLM, 30% rule-based) for robustness
                        parsed["confidence"] = round(
                            0.7 * parsed["confidence"] + 0.3 * consensus["combined_conf"], 4
                        )
                        print(f"[LLMInterpreter] Signal={parsed['signal']}  "
                              f"Trend={parsed['final_trend']}  "
                              f"LLM confidence (pre-MCP-penalty)="
                              f"{parsed['confidence']:.1%}")
                        return parsed
                    else:
                        print(f"[LLMInterpreter] JSON parse failed on attempt {attempt}. "
                              f"Raw: {raw_out[:300]}")

                except Exception as e:
                    print(f"[LLMInterpreter] LLM call error on attempt {attempt}: {e}")
                    time.sleep(2)

        if self.fallback_on_error:
            print("[LLMInterpreter] Using rule-based fallback.")
            return self._fallback(payload, consensus)
        else:
            raise RuntimeError("[LLMInterpreter] LLM failed and fallback is disabled.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mock_payload = {
        "ticker": "RELIANCE.NS", "company": "Reliance Industries Limited",
        "date":   "2026-04-19",
        "itransformer": {
            "predicted_prices":   [1287.26, 1291.74, 1282.80, 1305.47, 1330.74],
            "trend":              "Bearish", "confidence": 0.8804,
            "important_features": ["adj_close", "high", "bb_middle"],
            "val_mae": 86.51, "val_rmse": 103.04, "last_actual_price": 1365.0,
        },
        "tft": {
            "predicted_prices":   [1395.99, 1397.78, 1401.22, 1401.33, 1399.47],
            "trend":              "Bullish", "confidence": 0.9286,
            "important_features": ["open", "bb_lower", "low"],
            "val_mae": 51.64, "val_rmse": 64.25, "last_actual_price": 1365.0,
        },
        "news": {
            "sentiment_label": "Neutral", "sentiment_score": 0.032,
            "headline_summary": "Mixed signals for Reliance.", "num_articles": 30,
        },
    }
    interp = OllamaInterpreter(model_name="llama3.2", fallback_on_error=True)
    result = interp.interpret(mock_payload)
    print("\n── Result ──")
    for k, v in result.items():
        if k != "raw_llm_output":
            print(f"  {k:<22}: {v}")
