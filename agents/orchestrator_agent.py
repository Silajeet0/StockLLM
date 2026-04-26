"""
agents/orchestrator_agent.py
"""
import os, sys, json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TICKER, ITX_METRICS_PATH
from llm.ollama_interpreter import OllamaInterpreter
from mcp.mcp_connector import MCPConnector

class OrchestratorAgent:
    def __init__(self, ticker="RELIANCE.NS", company="Reliance Industries Limited",
                 ollama_model="llama3.2", ollama_url="http://localhost:11434", log_runs=True):
        self.ticker = ticker; self.company = company
        print("[Orchestrator] Initialising agents …")
        self.interpreter = OllamaInterpreter(model_name=ollama_model, base_url=ollama_url,
                                              temperature=0.1, fallback_on_error=True)
        self.connector = MCPConnector(ticker=ticker, company=company, log_runs=log_runs)
        print("[Orchestrator] Ready.")

    def run(self, itransformer_output, tft_output, news_output):
        date_str = datetime.now().strftime("%Y-%m-%d")
        print("\n" + "═"*60)
        print(f"  ORCHESTRATOR  |  {self.ticker}  |  {date_str}")
        print("═"*60)
        llm_payload = {"ticker": self.ticker, "company": self.company, "date": date_str,
                       "itransformer": itransformer_output, "tft": tft_output, "news": news_output}
        print("\n[Orchestrator] Step 1/2 — LLM Interpretation …")
        llm_output = self.interpreter.interpret(llm_payload)
        print("\n[Orchestrator] Step 2/2 — MCP Packaging & Validation …")
        package = self.connector.package(itransformer_output, tft_output, news_output, llm_output)
        final = self._build_final_output(package)
        self._print_final(final)
        return final

    def run_with_internal_itx(self, tft_output, news_output,
                               force_download=False, retrain=True):
        """
        Parameters
        ----------
        retrain : if False, loads the saved iTransformer checkpoint instead of training.
        """
        from agents.itransformer_agent import iTransformerAgent
        print("[Orchestrator] Running iTransformer agent internally …")
        itx_agent = iTransformerAgent()
        if retrain:
            itx_output = itx_agent.run(force_download=force_download)
        else:
            print("[Orchestrator] --no-retrain: loading saved iTransformer checkpoint.")
            itx_output = itx_agent.infer(force_download=force_download)
        return self.run(itx_output, tft_output, news_output)

    @staticmethod
    def _build_final_output(package):
        cons = package["consensus"]; llm = package["llm_interpretation"]
        itx  = package["models"]["itransformer"]; tft = package["models"]["tft"]
        news = package["news"]
        itx_prices = itx.get("predicted_prices", []); tft_prices = tft.get("predicted_prices", [])
        avg_prices = []
        for i in range(max(len(itx_prices), len(tft_prices))):
            vals = []
            if i < len(itx_prices): vals.append(itx_prices[i])
            if i < len(tft_prices): vals.append(tft_prices[i])
            avg_prices.append(round(sum(vals)/len(vals), 4) if vals else None)
        return {
            "run_id": package["run_id"], "timestamp": package["timestamp"],
            "ticker": package["ticker"], "company": package["company"],
            "last_price": package["last_price"],
            "signal": cons["signal"], "trend": cons["trend"],
            "confidence": cons["confidence"], "price_target": cons["price_target"],
            "predicted_prices": {"itransformer": itx_prices, "tft": tft_prices, "average": avg_prices},
            "explanation": llm.get("explanation", ""), "key_risks": llm.get("key_risks", ""),
            "model_agreement": cons["agreement"],
            "important_features": {"itransformer": itx.get("important_features", []),
                                   "tft": tft.get("important_features", [])},
            "news": {"sentiment_label": news.get("sentiment_label", "Neutral"),
                     "sentiment_score": news.get("sentiment_score", 0.0),
                     "headline_summary": news.get("headline_summary", "")},
            "metrics": {"itransformer_mae": itx.get("val_mae"),
                        "itransformer_rmse": itx.get("val_rmse")},
            "status": package["status"],
        }

    @staticmethod
    def _print_final(final):
        signal_emoji = {"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}.get(final["signal"],"⚪")
        trend_emoji  = {"Bullish":"📈","Bearish":"📉","Sideways":"➡️"}.get(final["trend"],"")
        print("\n"); print("╔"+"═"*62+"╗")
        print(f"║{'🎯  FINAL PREDICTION OUTPUT':^62}║")
        print("╠"+"═"*62+"╣")
        print(f"║  Ticker       : {final['ticker']:<45}║")
        print(f"║  Company      : {final['company']:<45}║")
        print(f"║  Date         : {final['timestamp'][:10]:<45}║")
        print(f"║  Last Price   : ₹{final['last_price']:<44.2f}║")
        print("╠"+"═"*62+"╣")
        print(f"║  Signal       : {signal_emoji}  {final['signal']:<42}║")
        print(f"║  Trend        : {trend_emoji}  {final['trend']:<42}║")
        print(f"║  Confidence   : {final['confidence']*100:<44.1f}%  ║")
        print(f"║  Price Target : ₹{final['price_target']:<44.2f}║")
        print("╠"+"═"*62+"╣")
        itx = final["predicted_prices"]["itransformer"]
        tft = final["predicted_prices"]["tft"]
        avg = final["predicted_prices"]["average"]
        print(f"║  {'Day':<5}  {'iTransformer':>14}  {'TFT':>14}  {'Average':>14}  ║")
        print(f"║  {'─'*5}  {'─'*14}  {'─'*14}  {'─'*14}  ║")
        for i in range(5):
            ip = f"₹{itx[i]:.2f}" if i < len(itx) else "─"
            tp = f"₹{tft[i]:.2f}" if i < len(tft) else "─"
            ap = f"₹{avg[i]:.2f}" if i < len(avg) and avg[i] else "─"
            print(f"║  D+{i+1:<3}  {ip:>14}  {tp:>14}  {ap:>14}  ║")
        print("╠"+"═"*62+"╣")
        ns = final["news"]
        print(f"║  News Sentiment : {ns['sentiment_label']} ({ns['sentiment_score']:+.2f}){'':<29}║")
        print("╠"+"═"*62+"╣")
        words, line, lines = final["explanation"].split(), "", []
        for w in words:
            if len(line)+len(w)+1 > 58: lines.append(line); line = w
            else: line = (line+" "+w).strip()
        if line: lines.append(line)
        print(f"║  {'EXPLANATION':^60}  ║")
        for ln in lines: print(f"║  {ln:<60}║")
        print("╠"+"═"*62+"╣")
        print(f"║  Key Risk : {final['key_risks'][:56]:<56}  ║")
        print(f"║  Models agree : {str(final['model_agreement']):<45}║")
        print(f"║  Status       : {final['status']:<45}║")
        print(f"║  Run ID       : {final['run_id']:<45}║")
        print("╚"+"═"*62+"╝")

    @staticmethod
    def save(final, path=None):
        if path is None:
            try:
                from config.settings import MODEL_DIR
                path = os.path.join(MODEL_DIR, "final_prediction.json")
            except ImportError:
                path = "final_prediction.json"
        with open(path, "w") as f:
            json.dump(final, f, indent=2)
        print(f"\n[Orchestrator] Final output saved → {path}")


if __name__ == "__main__":
    mock_tft = {"model":"TFT","predicted_prices":[1350.10,1360.45,1355.20,1365.80,1358.90],
                "trend":"Bullish","confidence":0.85,"important_features":["rsi","macd","volatility"],
                "last_actual_price":1343.3}
    mock_news = {"sentiment_label":"Positive","sentiment_score":0.35,
                 "headline_summary":"Reliance Q4 earnings beat estimates.","num_articles":12}
    orchestrator = OrchestratorAgent(ticker="RELIANCE.NS",company="Reliance Industries Limited")
    final = orchestrator.run_with_internal_itx(tft_output=mock_tft, news_output=mock_news, retrain=False)
    OrchestratorAgent.save(final)
