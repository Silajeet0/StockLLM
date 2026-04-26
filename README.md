# Stock Market LLM — AI-Powered Market Intelligence

> **Predict the next 5 trading days of Reliance Industries (RELIANCE.NS) using a dual-model AI pipeline — iTransformer + Temporal Fusion Transformer — enhanced with live news sentiment analysis and LLM-generated explanations.**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Ubuntu / Linux](#ubuntu--linux)
  - [Windows](#windows)
- [Installing Ollama + LLaMA 3.2](#installing-ollama--llama-32)
  - [Ubuntu / Linux](#ubuntu--linux-1)
  - [Windows](#windows-1)
- [Running the Project](#running-the-project)
  - [Ubuntu / Linux](#ubuntu--linux-2)
  - [Windows](#windows-2)
- [How It Works](#how-it-works)
- [Saved Models](#saved-models)
- [License](#license)

---

## Overview

Stock Market LLM is a predict-only pipeline that:

1. Downloads **15 years** of RELIANCE.NS OHLCV data fresh from Yahoo Finance on every run
2. Engineers **18 technical indicators** (RSI, MACD, Bollinger Bands, ATR, etc.)
3. Loads **pre-trained iTransformer and TFT checkpoints** from `saved_models/`
4. Predicts **closing prices for the next 5 trading days**
5. Scrapes **live news headlines** and computes sentiment
6. Passes everything through an **LLM Orchestrator** (via Ollama + LLaMA 3.2) to generate a final BUY / SELL / HOLD signal with a natural language explanation
7. Displays everything in a **Django web dashboard**

---

## Architecture

```
Yahoo Finance (15y OHLCV)
        ↓
Feature Engineering (18 indicators)
        ↓
   ┌────┴─────┐
   │          │
iTransformer  TFT          ← Both load from saved_models/
   │          │
   └────┬─────┘
        ↓
  News Sentiment (live scrape + NLP)
        ↓
  LLM Orchestrator (Ollama / LLaMA 3.2)
        ↓
  Final Signal: BUY / SELL / HOLD
  5-Day Price Forecast + Explanation
        ↓
  Django Web Dashboard
```

---

## Project Structure

```
stock-market-llm-v4/
├── main.py                        # Entry point — predict pipeline
├── start_gui.sh                   # Launch GUI (Ubuntu)
├── start_gui.bat                  # Launch GUI (Windows)
│
├── agents/
│   ├── itransformer_agent.py      # iTransformer inference agent
│   ├── tft_agent.py               # TFT inference agent
│   ├── news_agent.py              # News scraping + sentiment
│   └── orchestrator_agent.py      # Combines outputs → final signal
│
├── models/
│   ├── itransformer.py            # iTransformer model definition
│   ├── iTransformer.py            # Compatibility shim
│   └── tft_model.py               # Temporal Fusion Transformer
│
├── utils/
│   ├── data_loader.py             # yfinance downloader
│   ├── feature_engineering.py     # Technical indicators
│   ├── preprocessing.py           # Scaler / normalisation
│   └── sequence_builder.py        # Sliding window datasets
│
├── config/
│   └── settings.py                # All hyperparameters and paths
│
├── llm/
│   └── ollama_interpreter.py      # Ollama / LLaMA 3.2 interface
│
├── mcp/
│   └── mcp_connector.py           # Consensus engine
│
├── scraping/
│   └── news_scraper.py            # News headline fetcher
│
├── saved_models/                  # Pre-trained checkpoints
│   ├── reliance_ns_itransformer_best.pt
│   ├── reliance_ns_itransformer_scaler.pkl
│   ├── reliance_ns_tft_best.ckpt
│   ├── reliance_ns_tft_scaler.pkl
│   └── mcp_logs/
│
├── data/
│   ├── raw/                       # Downloaded OHLCV CSVs
│   └── processed/                 # Feature-engineered CSVs
│
└── gui/
    ├── manage.py
    ├── requirements_gui.txt
    ├── stockgui/                  # Django project settings
    └── dashboard/                 # Django app — views, templates
```

---

## Requirements

- Python 3.9 or higher
- CUDA-capable GPU recommended (CPU also works, will be slower)
- [Ollama](https://ollama.com) with **llama3.2** pulled (for LLM explanations)
- Internet connection (for Yahoo Finance data and news)

Install all Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Installation

### Ubuntu / Linux

```bash
# 1. Clone or extract the project
cd stock-llm

# 2. Create and activate a virtual environment (recommended)
python3 -m venv env
source env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install GUI dependencies
pip install -r gui/requirements_gui.txt
```

### Windows

```bat
:: 1. Open Command Prompt and navigate to the project folder
cd stock-market-llm-v4

:: 2. Create and activate a virtual environment (recommended)
python -m venv env
env\Scripts\activate

:: 3. Install dependencies
pip install -r requirements.txt

:: 4. Install GUI dependencies
pip install -r gui\requirements_gui.txt
```

---

## Installing Ollama + LLaMA 3.2

Ollama runs LLaMA 3.2 locally on your machine. It is required for the LLM Orchestrator to generate trade explanations. If Ollama is not running, the pipeline falls back to a rule-based summary automatically.

### Ubuntu / Linux

**Step 1 — Install Ollama:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Step 2 — Start the Ollama service:**

```bash
ollama serve
```

> Leave this terminal open, or run it in the background with `ollama serve &`

**Step 3 — Pull the LLaMA 3.2 model:**

```bash
ollama pull llama3.2
```

**Step 4 — Verify it works:**

```bash
ollama run llama3.2 "Hello, are you working?"
```

---

### Windows

**Step 1 — Download and install Ollama:**

Go to [https://ollama.com/download](https://ollama.com/download) and download the **Windows installer** (`.exe`). Run it and follow the setup wizard.

**Step 2 — Ollama starts automatically** as a background service after installation. You can verify it is running by checking the system tray.

**Step 3 — Pull the LLaMA 3.2 model:**

Open **Command Prompt** or **PowerShell** and run:

```bat
ollama pull llama3.2
```

**Step 4 — Verify it works:**

```bat
ollama run llama3.2 "Hello, are you working?"
```

> **Note:** LLaMA 3.2 requires approximately 2 GB of disk space. A GPU is not required — it runs on CPU too, though generation will be slower.

---

## Running the Project

### Ubuntu / Linux

**Option 1 — Web Dashboard (recommended):**

```bash
# From the project root (stock-market-llm-v4/)
bash start_gui.sh
```

Then open your browser and go to: **http://127.0.0.1:8000**

Click the **Predict** button. The pipeline will:
- Download fresh 15-year data
- Run both models
- Fetch news sentiment
- Display the 5-day forecast and BUY/SELL/HOLD signal

**Option 2 — Command line only (no browser):**

```bash
python main.py
```

Results are saved to `saved_models/reliance_ns_final_prediction.json`.

---

### Windows

**Option 1 — Web Dashboard (recommended):**

Double-click `start_gui.bat`, or run from Command Prompt:

```bat
start_gui.bat
```

Then open your browser and go to: **http://127.0.0.1:8000**

**Option 2 — Command line only:**

```bat
python main.py
```

> **Note:** Make sure your virtual environment is activated before running either option.

---

## How It Works

| Step | What happens |
|------|-------------|
| **Data Download** | Downloads RELIANCE.NS OHLCV from Yahoo Finance — start date is exactly 15 years before the latest available trading date |
| **Feature Engineering** | Computes RSI, MACD, Bollinger Bands, ATR, SMA/EMA, returns, volatility, and time features |
| **iTransformer** | Inverted Transformer — attention runs across the feature dimension rather than time. Loads checkpoint from `saved_models/` |
| **TFT** | Temporal Fusion Transformer with variable selection networks and interpretable attention. Loads checkpoint from `saved_models/` |
| **News Sentiment** | Scrapes recent Reliance news headlines, scores sentiment, caches for 24 hours |
| **Orchestrator** | Weighted consensus (iTransformer 40% + TFT 40% + News 20%), passed to LLaMA 3.2 via Ollama for natural language explanation |
| **Output** | BUY / SELL / HOLD signal, 5-day price table, confidence score, key risk, LLM explanation |

---

## Saved Models

The `saved_models/` folder contains pre-trained checkpoints:

| File | Description |
|------|-------------|
| `reliance_ns_itransformer_best.pt` | iTransformer weights |
| `reliance_ns_itransformer_scaler.pkl` | iTransformer feature scaler |
| `reliance_ns_tft_best.ckpt` | TFT weights (PyTorch Lightning checkpoint) |
| `reliance_ns_tft_scaler.pkl` | TFT feature scaler |

If a checkpoint is missing, the pipeline will **automatically train from scratch** on the first run using the downloaded 15-year dataset, then save the checkpoint for all future runs.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
