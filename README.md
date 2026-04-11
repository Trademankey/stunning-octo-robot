# 🏛️ Institutional-Grade Crypto Trading Bot

> **Multi-model ensemble system** with 80+ features, walk-forward validation, Monte Carlo stress-testing, real-time WebSocket execution, and full observability — engineered for professional crypto trading.

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  CCXT-Pro WebSocket (OHLCV, Orderbook, Trades)                  │
│  Historical Loader (Exchange REST + TimescaleDB cache)          │
│  On-chain: Glassnode / Dune (funding, OI, whale flows, MVRV)   │
│  Sentiment: NewsAPI + Reddit + Twitter/X (FinBERT + VADER)      │
├─────────────────────────────────────────────────────────────────┤
│                     FEATURE ENGINE (80+)                        │
│  TA-Lib + pandas-ta: SuperTrend, Ichimoku, MACD, RSI, BB, ATR  │
│  Statistical: z-score, skew, kurtosis, ADF, entropy, Hurst, FD │
│  Orderbook: imbalance, depth, spread, pressure, whale detection │
│  Regime: HMM (3-state), Isolation Forest, vol-clustering        │
│  Latent: PCA, UMAP, Autoencoder embeddings                     │
├─────────────────────────────────────────────────────────────────┤
│                      MODEL ZOO                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  XGBoost +   │  │  LSTM +      │  │  RL Agent    │          │
│  │  LightGBM +  │  │  Multi-Head  │  │  PPO / SAC   │          │
│  │  CatBoost    │  │  Attention   │  │  Custom Gym  │          │
│  │  (Optuna)    │  │  (MC Drop)   │  │  (Sharpe rwd)│          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         └──────────────────┼─────────────────┘                  │
│                   ┌────────┴────────┐                           │
│                   │  Ridge Stacking │                           │
│                   │  Meta-Learner   │                           │
│                   └────────┬────────┘                           │
│                            │ signal ∈ [-1, +1]                  │
├────────────────────────────┼────────────────────────────────────┤
│                     STRATEGY LAYER                              │
│  Signal Generator: regime + sentiment + liquidity + on-chain    │
│  Position Sizer: min(Kelly, VolTarget, ATR-risk)                │
│  Risk Manager: kill-switch, trailing stops, correlation filter  │
├─────────────────────────────────────────────────────────────────┤
│                    EXECUTION LAYER                              │
│  Paper Trader (slippage + fee sim) │ Live Trader (CCXT)        │
│  TWAP / VWAP Smart Order Routing │ Circuit Breaker              │
├─────────────────────────────────────────────────────────────────┤
│                   INFRASTRUCTURE                                │
│  PostgreSQL + TimescaleDB │ Redis │ Prometheus + Grafana        │
│  Telegram Bot (commands + alerts) │ Health Checks + Auto-restart│
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Docker** & **Docker Compose** v2+
- **Python 3.11+** (for local development)
- Exchange API keys (Binance recommended)

### 1. Clone & Configure

```bash
git clone <your-repo-url> crypto_trading_bot
cd crypto_trading_bot

# Copy env template and fill in your secrets
cp .env.example .env
nano .env   # ← Fill in API keys, Telegram token, etc.
```

### 2. One-Click Docker Deploy

```bash
# Build and start everything (bot + Postgres + Redis + Prometheus + Grafana)
docker compose up -d --build

# Check logs
docker compose logs -f bot

# Stop everything
docker compose down
```

### 3. Local Development

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies (requires TA-Lib C library)
# On Ubuntu: sudo apt-get install -y ta-lib
# On macOS:  brew install ta-lib
pip install -r requirements.txt

# Run paper trading
python main.py --mode paper

# Run backtest
python main.py --mode backtest

# Train models only
python main.py --mode train

# Run tests
pytest tests/ -v
```

---

## 🧪 Backtesting

### One-Click Backtest

```bash
python main.py --mode backtest
```

This will:
1. Fetch 365 days of historical data for all pairs
2. Compute 80+ features per bar
3. Run 5-fold walk-forward validation
4. Generate full metrics report (Sharpe, Sortino, Calmar, Omega, Max DD, etc.)
5. Run 1000 Monte Carlo simulations
6. Print a comprehensive performance report

### Sample Output

```
╔══════════════════════════════════════════════════╗
║          BACKTEST PERFORMANCE REPORT              ║
╠══════════════════════════════════════════════════╣
║  Total Return:           23.47%                  ║
║  Annual Return:          18.92%                  ║
║  Max Drawdown:            8.34%                  ║
║  Final Equity:       $12,347.00                  ║
╠══════════════════════════════════════════════════╣
║  Sharpe Ratio:            1.847                  ║
║  Sortino Ratio:           2.691                  ║
║  Calmar Ratio:            2.269                  ║
║  Omega Ratio:             1.342                  ║
╠══════════════════════════════════════════════════╣
║  Total Trades:              387                  ║
║  Win Rate:               56.33%                  ║
║  Profit Factor:           1.487                  ║
║  Expectancy:             $6.07                   ║
║  SQN:                     3.214                  ║
╚══════════════════════════════════════════════════╝

🎲 Monte Carlo (1000 sims):
   Profit probability:    78.4%
   Ruin probability:       0.3%
   95th pctl Max DD:      14.21%
   5th pctl final equity: $10,847.00
```

> ⚠️ **Realistic expectations:** Backtested metrics assume no look-ahead bias
> via walk-forward, but live slippage, latency, and regime shifts will degrade
> performance. A backtest Sharpe of 1.5–2.0 typically realises at 0.8–1.2 live.

---

## 📁 Project Structure

```
crypto_trading_bot/
├── main.py                      # Entry point — paper/live/backtest/train
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Multi-stage Docker build
├── docker-compose.yml           # Full infra stack
├── .env.example                 # Environment variables template
│
├── config/
│   ├── settings.py              # Pydantic settings (all config from .env)
│   ├── pairs.py                 # Pair universe + correlation groups
│   └── init_db.sql              # TimescaleDB schema
│
├── data/
│   ├── websocket_feed.py        # CCXT-Pro WebSocket (OHLCV, OB, trades)
│   ├── historical.py            # Historical OHLCV loader + caching
│   ├── db.py                    # Async PostgreSQL CRUD (asyncpg)
│   ├── onchain.py               # Glassnode + Dune on-chain metrics
│   └── sentiment.py             # NewsAPI + Reddit + Twitter + FinBERT
│
├── features/
│   ├── technical.py             # 40+ TA indicators (TA-Lib + pandas-ta)
│   ├── statistical.py           # z-score, skew, kurtosis, ADF, entropy
│   ├── orderbook.py             # Microstructure features (imbalance, depth)
│   ├── regime.py                # HMM, Isolation Forest, vol-regime
│   ├── dimensionality.py        # PCA, UMAP, Autoencoder latents
│   └── pipeline.py              # Master feature orchestrator (80+ features)
│
├── models/
│   ├── tft_model.py             # Temporal Fusion Transformer (PyTorch)
│   ├── lstm_attention.py        # BiLSTM + Multi-Head Attention + MC Dropout
│   ├── gradient_boost.py        # XGBoost + LightGBM + CatBoost (Optuna)
│   ├── rl_agent.py              # PPO/SAC in custom Gym env (SB3)
│   ├── ensemble.py              # Stacking meta-learner ensemble
│   └── trainer.py               # Training orchestrator + online retraining
│
├── strategies/
│   ├── signal_generator.py      # Signal fusion + regime/sentiment/OB filters
│   ├── position_sizer.py        # Kelly + vol-targeting + ATR sizing
│   └── risk_manager.py          # Kill-switch, trailing stops, correlation
│
├── execution/
│   ├── order_manager.py         # Dispatch to paper/live + circuit breaker
│   ├── paper_trader.py          # Simulated execution (slippage + fees)
│   ├── live_trader.py           # Real CCXT execution (hedge mode, leverage)
│   └── smart_router.py          # TWAP / VWAP order splitting
│
├── backtest/
│   ├── engine.py                # Full backtest pipeline orchestrator
│   ├── walk_forward.py          # Walk-forward + Monte Carlo + stress tests
│   └── metrics.py               # Sharpe, Sortino, Calmar, Omega, SQN, etc.
│
├── monitoring/
│   ├── telegram_bot.py          # Alerts + /status /stop /resume commands
│   ├── prometheus_metrics.py    # Prometheus gauges, counters, histograms
│   ├── health_check.py          # DB/Redis/memory health probes
│   ├── prometheus.yml           # Prometheus scrape config
│   └── grafana_dashboards/
│       └── dashboard.json       # Pre-built Grafana dashboard
│
└── tests/
    ├── test_features.py         # Feature engineering tests
    ├── test_strategies.py       # Signal, sizing, risk tests
    └── test_backtest.py         # Backtest engine + metrics tests
```

---

## ⚙️ Configuration

All configuration flows through `.env` → Pydantic settings. Key knobs:

| Variable | Description | Default |
|---|---|---|
| `TRADING_MODE` | `paper` / `live` / `backtest` | `paper` |
| `TRADING_PAIRS` | Comma-separated pairs | `BTC/USDT,ETH/USDT,...` |
| `MARKET_TYPE` | `spot` / `futures` | `futures` |
| `MAX_RISK_PER_TRADE` | Maximum risk per trade | `0.01` (1%) |
| `MAX_DAILY_DRAWDOWN` | Kill-switch threshold | `0.05` (5%) |
| `KELLY_FRACTION` | Fraction of full Kelly to use | `0.25` |
| `RETRAIN_INTERVAL_HOURS` | Online retraining frequency | `8` |
| `OPTUNA_TRIALS` | Bayesian HP search trials | `500` |
| `EXCHANGE_SANDBOX` | Use testnet | `true` |

---

## 📊 Adding New Pairs

1. Edit `.env`:
   ```
   TRADING_PAIRS=BTC/USDT,ETH/USDT,SOL/USDT,DOGE/USDT,AVAX/USDT
   ```

2. Add pair config in `config/pairs.py`:
   ```python
   DEFAULT_PAIRS["DOGE/USDT"] = PairConfig(
       symbol="DOGE/USDT", base="DOGE",
       max_leverage=5.0, tick_size=0.0001, lot_size=10,
       min_notional=10.0, correlation_group="meme_group",
   )
   ```

3. Restart the bot — it auto-fetches history and retrains.

---

## 🔬 Adding New Models

1. Create `models/your_model.py` with a `train()` + `predict()` interface
2. Register in `models/ensemble.py`:
   ```python
   self.your_model = YourModel()
   # Add to _model_weights and predict() blending
   ```
3. The ensemble meta-learner will automatically learn optimal weighting

---

## 📡 Monitoring

| Service | URL | Credentials |
|---|---|---|
| **Grafana** | `http://localhost:3000` | admin / admin |
| **Prometheus** | `http://localhost:9090` | — |
| **Bot Metrics** | `http://localhost:8000/metrics` | — |
| **Telegram** | DM your bot | `/help` for commands |

### Telegram Commands

| Command | Action |
|---|---|
| `/status` | Current positions, equity, P&L |
| `/metrics` | Win rate, expectancy, profit factor |
| `/stop` | Emergency kill-switch (halt all trading) |
| `/resume` | Resume trading after kill-switch |
| `/help` | List all commands |

---

## 🛡️ Risk Management

| Control | Description |
|---|---|
| **Daily Drawdown Kill-Switch** | Halts all trading if daily DD > 5% |
| **Max Risk Per Trade** | 1% of equity per trade (configurable) |
| **Fractional Kelly** | 25% of full Kelly criterion |
| **Volatility Targeting** | 15% annualised target vol |
| **Correlation Filter** | Max 3 correlated pairs simultaneously |
| **ATR Trailing Stops** | Dynamic stops based on 2.5× ATR |
| **Circuit Breaker** | Pauses after 5 consecutive execution failures |
| **MC Dropout Uncertainty** | Reduces size when model confidence is low |
| **LLN Edge Validation** | Requires 30+ trades to confirm positive edge |

---

## ⚡ Performance Expectations

> **Be realistic.** This system is designed for consistent, risk-adjusted returns — not moonshots.

| Metric | Conservative Target | Optimistic Target |
|---|---|---|
| Annual Return | 15–25% | 30–50% |
| Sharpe Ratio | 0.8–1.2 | 1.5–2.5 |
| Max Drawdown | 8–15% | 5–10% |
| Win Rate | 52–58% | 55–62% |
| Profit Factor | 1.2–1.5 | 1.5–2.0 |

**Key caveats:**
- Past performance does not predict future results
- Crypto markets can experience black swan events with 30%+ single-day moves
- Slippage, latency, and exchange outages degrade live performance
- Models require periodic retraining as market regimes shift
- Start with paper trading and small live capital first

---

## 🔧 Development

```bash
# Run tests
pytest tests/ -v --cov=. --cov-report=html

# Type checking
mypy . --ignore-missing-imports

# Lint
ruff check .

# Format
ruff format .
```

---

## 📜 Licence

MIT — see `LICENSE` file. Use at your own risk. This is not financial advice.

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. Cryptocurrency
trading carries substantial risk of loss. You are solely responsible for your
trading decisions and any financial outcomes. Always start with paper trading and
never risk more than you can afford to lose.
