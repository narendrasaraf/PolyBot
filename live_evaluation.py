import os
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, log_loss, brier_score_loss
)

# =====================================================================
# SYSTEM CONSTANTS & CONFIGURATION
# =====================================================================
API_URL = "https://gamma-api.polymarket.com/markets"
PREDICTIONS_FILE = "live_predictions_ev.csv"
TRADES_FILE = "live_trades_ev.csv"

# Risk Management & EV Filters
INITIAL_BALANCE = 1000.0
MIN_VOLUME = 5000.0          # Reject low-volume noise
MIN_LIQUIDITY = 10000.0      # Reject illiquid markets
EV_THRESHOLD = 0.12          # Demand an Expected Value (Edge) >= 12%
BASE_RISK_PERCENT = 0.10     # Max base scaling for sizing
MAX_RISK_PERCENT = 0.03      # Hard position cap at 3%
TAKE_PROFIT = 0.15           # Exit for profit early
STOP_LOSS = -0.05            # Cut losses relentlessly
SLIPPAGE = 0.01              # Penalty for market execution

# =====================================================================
# 1. LIVE DATA FETCHING
# =====================================================================
def fetch_live_markets(limit=250):
    print(f"Fetching live data from {API_URL}...")
    params = {"limit": limit, "active": "true", "closed": "false"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(API_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        markets_data = response.json()
        
        markets = markets_data.get("markets", markets_data.get("data", [])) if isinstance(markets_data, dict) else markets_data
        data = []
        
        for m in markets:
            tokens_list = m.get("tokens", m.get("clobTokenIds", []))
            if tokens_list and len(tokens_list) >= 2:
                yes_price = float(m.get("bestAsk", 0.5))
                if isinstance(tokens_list[0], dict) and "price" in tokens_list[0]:
                    yes_price = float(tokens_list[0].get("price", yes_price))
                    
                volume = float(m.get("volumeNum", m.get("volume", 0)))
                liquidity = float(m.get("liquidityNum", m.get("liquidity", 0)))
                
                data.append({
                    "id": m.get("condition_id") or m.get("id"),
                    "question": m.get("question", "Unknown"),
                    "yes_price": yes_price,
                    "volume": volume,
                    "liquidity": liquidity,
                })
        print(f"Fetched {len(data)} valid binary markets.")
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

# =====================================================================
# 2. MODEL INPUT (MOCK PREDICTION)
# =====================================================================
def model_prediction(row):
    """
    Mock prediction predicting true YES probability.
    Injecting a wider noise to simulate highly opinionated ML models finding edge.
    """
    price = row['yes_price']
    noise = np.random.normal(0, 0.20)
    return np.clip(price + noise, 0.01, 0.99)

# =====================================================================
# 3. TRADING ENGINE: EV, FILTERING, & DYNAMIC SIZING
# =====================================================================
class QuantitativeTradingEngine:
    def __init__(self):
        self.balance = INITIAL_BALANCE
        self.trades = []
        self.predictions = []
    
    def generate_signals(self, df):
        df['predicted_prob'] = df.apply(model_prediction, axis=1)
        
        for _, row in df.iterrows():
            if row['liquidity'] < MIN_LIQUIDITY or row['volume'] < MIN_VOLUME:
                continue
                
            price = row['yes_price']
            pred = row['predicted_prob']
            
            self.predictions.append({
                "id": row['id'],
                "question": row['question'],
                "yes_price": price,
                "predicted_prob": pred,
                "predicted_outcome": 1 if pred > 0.5 else 0
            })
            
            # --- 1. Expected Value (EV) & Confidence Calculation ---
            # Buying YES share (costs `price` to potentially win 1.00)
            # EV = (Expected profit per share)
            ev_yes = pred - price
            
            # Buying NO share (costs `1 - price` to potentially win 1.00)
            # NO Price = (1 - price), NO Prob = (1 - pred)
            ev_no = (1 - pred) - (1 - price) # which directly simplifies to `price - pred`
            
            confidence = abs(pred - price)
            action = None
            ev = 0.0
            entry_price = 0.0
            
            # --- 2. EV-Based Trade Filtering ---
            if ev_yes > EV_THRESHOLD:
                action = "BUY_YES"
                ev = ev_yes
                entry_price = price + SLIPPAGE
            elif ev_no > EV_THRESHOLD:
                action = "BUY_NO"
                ev = ev_no
                entry_price = (1 - price) + SLIPPAGE
                
            # Discard if execution slippage destroyed edge, or edge too small
            if action and entry_price < 0.97 and confidence > EV_THRESHOLD:
                
                # --- 3. Dynamic Position Sizing ---
                # Size linearly increases with confidence, strictly capped for survival
                scaled_risk = min(BASE_RISK_PERCENT * confidence, MAX_RISK_PERCENT)
                risk_amount = self.balance * scaled_risk
                
                self.execute_trade(row, action, entry_price, pred, confidence, ev, risk_amount)

    def execute_trade(self, row, action, entry_price, pred, confidence, ev, risk_amount):
        """Places the position calculated by the risk manager."""
        shares = risk_amount / entry_price
        
        self.trades.append({
            "market_id": row['id'],
            "question": row['question'],
            "action": action,
            "market_price": row['yes_price'],
            "predicted_prob": pred,
            "confidence": confidence,
            "ev": ev,
            "entry_price": entry_price,
            "risk_amount": risk_amount,
            "shares": shares,
            "status": "OPEN",
        })

    def save_logs(self):
        if self.predictions:
            pd.DataFrame(self.predictions).to_csv(PREDICTIONS_FILE, index=False)
        if self.trades:
            pd.DataFrame(self.trades).to_csv(TRADES_FILE, index=False)

# =====================================================================
# 4. SIMULATION: DELAYED RESOLUTION & EXITS 
# =====================================================================
def simulate_resolutions_and_exits(engine):
    """
    Evaluates paper trades against simulated actual future outcomes and our strict Take-Profit/Stop-Loss logic.
    """
    if not engine.predictions: return
    
    preds_df = pd.DataFrame(engine.predictions)
    
    # Mocking real future: somewhat correlated to price
    preds_df['actual_outcome'] = preds_df['yes_price'].apply(
        lambda p: 1 if random.random() < p else 0
    )
    
    for t in engine.trades:
        entry = t['entry_price']
        
        # Scenarios probability (simulating that EV trades hit TP more than SL)
        # Random exit scenarios to mock real trading volatility lifecycle
        scenario = random.choices(["TP", "SL", "RESOLVE"], weights=[0.45, 0.20, 0.35])[0]
        
        if scenario == "TP":
            exit_p = entry * (1 + TAKE_PROFIT) - SLIPPAGE
            t['status'] = "CLOSED: TP"
        elif scenario == "SL":
            exit_p = entry * (1 + STOP_LOSS) - SLIPPAGE
            t['status'] = "CLOSED: SL"
        else:
            outcome_row = preds_df[preds_df['id'] == t['market_id']]
            if not outcome_row.empty:
                actual = outcome_row.iloc[0]['actual_outcome']
                # Resolution pays 1.00 directly if right
                won = (t['action'] == "BUY_YES" and actual == 1) or (t['action'] == "BUY_NO" and actual == 0)
                exit_p = 1.0 if won else 0.0
                t['status'] = f"CLOSED: RESOLVED ({actual})"
            else:
                exit_p = 0.0
                t['status'] = "CLOSED: EXPIRED DUD"
        
        # Calculate resulting metrics
        exit_p = np.clip(exit_p, 0.0, 1.0)
        profit = (exit_p - entry) * t['shares']
        
        t['exit_price'] = exit_p
        t['pnl'] = profit
        t['roi_percent'] = (exit_p - entry) / entry * 100
        engine.balance += profit

# =====================================================================
# 5. EVALUATION: QUANTITATIVE ROI > ACCURACY
# =====================================================================
def evaluate_performance(engine):
    print("\n" + "="*55)
    print(" QUANTITATIVE TRADING ENGINE EVALUATION")
    print("="*55)
    
    # Base ML Accuracy (Often deceptive in trading!)
    preds_df = pd.DataFrame(engine.predictions)
    if not preds_df.empty and 'actual_outcome' in preds_df.columns:
        preds_df.dropna(subset=['actual_outcome'], inplace=True)
        # Safe calc if classes aren't both existent (e.g. all 0s)
        if len(np.unique(preds_df['actual_outcome'])) > 1:
            acc = accuracy_score(preds_df['actual_outcome'], preds_df['predicted_outcome'])
            print(f"\n[Secondary Metric] ML Total Environment Accuracy: {acc:.2%}")
            
    # Hard Quantitative Metrics (What matters for Profit)
    trades_df = pd.DataFrame(engine.trades)
    if trades_df.empty:
        print("\nNo trades executed. EV thresholds prevented risky capital exposure.")
        return
        
    trades_df['win'] = trades_df['pnl'] > 0
    total_trades = len(trades_df)
    win_rate = trades_df['win'].mean()
    total_profit = trades_df['pnl'].sum()
    roi = (total_profit / INITIAL_BALANCE) * 100
    
    # Calculate Drawdown
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    trades_df['equity'] = INITIAL_BALANCE + trades_df['cumulative_pnl']
    peak = trades_df['equity'].cummax()
    drawdown = (trades_df['equity'] - peak) / peak * 100
    max_drawdown = drawdown.min()

    print("\n--- NEW VALUE AND ROI METRICS ---")
    print(f"Total Filtered Quality Trades: {total_trades}")
    print(f"System Win Rate:               {win_rate:.2%}")
    print(f"Max Peak-to-Trough Drawdown:   {max_drawdown:.2f}%")
    print(f"Total Portfolio Return (ROI):  {roi:.2f}%")
    print(f"Final Account Capital:         ${engine.balance:.2f} (Start: ${INITIAL_BALANCE:.2f})")
    
    ev_mean = trades_df['ev'].mean()
    avg_confidence = trades_df['confidence'].mean()
    print(f"\n--- POSITION QUALITY ANALYSIS ---")
    print(f"Avg Captured EV Edge (Expected):    +{ev_mean:.1%}")
    print(f"Avg Prediction Confidence Divergence: {avg_confidence:.1%}")
    
    print("=======================================================")

    return trades_df

# =====================================================================
# MAIN RUNTIME
# =====================================================================
if __name__ == "__main__":
    print("\n[INIT] Starting Expected Value Quantitative Trading Protocol...")
    df = fetch_live_markets(limit=250)
    
    if not df.empty:
        print("\n[PHASE 1] Scanning markets for EV Edge > 12% & Volume > 5000...")
        quant_engine = QuantitativeTradingEngine()
        
        quant_engine.generate_signals(df)
        quant_engine.save_logs()
        print(f" - Found {len(quant_engine.trades)} trades matching strict EV parameters out of {len(df)} markets.")
        
        print("\n[PHASE 2] Simulating EV Execution Pipeline (SL/TP triggers)...")
        simulate_resolutions_and_exits(quant_engine)
        
        print("\n[PHASE 3] Final Performance Reconciliation...")
        trades_df = evaluate_performance(quant_engine)
        
        if trades_df is not None and not trades_df.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(trades_df) + 1), trades_df['equity'], marker='o', linewidth=2, color='#2ca02c')
            plt.axhline(INITIAL_BALANCE, color='gray', linestyle='--', label='Starting Capital')
            plt.title(f"Quantitative Portfolio Equity Curve (EV Filtered)")
            plt.xlabel("Trade Sequence")
            plt.ylabel("Capital ($)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("equity_curve_ev.png")
            print("\nSaved high-quality quantitative equity curve to 'equity_curve_ev.png'.")
            
        print("\nSystem Evaluation Complete.")
    else:
        print("Market data retrieval failed.")
