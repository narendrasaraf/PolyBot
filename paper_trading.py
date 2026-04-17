import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class PaperTradingSimulator:
    """
    Paper Trading Simulator for Polymarket Trading Bot.
    Evaluates the profitability and robustness of a trading strategy under realistic conditions.
    """
    def __init__(self, initial_balance=1000.0, risk_pct=0.02):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_pct = risk_pct
        
        self.open_positions = {}
        self.trade_log = []
        self.equity_curve = [initial_balance]
        self.trade_count = 0

    def compute_ev(self, predicted_prob, market_price):
        """Calculates Expected Value based on predicted probability and current market price."""
        return (predicted_prob * (1 - market_price)) - ((1 - predicted_prob) * market_price)

    def check_exit(self, pos, current_price, predicted_prob):
        """
        Check if an open position hits Take Profit, Stop Loss, or Reversal.
        Returns the exit reason if triggered, None otherwise.
        """
        entry_price = pos['entry_price']
        side = pos['side']
        
        take_profit_target = 0.10
        stop_loss_target = -0.05
        
        if side == "BUY_YES":
            pnl_pct = (current_price - entry_price) / entry_price
            reversal = predicted_prob < current_price
        else: # BUY_NO
            # Pricing for NO token is (1 - YES market_price)
            current_no_price = 1.0 - current_price
            entry_no_price = 1.0 - entry_price
            pnl_pct = (current_no_price - entry_no_price) / entry_no_price
            reversal = predicted_prob > current_price
            
        if pnl_pct >= take_profit_target:
            return "TAKE_PROFIT"
        elif pnl_pct <= stop_loss_target:
            return "STOP_LOSS"
        elif reversal:
            return "REVERSAL"
            
        return None

    def run_simulation(self, df: pd.DataFrame):
        """
        Iterates over the market data chronologically, managing entries, exits, and capital allocation.
        DataFrame must contain: ['timestamp', 'question', 'market_price', 'predicted_prob', 'volume', 'liquidity']
        """
        print(f"Starting Paper Trading Simulation with ${self.initial_balance:.2f}...")
        
        if 'timestamp' in df.columns:
            df = df.sort_values(by='timestamp').reset_index(drop=True)
            
        for idx, row in df.iterrows():
            question = row['question']
            market_price = float(row['market_price'])
            predicted_prob = float(row['predicted_prob'])
            volume = float(row['volume'])
            liquidity = float(row['liquidity'])
            
            # 1. Evaluate Open Positions for Exit
            if question in self.open_positions:
                pos = self.open_positions[question]
                exit_reason = self.check_exit(pos, market_price, predicted_prob)
                
                if exit_reason:
                    # Calculate PnL
                    if pos['side'] == "BUY_YES":
                        pnl_pct = (market_price - pos['entry_price']) / pos['entry_price']
                    else:
                        pnl_pct = ((1.0 - market_price) - (1.0 - pos['entry_price'])) / (1.0 - pos['entry_price'])
                        
                    profit_loss = pos['size_dollars'] * pnl_pct
                    self.balance += pos['size_dollars'] + profit_loss
                    
                    self.trade_log.append({
                        "question": question,
                        "side": pos['side'],
                        "entry_price": pos['entry_price'],
                        "exit_price": market_price,
                        "predicted_prob": pos['entry_prob'], # Probability at entry
                        "EV": pos['EV'],
                        "exit_reason": exit_reason,
                        "profit_loss": round(profit_loss, 4),
                        "balance_after_trade": round(self.balance, 2)
                    })
                    
                    del self.open_positions[question]
                    self.equity_curve.append(self.balance)
                    self.trade_count += 1
            
            # 2. Evaluate for Entry Constraints
            if question not in self.open_positions:
                ev = self.compute_ev(predicted_prob, market_price)
                edge = abs(predicted_prob - market_price)
                
                # Minimum viable operational constraints
                if ev > 0.05 and edge > 0.10 and volume > 5000 and liquidity > 10000:
                    side = "BUY_YES" if predicted_prob > market_price else "BUY_NO"
                    
                    # Position Sizing
                    size_dollars = self.balance * self.risk_pct
                    
                    # Protect against completely allocating empty balances implicitly
                    if size_dollars > 1.0: 
                        self.balance -= size_dollars
                        
                        self.open_positions[question] = {
                            "side": side,
                            "entry_price": market_price,
                            "entry_prob": predicted_prob,
                            "EV": ev,
                            "size_dollars": size_dollars
                        }
                        
        print("Simulation Completed.")
        
    def save_logs(self, file_name="paper_trades_1000.csv"):
        """Stores historical logging logic."""
        if not self.trade_log:
            print("No trades logged to save.")
            return
            
        log_df = pd.DataFrame(self.trade_log)
        log_df.to_csv(file_name, index=False)
        print(f"Saved trades to {file_name}")

    def plot_equity_curve(self, output_file="equity_curve.png"):
        """Renders explicit systemic parameters graphing trades onto performance map over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.equity_curve)), self.equity_curve, marker='o', linestyle='-', color='purple')
        plt.title('Polymarket Paper Trading - Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Balance ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Equity curve saved to {output_file}")

    def summary(self):
        """Prints formatting explicit to portfolio generation specifications requested."""
        if not self.trade_log:
            print("\n## PAPER TRADING REPORT ($1000)")
            print("No trades were executed. Check execution gates regarding liquidity, EV, and active spread parameters.")
            return

        total_profit = self.balance - self.initial_balance
        roi = (total_profit / self.initial_balance) * 100
        
        wins = [t for t in self.trade_log if t['profit_loss'] > 0]
        win_rate = len(wins) / len(self.trade_log) * 100
        
        avg_profit = np.mean([t['profit_loss'] for t in self.trade_log])
        
        # Calculate Max Drawdown
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        report = f"""
## PAPER TRADING REPORT (${self.initial_balance:.0f})
---------------------------------------------------
Final Balance:  ${self.balance:.2f}
Total Profit:   ${total_profit:+.2f}
ROI:            {roi:+.2f}%
Win Rate:       {win_rate:.1f}%
Max Drawdown:   {max_drawdown:.2f}%
Total Trades:   {len(self.trade_log)}
Avg Profit/Trade: ${avg_profit:+.2f}
---------------------------------------------------
"""
        print(report)


# --- Example Execution Block ---
if __name__ == "__main__":
    # Generate realistic dummy data for evaluation
    np.random.seed(42)
    rows = 500
    
    # Simulating 5 independent markets over 100 timesteps explicitly
    questions = [f"Market_{i}" for i in range(5)]
    data = []
    
    for q in questions:
        base_price = np.random.uniform(0.3, 0.7)
        for t in range(100):
            # Price takes random-walk
            base_price = np.clip(base_price + np.random.normal(0, 0.02), 0.05, 0.95)
            # Model predicts slightly ahead/lagging
            pred_prob = np.clip(base_price + np.random.normal(0, 0.15), 0.05, 0.95)
            
            data.append({
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=t),
                "question": q,
                "market_price": base_price,
                "predicted_prob": pred_prob,
                "volume": np.random.uniform(6000, 20000),     # Passes explicit liquidity constraint > 5k
                "liquidity": np.random.uniform(15000, 50000), # Passes > 10k constraint
            })
            
    df = pd.DataFrame(data)
    
    # Initialize and execute core parameters
    simulator = PaperTradingSimulator(initial_balance=1000.0, risk_pct=0.02)
    simulator.run_simulation(df)
    simulator.save_logs()
    simulator.plot_equity_curve()
    simulator.summary()
