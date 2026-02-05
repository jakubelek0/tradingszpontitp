import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def evaluate_model(model, vec_env, deterministic=True):
    """Run one complete episode and collect metrics"""
    obs = vec_env.reset()
    equity_curve = []
    closed_trades = []
    rewards = []
    
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = vec_env.step(action)
        
        if len(step_out) == 4:
            obs, reward, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, reward, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])
        
        rewards.append(reward[0] if isinstance(reward, np.ndarray) else reward)
        equity_curve.append(vec_env.get_attr("equity_usd")[0])
        
        trade_info = vec_env.get_attr("last_trade_info")[0]
        if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
            closed_trades.append(trade_info)
        
        if done:
            break
    
    return equity_curve, closed_trades, rewards


def calculate_metrics(equity_curve, closed_trades, initial_equity=10000.0):
    """Calculate trading performance metrics"""
    metrics = {}
    
    # Basic metrics
    final_equity = equity_curve[-1]
    total_return = (final_equity - initial_equity) / initial_equity * 100
    
    metrics['Initial Equity'] = initial_equity
    metrics['Final Equity'] = final_equity
    metrics['Total Return (%)'] = total_return
    metrics['Profit/Loss ($)'] = final_equity - initial_equity
    
    # Trade statistics
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades)
        
        metrics['Total Trades'] = len(trades_df)
        
        winning_trades = trades_df[trades_df['net_pips'] > 0]
        losing_trades = trades_df[trades_df['net_pips'] < 0]
        
        metrics['Winning Trades'] = len(winning_trades)
        metrics['Losing Trades'] = len(losing_trades)
        metrics['Win Rate (%)'] = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        metrics['Avg Win (pips)'] = winning_trades['net_pips'].mean() if len(winning_trades) > 0 else 0
        metrics['Avg Loss (pips)'] = losing_trades['net_pips'].mean() if len(losing_trades) > 0 else 0
        
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            metrics['Profit Factor'] = abs(winning_trades['net_pips'].sum() / losing_trades['net_pips'].sum())
        else:
            metrics['Profit Factor'] = 0
        
        metrics['Avg Trade Duration (bars)'] = trades_df['time_in_trade'].mean()
        
        # Breakdown by reason
        metrics['TP Hits'] = len(trades_df[trades_df['reason'] == 'TP_HIT'])
        metrics['SL Hits'] = len(trades_df[trades_df['reason'] == 'SL_HIT'])
        metrics['Manual Closes'] = len(trades_df[trades_df['reason'] == 'MANUAL_CLOSE'])
    
    # Equity curve metrics
    equity_array = np.array(equity_curve)
    returns = np.diff(equity_array) / equity_array[:-1]
    
    metrics['Max Equity'] = equity_array.max()
    metrics['Min Equity'] = equity_array.min()
    
    # Maximum Drawdown
    peak = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - peak) / peak * 100
    metrics['Max Drawdown (%)'] = abs(drawdown.min())
    
    # Sharpe Ratio (annualized, assuming hourly data)
    if len(returns) > 0 and returns.std() > 0:
        metrics['Sharpe Ratio'] = returns.mean() / returns.std() * np.sqrt(252 * 24)  # hourly
    else:
        metrics['Sharpe Ratio'] = 0
    
    return metrics


def main():
    print("="*60)
    print("FOREX TRADING BOT - EVALUATION RESULTS")
    print("="*60)
    
    # Load data
    file_path = "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"
    df, feature_cols = load_and_preprocess_data(file_path)
    
    # Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"\nData loaded:")
    print(f"  Training bars: {len(train_df)}")
    print(f"  Testing bars: {len(test_df)}")
    
    # Environment parameters (must match training)
    SL_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
    TP_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
    WIN = 30
    
    # Create environments
    def make_eval_env(data_df):
        return ForexTradingEnv(
            df=data_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            hold_reward_weight=0.0,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0
        )
    
    train_eval_env = DummyVecEnv([lambda: make_eval_env(train_df)])
    test_eval_env = DummyVecEnv([lambda: make_eval_env(test_df)])
    
    # Check for models
    models_to_evaluate = []
    
    if os.path.exists("model_eurusd_best.zip"):
        models_to_evaluate.append(("Best Model", "model_eurusd_best.zip"))
    
    if os.path.exists("./checkpoints"):
        checkpoints = sorted(
            [f for f in os.listdir("./checkpoints") if f.endswith(".zip")],
            key=lambda x: os.path.getmtime(os.path.join("./checkpoints", x))
        )
        for ckpt in checkpoints[-3:]:  # Last 3 checkpoints
            models_to_evaluate.append((ckpt, os.path.join("./checkpoints", ckpt)))
    
    if not models_to_evaluate:
        print("\n❌ No models found! Please train first.")
        return
    
    print(f"\n📊 Found {len(models_to_evaluate)} model(s) to evaluate\n")
    
    # Evaluate each model
    results = []
    
    for model_name, model_path in models_to_evaluate:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        try:
            model = PPO.load(model_path)
            
            # Evaluate on training data
            print("\n📈 In-Sample (Training) Performance:")
            train_equity, train_trades, train_rewards = evaluate_model(model, train_eval_env)
            train_metrics = calculate_metrics(train_equity, train_trades)
            
            for key, value in train_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            
            # Evaluate on test data
            print("\n🎯 Out-of-Sample (Test) Performance:")
            test_equity, test_trades, test_rewards = evaluate_model(model, test_eval_env)
            test_metrics = calculate_metrics(test_equity, test_trades)
            
            for key, value in test_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            
            # Save trade history
            if test_trades:
                trades_df = pd.DataFrame(test_trades)
                csv_name = f"trades_{model_name.replace('.zip', '')}.csv"
                trades_df.to_csv(csv_name, index=False)
                print(f"\n💾 Trade history saved to: {csv_name}")
            
            # Store for comparison
            results.append({
                'name': model_name,
                'train_equity': train_equity,
                'test_equity': test_equity,
                'test_return': test_metrics['Total Return (%)'],
                'test_final': test_metrics['Final Equity']
            })
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
    
    # Plot comparison
    if results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Train equity curves
        ax = axes[0, 0]
        for r in results:
            ax.plot(r['train_equity'], label=r['name'], alpha=0.7)
        ax.set_title('In-Sample Equity Curves')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Equity ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Test equity curves
        ax = axes[0, 1]
        for r in results:
            ax.plot(r['test_equity'], label=r['name'], alpha=0.7)
        ax.set_title('Out-of-Sample Equity Curves')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Equity ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Model comparison
        ax = axes[1, 0]
        names = [r['name'] for r in results]
        returns = [r['test_return'] for r in results]
        ax.bar(range(len(names)), returns)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Return (%)')
        ax.set_title('Test Returns Comparison')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Final equity
        ax = axes[1, 1]
        finals = [r['test_final'] for r in results]
        ax.bar(range(len(names)), finals)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Final Equity ($)')
        ax.set_title('Final Equity Comparison')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=10000, color='r', linestyle='--', alpha=0.5, label='Initial')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('evaluation_results.png', dpi=150)
        print(f"\n📊 Evaluation plots saved to: evaluation_results.png")
        plt.show()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()