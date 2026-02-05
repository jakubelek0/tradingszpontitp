import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def main():
    file_path = "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"
    df, feature_cols = load_and_preprocess_data(file_path)
    
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    SL_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
    TP_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
    WIN = 30
    
    test_env = ForexTradingEnv(
        df=test_df,
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
    
    vec_env = DummyVecEnv([lambda: test_env])
    
    # Use the latest checkpoint
    model_path = "./checkpoints/ppo_eurusd_250000_steps.zip"
    
    try:
        model = PPO.load(model_path)
        print(f"✅ Model loaded: {model_path}\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}\n")
        return
    
    # Run with detailed logging
    obs = vec_env.reset()
    
    action_counts = {}
    trades_opened = 0
    trades_closed = 0
    steps = 0
    max_steps = 500
    
    print("="*70)
    print("DIAGNOSTIC RUN - First 500 steps")
    print("="*70)
    
    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        action_val = int(action[0])
        
        # Count actions
        action_counts[action_val] = action_counts.get(action_val, 0) + 1
        
        # Decode action
        act_type, direction, sl, tp = test_env.action_map[action_val]
        
        step_out = vec_env.step(action)
        
        if len(step_out) == 4:
            obs, reward, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, reward, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])
        
        info = infos[0] if isinstance(infos, list) else infos
        
        # Check for trade events
        trade_info = vec_env.get_attr("last_trade_info")[0]
        if trade_info:
            if trade_info.get("event") == "OPEN":
                trades_opened += 1
                dir_name = "LONG" if direction == 1 else "SHORT"
                print(f"\n📈 Step {steps}: OPENED {dir_name} | SL={sl} TP={tp}")
                print(f"   Entry: {trade_info['entry_price']:.5f}")
            elif trade_info.get("event") == "CLOSE":
                trades_closed += 1
                print(f"\n📉 Step {steps}: CLOSED ({trade_info['reason']})")
                print(f"   Net pips: {trade_info['net_pips']:.2f}")
                print(f"   Equity: ${trade_info['equity_usd']:.2f}")
        
        # Print periodic updates
        if steps % 100 == 0:
            equity = vec_env.get_attr("equity_usd")[0]
            position = vec_env.get_attr("position")[0]
            print(f"\nStep {steps}: Equity=${equity:.2f} | Position={position} | Last action={act_type}")
        
        if done:
            break
        
        steps += 1
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    final_equity = vec_env.get_attr("equity_usd")[0]
    
    print(f"\nFinal Equity: ${final_equity:.2f}")
    print(f"Trades Opened: {trades_opened}")
    print(f"Trades Closed: {trades_closed}")
    print(f"\nAction Distribution (first {steps} steps):")
    
    # Show top 10 most common actions
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for action_idx, count in sorted_actions:
        act_type, direction, sl, tp = test_env.action_map[action_idx]
        pct = count / steps * 100
        if act_type == "OPEN":
            dir_str = "LONG" if direction == 1 else "SHORT"
            print(f"  Action {action_idx}: {act_type} {dir_str} SL={sl} TP={tp} - {count} times ({pct:.1f}%)")
        else:
            print(f"  Action {action_idx}: {act_type} - {count} times ({pct:.1f}%)")
    
    print("\n" + "="*70)
    
    # Diagnose the problem
    print("\n🔍 DIAGNOSIS:")
    if trades_opened == 0:
        print("❌ PROBLEM: The agent NEVER opens any trades!")
        print("\n   Possible causes:")
        print("   1. Over-penalized for trading (open_penalty_pips, time_penalty_pips)")
        print("   2. Agent learned HOLD is safest")
        print("   3. Features not informative")
        print("\n   SOLUTION: Retrain with NO penalties and reward shaping:")
        print("   - Set open_penalty_pips=0.0")
        print("   - Set time_penalty_pips=0.0")
        print("   - Set unrealized_delta_weight=0.1 or 0.15")
    elif trades_closed == 0:
        print("⚠️  PROBLEM: Trades opened but never closed")
        print("   Check SL/TP logic")
    elif final_equity == 10000.0:
        print("⚠️  Trades happened but equity unchanged")
    else:
        print(f"✅ Agent IS trading! Final equity: ${final_equity:.2f}")
        if final_equity < 10000:
            print(f"   But losing money: ${final_equity - 10000:.2f}")
        else:
            print(f"   Making money: ${final_equity - 10000:.2f}")
    
    print("="*70)


if __name__ == "__main__":
    main()