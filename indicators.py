import pandas as pd
import ta


def load_and_preprocess_data(csv_path: str):
    """
    Loads EURUSD data from CSV and preprocesses it by adding RELATIVE technical features.

    CSV expected columns: [Gmt time, Open, High, Low, Close, Volume]
    The returned DataFrame still contains OHLCV for env internals,
    but `feature_cols` lists only the RELATIVE columns to feed the agent.
    """
    df = pd.read_csv(
        csv_path,
        parse_dates=["Gmt time"],
        dayfirst=True,
    )

    # Strip any trailing spaces in headers (e.g. 'Volume ')
    df.columns = df.columns.str.strip()

    # Datetime index
    df = df.set_index("Gmt time")
    df.sort_index(inplace=True)

    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Technicals ----
    # RSI (using ta library's momentum module)
    df["rsi_14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    
    # ATR (using ta library's volatility module)
    df["atr_14"] = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], window=14
    ).average_true_range()

    # Moving averages (using ta library's trend module)
    df["ma_20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
    df["ma_50"] = ta.trend.SMAIndicator(df["Close"], window=50).sma_indicator()

    # Slopes of the MAs
    df["ma_20_slope"] = df["ma_20"].diff()
    df["ma_50_slope"] = df["ma_50"].diff()

    # Distance of price from each MA (relative level)
    df["close_ma20_diff"] = df["Close"] - df["ma_20"]
    df["close_ma50_diff"] = df["Close"] - df["ma_50"]

    # MA divergence: MA20 vs MA50
    df["ma_spread"] = df["ma_20"] - df["ma_50"]
    df["ma_spread_slope"] = df["ma_spread"].diff()

    # Drop initial NaNs from indicators
    df.dropna(inplace=True)

    # Columns the AGENT should see (no raw price levels / raw MAs)
    feature_cols = [
        "rsi_14",
        "atr_14",
        "ma_20_slope",
        "ma_50_slope",
        "close_ma20_diff",
        "close_ma50_diff",
        "ma_spread",
        "ma_spread_slope",
    ]

    return df, feature_cols