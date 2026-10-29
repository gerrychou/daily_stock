from pathlib import Path
from datetime import datetime
import time

import pandas as pd
import yfinance as yf
import mplfinance as mpf

# Settings
TICKER_FILE = "tickers.txt"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Read selected Yahoo Finance symbols
tickers = [
    line.strip()
    for line in open(TICKER_FILE, encoding="utf-8")
    if line.strip() and not line.strip().startswith("#")
]

# Download 3 months so the 20-day indicators have enough data
# Display only the most recent 2 calendar months
display_start = pd.Timestamp.now().normalize() - pd.DateOffset(months=2)

success = []
failed = []

for i, ticker in enumerate(tickers, start=1):
    try:
        print(f"[{i}/{len(tickers)}] Downloading {ticker}")

        df = yf.download(
            ticker,
            period="3mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=25,
            multi_level_index=False,
        )

        if df.empty:
            raise ValueError("No price data returned")

        # Handle possible multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns=str.title)
        required = ["Open", "High", "Low", "Close", "Volume"]
        df = df[required].dropna(subset=["Open", "High", "Low", "Close"])

        if len(df) < 20:
            raise ValueError("Fewer than 20 daily records")

        # Bollinger Bands: 20-day moving average +/- 2 standard deviations
        df["MA20"] = df["Close"].rolling(20).mean()
        std20 = df["Close"].rolling(20).std(ddof=0)
        df["BB_Upper"] = df["MA20"] + 2 * std20
        df["BB_Lower"] = df["MA20"] - 2 * std20

        # Calculate indicators first, then select the display period
        df = df.loc[df.index >= display_start].copy()
        df = df.dropna(subset=["MA20", "BB_Upper", "BB_Lower"])

        if df.empty:
            raise ValueError("No data in the display period")

        plots = [
            mpf.make_addplot(df["MA20"], color="orange", width=1.0),
            mpf.make_addplot(df["BB_Upper"], color="blue", width=0.9),
            mpf.make_addplot(df["BB_Lower"], color="blue", width=0.9),
        ]

        market_colors = mpf.make_marketcolors(
            up="red", down="green",
            edge="inherit", wick="inherit",
            volume="inherit",
        )
        style = mpf.make_mpf_style(
            marketcolors=market_colors,
            gridstyle=":",
        )

        output_file = OUTPUT_DIR / f"{ticker.replace('.', '_')}.png"

        mpf.plot(
            df,
            type="candle",
            style=style,
            addplot=plots,
            volume=True,
            figsize=(12, 8),
            title=f"{ticker} - Daily K Chart + Bollinger Bands",
            ylabel="Price",
            ylabel_lower="Volume",
            datetime_format="%m/%d",
            xrotation=0,
            tight_layout=True,
            savefig=dict(fname=str(output_file), dpi=140),
        )

        success.append(ticker)
        print(f"  Saved: {output_file}")

        # Small pause to reduce request pressure on Yahoo Finance
        time.sleep(0.3)

    except Exception as exc:
        failed.append((ticker, str(exc)))
        print(f"  FAILED: {ticker}: {exc}")

# Save a report so failed downloads are not silently missed
with open(OUTPUT_DIR / "report.txt", "w", encoding="utf-8") as f:
    f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    f.write(f"Successful: {len(success)} / {len(tickers)}\n\n")
    f.write("Failed tickers:\n")
    for ticker, reason in failed:
        f.write(f"{ticker}: {reason}\n")

print(f"Finished: {len(success)} succeeded, {len(failed)} failed.")
