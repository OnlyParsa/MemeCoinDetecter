#!/usr/bin/env python3
"""
Meme Coin Pump & Ultra-Precision Ultimate Detector
==================================================

This is the final, production-ready version of the Meme Coin Pump Detector. It leverages advanced
algorithms, adaptive EMA smoothing, volatility measurement, momentum analysis, and z-score based
outlier detection to pinpoint the highest potential meme coins in real time.

Features:
  - Real-time updates with dynamic, adaptive EMA smoothing based on price volatility.
  - Advanced metrics: raw pump_score, EMA-smoothed pump_score, momentum, volatility, potential_score,
    and standardized z-score.
  - Elite signal detection based on multi-dimensional analysis (z-score, momentum, and growth factors).
  - Rich terminal-based UI showcasing coin details, dynamic sparklines, and overall statistics.
  - Resilient network request handling with exponential backoff.
  - Comprehensive logging for debugging and monitoring.

Usage:
  python main2.py --mode terminal --interval 1 --min_market_cap 0 --max_market_cap 1e12 --ema_alpha 0.3

Requirements:
  - Python 3.7+
  - requests
  - rich

Author: OnlyParsa
Date: 2025-02-15
"""

import sys
import time
import math
import requests
import argparse
import logging
import statistics
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.align import Align
from rich.text import Text
from rich.box import DOUBLE_EDGE
from rich.layout import Layout

# Setup logging for console and optional file logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
console = Console()

# Global dictionaries for tracking EMA values and pump score history.
prev_ema_scores = {}    # coin_id -> previous EMA pump score
pump_history = {}       # coin_id -> list of pump_score history values (for sparkline, momentum, volatility)
HISTORY_LENGTH = 10

# Dictionaries to store momentum and volatility for each coin.
pump_momentum = {}      # coin_id -> difference between last two EMA pump scores
pump_volatility = {}    # coin_id -> standard deviation of pump score history

# Sparkline characters for visualization.
SPARK_CHARS = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

def make_sparkline(history):
    """Convert a list of numeric values into a sparkline string for quick visualization."""
    if not history:
        return ""
    mini = min(history)
    maxi = max(history)
    if maxi == mini:
        return "".join([SPARK_CHARS[0]] * len(history))
    sparkline = ""
    for value in history:
        norm = (value - mini) / (maxi - mini)
        index = int(round(norm * (len(SPARK_CHARS) - 1)))
        sparkline += SPARK_CHARS[index]
    return sparkline

def update_history(coin_id, pump_score):
    """Update pump score history for a given coin and return its updated history list."""
    if coin_id not in pump_history:
        pump_history[coin_id] = []
    pump_history[coin_id].append(pump_score)
    if len(pump_history[coin_id]) > HISTORY_LENGTH:
        pump_history[coin_id] = pump_history[coin_id][-HISTORY_LENGTH:]
    return pump_history[coin_id]

def compute_volatility(history):
    """Calculate the standard deviation from the pump score history."""
    if len(history) > 1:
        return statistics.stdev(history)
    return 0.0

def dynamic_alpha(coin_id, default_alpha):
    """
    Calculate a dynamic EMA alpha based on the volatility of the pump score history.
    If volatility is high, assign a higher alpha (up to a cap of 0.8) for greater responsiveness.
    """
    history = pump_history.get(coin_id, [])
    vol = compute_volatility(history)
    # Dynamic alpha is inversely related to volatility, bounded between default_alpha and 0.8.
    dynamic = default_alpha
    if vol > 0:
        dynamic = min(0.8, max(default_alpha, 1.0 / (1.0 + vol)))
    return dynamic

def exponential_moving_average(new_value, prev_ema, alpha):
    """Compute the exponential moving average for the current value with given alpha."""
    return alpha * new_value + (1 - alpha) * prev_ema

def calculate_momentum(coin_id):
    """
    Calculate momentum as the difference between the last two EMA pump scores.
    A positive momentum indicates upward acceleration.
    """
    history = pump_history.get(coin_id, [])
    if len(history) >= 2:
        momentum = history[-1] - history[-2]
    else:
        momentum = 0.0
    pump_momentum[coin_id] = round(momentum, 2)
    return pump_momentum[coin_id]

def retry_request(max_attempts=7, initial_wait=5):
    """
    Decorator for network requests that retries with exponential backoff on failure.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            wait_time = initial_wait
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.warning(f"[Retry] {func.__name__} attempt {attempt}/{max_attempts} failed: {e}")
                    if attempt < max_attempts:
                        logging.info(f"Retrying after {wait_time} seconds...")
                        time.sleep(wait_time)
                        wait_time *= 2
                    else:
                        raise Exception(f"{func.__name__} failed after {max_attempts} attempts.")
        return wrapper
    return decorator

# Create a session for API requests with custom headers.
session = requests.Session()
session.headers.update({
    "User-Agent": "memebot/3.0 (https://github.com/OnlyParsa/memebot)"
})

@retry_request(max_attempts=7, initial_wait=5)
def get_with_retry(url, params=None, timeout=10):
    """Send an HTTP GET request with retry logic, handling rate limiting via Retry-After."""
    response = session.get(url, params=params, timeout=timeout)
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                wait_seconds = int(retry_after)
                logging.info(f"Received Retry-After header. Sleeping for {wait_seconds} seconds.")
                time.sleep(wait_seconds)
            except ValueError:
                logging.info("Invalid Retry-After value; using default wait time.")
        raise Exception("429 Too Many Requests")
    response.raise_for_status()
    return response

def get_meme_coin_ids(keywords):
    """
    Retrieve a list of coin IDs from the CoinCap API that match any of the specified keywords.
    Keywords are matched against the coin's name, id, or symbol (case-insensitive).
    """
    url = "https://api.coincap.io/v2/assets"
    params = {"limit": 2000}
    try:
        response = get_with_retry(url, params=params, timeout=10)
        data = response.json().get("data", [])
        filtered_ids = []
        for coin in data:
            lower_name = coin.get("name", "").lower()
            lower_id = coin.get("id", "").lower()
            lower_symbol = coin.get("symbol", "").lower()
            if any(keyword in lower_name or keyword in lower_id or keyword in lower_symbol for keyword in keywords):
                filtered_ids.append(coin.get("id"))
        unique_ids = list(set(filtered_ids))
        if not unique_ids:
            logging.warning("No meme coins found using provided keywords.")
        else:
            logging.info(f"Identified {len(unique_ids)} potential meme coins based on keywords.")
        return unique_ids
    except Exception as e:
        logging.error("Error fetching coin list: %s", e)
        return []

class MemeCoinPumpDetector:
    def __init__(self, update_interval=1, keywords=None, min_market_cap=0, max_market_cap=float('inf'), ema_alpha=0.3):
        """
        Initialize the Meme Coin Pump Detector.

        Parameters:
          update_interval: Interval (in seconds) between data updates.
          keywords: List of keywords used to filter meme coins (default set provided).
          min_market_cap: Minimum market capitalization in USD for filtering.
          max_market_cap: Maximum market capitalization in USD for filtering.
          ema_alpha: Default alpha value for the exponential moving average.
        """
        self.update_interval = update_interval
        if keywords is None:
            keywords = ["meme", "doge", "inu", "shiba", "moon", "safemoon", "floki", "pepe", "elon", "snoop", "kishu"]
        self.keywords = keywords
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap
        self.ema_alpha = ema_alpha
        self.meme_coins = get_meme_coin_ids(self.keywords)
    
    def fetch_data(self):
        """
        Fetch market data from the CoinCap API and filter coins based on the selected meme coin IDs
        and the specified market capitalization range.
        """
        url = "https://api.coincap.io/v2/assets"
        params = {"limit": 2000}
        try:
            response = get_with_retry(url, params=params, timeout=10)
            tickers = response.json().get("data", [])
            filtered = [
                ticker for ticker in tickers
                if ticker.get("id") in self.meme_coins and 
                   self.min_market_cap <= float(ticker.get("marketCapUsd") or 0) <= self.max_market_cap
            ]
            return filtered
        except Exception as e:
            logging.error("Error fetching market data: %s", e)
            return []
    
    def analyze_data(self, data):
        """
        Process the retrieved coin data to compute various advanced metrics:
          - Raw pump_score from 24h percentage change, volume-to-market cap ratio, and bonus factors.
          - Adaptive EMA-smoothed pump_score.
          - Momentum calculated as the change in EMA pump score.
          - Volatility computed as the standard deviation of the pump score history.
          - potential_score enhanced with a growth factor.
          - z_score: the standardized score for pump_score.
          - Elite flag: marks coins with high z_score, positive momentum, and strong potential.
        """
        processed = []
        pump_scores_all = []  # To calculate overall averages for z_score computation
        
        # Primary processing: calculate raw score and EMA, update history.
        for coin in data:
            try:
                p24 = float(coin.get("changePercent24Hr") or 0)
                current_price = float(coin.get("priceUsd") or 0)
                volume = float(coin.get("volumeUsd24Hr") or 0)
                market_cap = float(coin.get("marketCapUsd") or 0)
            except Exception:
                p24, current_price, volume, market_cap = 0, 0, 0, 0

            volume_ratio = volume / market_cap if market_cap > 0 else 0

            # Calculate raw pump_score based on change, volume ratio, and bonus criteria.
            score = 0
            if p24 > 0:
                score += p24 * 2.0
            score += min(volume_ratio, 1.0) * 30
            if 0 < market_cap < 10e6:
                score += 25
            elif 10e6 <= market_cap < 50e6:
                score += 10
            if 0 < current_price < 0.001:
                score += 15
            pump_score_raw = round(score, 2)
            pump_scores_all.append(pump_score_raw)

            coin_id = coin.get("id")
            # Calculate a dynamic EMA alpha based on historical volatility.
            dyn_alpha = dynamic_alpha(coin_id, self.ema_alpha)
            prev_ema = prev_ema_scores.get(coin_id, pump_score_raw)
            pump_score = exponential_moving_average(pump_score_raw, prev_ema, dyn_alpha)
            pump_score = round(pump_score, 2)
            prev_ema_scores[coin_id] = pump_score

            # Update history for sparkline visualization, volatility, and momentum calculation.
            history = update_history(coin_id, pump_score)
            volatility = compute_volatility(history)
            pump_volatility[coin_id] = round(volatility, 2)
            momentum = calculate_momentum(coin_id)

            # Calculate potential_score with an additive growth factor.
            potential_score = pump_score
            if market_cap > 0:
                growth_factor = (p24 / market_cap) * 1e9  # Scale factor for growth
                potential_score += growth_factor
            potential_score = round(potential_score, 2)

            processed.append({
                "id": coin_id,
                "name": coin.get("name"),
                "symbol": coin.get("symbol"),
                "current_price": current_price,
                "price_change_24h": round(p24, 2),
                "total_volume": volume,
                "market_cap": market_cap,
                "volume_ratio": round(volume_ratio, 2),
                "raw_pump_score": pump_score_raw,
                "pump_score": pump_score,
                "momentum": momentum,
                "volatility": pump_volatility.get(coin_id),
                "potential_score": potential_score,
                "history": make_sparkline(history)
            })

        # Secondary processing: compute z-scores to standardize pump_scores.
        if pump_scores_all:
            avg_score = statistics.mean(pump_scores_all)
            std_score = statistics.stdev(pump_scores_all) if len(pump_scores_all) > 1 else 1
        else:
            avg_score, std_score = 0, 1

        for coin in processed:
            z_score = (coin["pump_score"] - avg_score) / std_score
            coin["z_score"] = round(z_score, 2)
            # Set elite flag if coin exhibits strong signals.
            coin["elite"] = (z_score > 1 and coin["momentum"] > 0.5 and coin["potential_score"] >= 80)
            coin["alert"] = coin["potential_score"] >= 80

        return processed

    def run_once(self):
        """Fetch and process the coin data once."""
        raw_data = self.fetch_data()
        return self.analyze_data(raw_data)

def generate_header():
    """Generate the stylized ASCII header for the terminal UI."""
    header_text = Text()
    header_text.append("\n  __  __                       ____            _   \n", style="bold cyan")
    header_text.append(" |  \\/  | ___  _ __ ___  _   _|  _ \\ ___  __ _| |_ \n", style="bold cyan")
    header_text.append(" | |\\/| |/ _ \\| '_ ` _ \\| | | | |_) / _ \\/ _` | __|\n", style="bold cyan")
    header_text.append(" | |  | | (_) | | | | | | |_| |  _ <  __/ (_| | |_ \n", style="bold cyan")
    header_text.append(" |_|  |_|\\___/|_| |_| |_|\\__, |_| \\_\\___|\\__,_|\\__|\n", style="bold cyan")
    header_text.append("                        |___/                    \n", style="bold cyan")
    header_text.append("Meme Coin Pump & Ultra-Precision Ultimate Detector\n", style="bold yellow")
    return header_text

def build_table(data):
    """
    Build a rich table displaying the processed coin data including advanced signals.
    Columns: ID, Name, Symbol, Price, 24h%, Volume, MktCap, VolRatio, PumpScore, Momentum,
    Volatility, Trend, History, Potential, Elite flag, z-Score, Alert flag.
    """
    table = Table(box=DOUBLE_EDGE, border_style="bright_blue", show_lines=True)
    table.add_column("ID", style="magenta", no_wrap=True)
    table.add_column("Name", style="bold white")
    table.add_column("Sym", justify="center", style="bright_green")
    table.add_column("Price", justify="right", style="cyan")
    table.add_column("24h%", justify="right", style="yellow")
    table.add_column("Volume", justify="right", style="green")
    table.add_column("MktCap", justify="right", style="bright_blue")
    table.add_column("VolRatio", justify="right", style="blue")
    table.add_column("PumpScore", justify="right", style="red")
    table.add_column("Momnt", justify="right", style="white")
    table.add_column("Vol", justify="right", style="white")
    table.add_column("Trend", justify="center", style="bright_white")
    table.add_column("History", justify="center", style="white")
    table.add_column("Potential", justify="right", style="red")
    table.add_column("Elite", justify="center", style="bold magenta")
    table.add_column("z-Score", justify="center", style="bold cyan")
    table.add_column("Alert", justify="center", style="bold red")

    # Sort coins by potential_score in descending order and display top 10.
    for coin in sorted(data, key=lambda x: x["potential_score"], reverse=True)[:10]:
        trend_symbol = "↑" if coin["momentum"] > 0 else "↓" if coin["momentum"] < 0 else "→"
        elite_flag = "[bold magenta]YES[/bold magenta]" if coin.get("elite") else "[dim]NO[/dim]"
        alert_flag = "[bold red]YES[/bold red]" if coin["alert"] else "[green]NO[/green]"
        table.add_row(
            coin["id"][:15],
            coin["name"][:20],
            coin["symbol"][:7],
            f"{coin['current_price']:10.4f}",
            f"{coin['price_change_24h']:6.2f}",
            f"{coin['total_volume']:12,.0f}",
            f"{coin['market_cap']:12,.0f}",
            f"{coin['volume_ratio']:9.2f}",
            f"{coin['pump_score']:10.2f}",
            f"{coin['momentum']:6.2f}",
            f"{coin['volatility']:6.2f}",
            trend_symbol,
            coin["history"],
            f"{coin['potential_score']:10.2f}",
            elite_flag,
            f"{coin['z_score']:6.2f}",
            alert_flag
        )
    return table

def build_statistics(data):
    """Construct a statistics panel summarizing overall metrics of the analyzed coins."""
    total_coins = len(data)
    if total_coins > 0:
        top_coin = max(data, key=lambda x: x["potential_score"])
        avg_potential = statistics.mean([coin["potential_score"] for coin in data])
        stats_str = (
            f"Total Coins: {total_coins} | "
            f"Top Coin: {top_coin['name']} [{top_coin['symbol']}] (Potential: {top_coin['potential_score']}) | "
            f"Avg Potential: {avg_potential:.2f}"
        )
    else:
        stats_str = "No coin data available."
    return Panel(Text(stats_str, style="bold magenta"), border_style="magenta")

def build_layout(data, update_interval):
    """Assemble the terminal layout including header, update info, coin table, and statistics panel."""
    header = generate_header()
    table = build_table(data)
    stats = build_statistics(data)
    update_info = Panel(
        f"Last Update: {time.strftime('%Y-%m-%d %H:%M:%S')} | Updating every {update_interval} sec",
        style="bold yellow", border_style="yellow"
    )
    layout = Layout()
    layout.split_column(
        Layout(header, name="header", size=9),
        Layout(update_info, name="update_info", size=3),
        Layout(table, name="table"),
        Layout(stats, name="stats", size=3)
    )
    return layout

def run_terminal_mode(update_interval, min_market_cap, max_market_cap, ema_alpha):
    """Run the detector in live terminal mode with continuously updating data."""
    detector = MemeCoinPumpDetector(
        update_interval=update_interval,
        min_market_cap=min_market_cap,
        max_market_cap=max_market_cap,
        ema_alpha=ema_alpha
    )
    with Live(console=console, screen=True, refresh_per_second=4) as live:
        try:
            while True:
                data = detector.run_once()
                layout = build_layout(data, update_interval)
                live.update(Align.center(layout))
                time.sleep(update_interval)
        except KeyboardInterrupt:
            console.print("\nExiting program.", style="bold red")
            sys.exit()

def run_once_mode(min_market_cap, max_market_cap, ema_alpha):
    """Run the detector once and output the JSON formatted result."""
    detector = MemeCoinPumpDetector(
        min_market_cap=min_market_cap,
        max_market_cap=max_market_cap,
        ema_alpha=ema_alpha
    )
    data = detector.run_once()
    from rich.json import JSON
    console.print(JSON.from_data(data))

def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Meme Coin Pump & Ultra-Precision Ultimate Detector with advanced analytics."
    )
    parser.add_argument('--mode', choices=['terminal', 'once'], default='terminal',
                        help="Mode: 'terminal' for live updates, 'once' for a single run")
    parser.add_argument('--interval', type=int, default=1,
                        help="Data update interval in seconds (default: 1)")
    parser.add_argument('--min_market_cap', type=float, default=0,
                        help="Minimum market cap in USD")
    parser.add_argument('--max_market_cap', type=float, default=float("inf"),
                        help="Maximum market cap in USD (default: no maximum)")
    parser.add_argument('--ema_alpha', type=float, default=0.3,
                        help="Default EMA alpha value for smoothing (default: 0.3)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if args.mode == 'terminal':
        run_terminal_mode(args.interval, args.min_market_cap, args.max_market_cap, args.ema_alpha)
    elif args.mode == 'once':
        run_once_mode(args.min_market_cap, args.max_market_cap, args.ema_alpha)
