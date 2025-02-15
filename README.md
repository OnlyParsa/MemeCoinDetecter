# Meme Coin Pump & Ultra-Precision Ultimate Detector

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/OnlyParsa/memebot)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-blueviolet)](https://github.com/OnlyParsa/memebot/releases)

Welcome to the **Meme Coin Pump & Ultra-Precision Ultimate Detector** – a cutting-edge tool designed for crypto enthusiasts and traders who want to identify high-potential meme coins in real time.

## Overview

This production-ready tool leverages advanced techniques such as:
- **Dynamic EMA Smoothing**: Adaptive exponential moving average to capture short-term trends.
- **Momentum & Volatility Analysis**: Measuring momentum shifts and market volatility for robust detection.
- **Multi-Dimensional Scoring**: Combining raw pump scores, growth factors, and z-score standardization to highlight elite coins.
- **Resilient Data Fetching**: Exponential backoff handling for API requests ensuring reliability.
- **Live Terminal UI**: Beautiful interface built with [Rich](https://github.com/willmcgugan/rich) that dynamically updates with real-time market data.

It is engineered to sift through thousands of coins and isolate the ones with the highest potential, giving you an edge in the rapidly evolving crypto market.

## Features

- **Real-Time Data Updates**: Continuous monitoring and processing of live data from the CoinCap API.
- **Advanced Metrics Calculation**: Computes raw pump scores, EMA-smoothed values, momentum, volatility, potential scores, and standardized z-scores.
- **Elite Signal Highlights**: Coins meeting pre-defined criteria are marked as "elite."
- **Interactive Terminal Dashboard**: View detailed coin stats, dynamic sparklines, and overall market statistics.
- **Robust Networking**: Integrated retry mechanisms for resilient API communication.
- **Customizable Parameters**: Configure update intervals, market cap filters, and smoothing sensitivity via command-line arguments.

## Requirements

- **Python 3.7+**
- **Packages**:
  - `requests`
  - `rich`
  
Install the required packages using pip:

```bash
pip install requests rich
```

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/OnlyParsa/memebot.git
   cd memebot
   ```
2. **Review the Code**: Open the `main2.py` file to familiarize yourself with the structure and configuration.

## Usage

The detector supports two modes:
- **Terminal Mode**: Live updates in your terminal.
- **Once Mode**: Single data fetch and JSON output analysis.

### Terminal Mode

Run the detector in live mode with dynamic updates:

```bash
python main2.py --mode terminal --interval 1 --min_market_cap 0 --max_market_cap 1e12 --ema_alpha 0.3
```

### Once Mode

Fetch the data once and output the analysis in JSON format:

```bash
python main2.py --mode once --min_market_cap 0 --max_market_cap 1e12 --ema_alpha 0.3
```

## Command-Line Arguments

- `--mode`: Operation mode (`terminal` or `once`). Default is `terminal`.
- `--interval`: Update interval in seconds (default: `1`).
- `--min_market_cap`: Minimum market capitalization filter in USD.
- `--max_market_cap`: Maximum market capitalization filter in USD.
- `--ema_alpha`: Default EMA alpha (smoothing factor) value (default: `0.3`).

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Create a new Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions, suggestions, or bug reports, please open an issue on GitHub or contact the maintainer at [OnlyParsa](https://github.com/OnlyParsa).

---

*Happy Trading!*
