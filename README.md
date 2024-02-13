# ML_Stock_Data_Analysis

# Stock Analysis Toolkit

This toolkit consists of two main Python scripts designed to fetch, analyze, and visualize stock market data. Utilizing the Alpha Vantage API, pandas, NumPy, sklearn, and matplotlib, these scripts offer a comprehensive suite of tools for individual investors or analysts interested in stock market data.

## Scripts Overview

### `fetch2.py` - Stock Data Fetcher

**Purpose:** Fetches daily stock data for a given symbol from Alpha Vantage and saves it to a CSV file.

**Features:**
- Fetch daily open, high, low, close, and volume data.
- Save fetched data to CSV for further analysis.

**Requirements:**
- Alpha Vantage API key (replace `'YOUR_API_KEY'` with your actual API key in the script).

### `p.py` - Stock Analysis and Visualization

**Purpose:** Offers multiple analysis and visualization options for stock market data, including price prediction, buy/sell signal classification, and technical indicator analysis.

**Features:**
- Stock price prediction using linear regression.
- Buy/Sell signal generation based on simple classification.
- Technical indicator analysis (50-day moving average and RSI).

**How to Use:**
1. Ensure you have a CSV file with stock data (can be generated using `fetch2.py`).
2. Run the script and choose one of the analysis or visualization options.

**Requirements:**
- Stock data in CSV format.

## Getting Started

1. Install the required Python packages:

```bash
pip install pandas numpy matplotlib sklearn alpha_vantage
```

2. Obtain an Alpha Vantage API key from [their website](https://www.alphavantage.co/).

3. Run `fetch2.py` to fetch stock data:

```bash
python fetch2.py
```

4. Run `p.py` to perform analysis or visualization on the fetched data:

```bash
python p.py
```

## License

This toolkit is open-source and free to use. Please ensure you comply with Alpha Vantage's API terms of use.
