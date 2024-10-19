# Momentum Factor Performance Analysis Using S&P 500 Data

## Project Overview
This project conducts a comprehensive analysis of momentum-based trading strategies applied to S&P 500 equities. The analysis evaluates various momentum horizons, rebalancing frequencies, and smoothing techniques to provide insights into their impact on strategy performance.

## Objectives
- Calculate momentum across various horizons (1, 3, 6, and 12 months) and lagged momentum
- Evaluate factor performance using cross-sectional z-scores
- Conduct backtests to assess the effectiveness of different momentum strategies
- Analyze the impact of changes in momentum horizons and smoothing inputs on performance metrics and turnover

## Data Sources
- S&P 500 equities data from Tiingo API with the hourly limit of 50 stock tickers and monthly limit of 500 stock tickers. When running the data_fethcing.py file please make sure to put your own API key and change the hourly request limit if necessary.
- S&P 500 company list scraped from Wikipedia
- S&P 500 index data from Yahoo Finance for the 3 additional stock tickers since Tiingo has a monthly limit of 500 stock tickers.
- Risk-free rate data from Federal Reserve's FRED database (optional)

## Methodology
1. Data Fetching and Cleaning
2. Momentum Calculations (including lagged momentum)
3. Backtesting Framework Implementation
4. Performance Analysis and Visualization

## Key Findings
- Longer momentum horizons (12-month) consistently outperformed shorter horizons
- Monthly rebalancing showed superior performance compared to weekly rebalancing
- Applying smoothing to z-scores significantly enhanced strategy performance
- Lagged momentum strategy demonstrated substantial outperformance compared to current momentum approach

## Limitations and Future Work
- Limited backtesting period (5 years) coinciding with a largely bullish market
- Need for extended backtesting across different market cycles
- Potential for incorporating fundamental signals and exploring alternative portfolio construction methods

## Next Steps
1. Conduct extended backtesting over longer periods
2. Perform econometric analysis to ensure statistical significance
3. Explore combined momentum signals and incorporation of fundamental factors
4. Investigate the use of leverage and short positions
5. Experiment with advanced smoothing techniques and robust performance metrics

## How to Use
1. Clone the GitHub repository

2. Set up a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required Python libraries:
   ```
   pip install -r requirements.txt
   ```
   

4. Run the scripts in the scripts directory in the following order:
   ```
   1. data_fetching.py
   2. data_cleaning.py
   3. momentum_calculation.py
   4. backtesting_sensitivity_analysis.py
   5. backtesting_lagged.py
   6. backtesting_optimal.py
   ```

5. View the results:
    - Plots and summarised results are saved in the /data/backtesting directory.
    - The final report with all the analysis and results can be found in the /reports directory

Note: Ensure you have Python 3.7+ installed on your system before starting.

## Contributors
Prachin Patel
