import os
from datetime import datetime, timedelta
from scripts.data_fetch import SP500TickerScraper, TiingoDataFetcher, IndexDataFetcher, FREDDataFetcher


def test_data_fetcher():
    """
    Function to test the various data fetchers for SP500 stock tickers, stock prices, index data, and FRED data.

    This function performs the following tests:
    1. Fetch the list of S&P 500 companies.
    2. Fetch stock price data for a subset of tickers from the Tiingo API.
    3. Fetch S&P 500 index data using yfinance.
    4. Fetch economic data (e.g., Treasury rate) using the FRED API.

    Test data is saved to a 'data/test/' directory.
    """

    # Set up test parameters
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    api_key = '56054c41dd125c5b9a2678c7b34451d999f09d39'  # Replace with your Tiingo API key
    test_save_dir = 'data/test/'  # Directory for saving test data
    os.makedirs(test_save_dir, exist_ok=True)

    # Test SP500TickerScraper
    ticker_scraper = SP500TickerScraper()
    all_tickers = ticker_scraper.get_sp500_companies()
    test_tickers = all_tickers[:5]  # Take first 5 tickers for testing
    print(f"Testing with tickers: {test_tickers}")  # Display the tickers being tested

    # Test TiingoDataFetcher
    tiingo_fetcher = TiingoDataFetcher(api_key, test_tickers, start_date, end_date, save_dir=test_save_dir)
    tiingo_fetcher.fetch_data()

    # Test IndexDataFetcher
    index_fetcher = IndexDataFetcher(start_date, end_date, save_dir=test_save_dir)
    index_fetcher.fetch_data()

    # Test FREDDataFetcher
    fred_series_id = 'DGS1'
    fred_fetcher = FREDDataFetcher(fred_series_id, start_date, end_date, save_dir=test_save_dir)
    fred_fetcher.fetch_data()

    print("Test completed. Check the 'data/test/' directory for results.")


if __name__ == '__main__':
    test_data_fetcher()
