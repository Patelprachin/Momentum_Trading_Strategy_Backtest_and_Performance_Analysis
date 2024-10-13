import requests
import pandas as pd
import os
import time
import pandas_datareader.data as web
from tqdm import tqdm
import logging
from ratelimit import limits, sleep_and_retry
import pickle
import yfinance as yf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SP500TickerScraper:
    """
    This class scrapes the list of S&P 500 companies from Wikipedia and saves it as a CSV file.
    """

    def __init__(self, url='https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'):
        """
        Initializes the scraper with a URL to fetch the S&P 500 companies.

        :param url: URL of the Wikipedia page containing the list of S&P 500 companies.
        """

        self.url = url
        self.save_dir = '/Users/macbook/Desktop/Farrer_Quant_Assignment/data/raw/'  # Directory to save the scraped data
        os.makedirs(self.save_dir, exist_ok=True)  # Ensure save directory exists

    def get_sp500_companies(self):
        """
        Scrapes the S&P 500 company data from Wikipedia and saves it as a CSV file.

        :return: List of S&P 500 ticker symbols.
        """

        try:
            # Scrape the first table on the Wikipedia page and clean up ticker symbols
            df = pd.read_html(self.url)[0]
            df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
            csv_filename = os.path.join(self.save_dir, 'sp500_companies.csv')
            df.to_csv(csv_filename, index=False)  # Save the list to a CSV file
            logging.info(f'S&P 500 company data saved to {csv_filename}')
            return df['Symbol'].tolist()  # Return the list of ticker symbols
        except Exception as e:
            logging.error(f'Error occurred while fetching tickers: {e}')
            return []


class TiingoDataFetcher:
    """
    This class fetches historical stock data for S&P 500 companies from the Tiingo API and saves them as CSV files.
    """

    def __init__(self, api_key, tickers, start_date, end_date, save_dir='/Users/macbook/Desktop/Farrer_Quant_Assignment/data/raw/'):
        """
        Initializes the Tiingo data fetcher with API key, tickers, date range, and save directory.

        :param api_key: Tiingo API key.
        :param tickers: List of stock ticker symbols to fetch data for.
        :param start_date: Start date for fetching historical data.
        :param end_date: End date for fetching historical data.
        :param save_dir: Directory to save the stock data files.
        """

        self.api_key = api_key
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)  # Ensure save directory exists
        self.checkpoint_file = os.path.join(self.save_dir, 'fetch_checkpoint.pkl')  # File to save progress checkpoints

    @sleep_and_retry
    @limits(calls=50, period=3600)  # Limit to 50 API calls per hour
    def fetch_data_for_ticker(self, ticker):
        """
        Fetches historical price data for a specific ticker from the Tiingo API.

        :param ticker: Stock ticker symbol.
        :return: DataFrame containing date, adjusted close price, and ticker symbol.
        """

        headers = {'Content-Type': 'application/json'}
        url = f'https://api.tiingo.com/tiingo/daily/{ticker}/prices'
        params = {
            'token': self.api_key,
            'startDate': self.start_date,
            'endDate': self.end_date,
            'format': 'json'
        }
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            df = pd.DataFrame(data)
            df = df[['date', 'adjClose']]  # Only keep date and adjusted close
            df['ticker'] = ticker  # Add a column for the ticker symbol
            return df
        except requests.exceptions.RequestException as e:
            logging.error(f'Error fetching data for {ticker}: {e}')
            return pd.DataFrame()

    def fetch_data(self):
        """
         Fetches historical data for all tickers, saves them as individual CSV files, and combines them into one CSV file.
        """

        fetched_tickers = self.load_checkpoint()  # Load already fetched tickers from checkpoint
        remaining_tickers = [ticker for ticker in self.tickers if ticker not in fetched_tickers]

        # Iterate over remaining tickers to fetch their data
        for ticker in tqdm(remaining_tickers, desc="Fetching stock data"):
            df = self.fetch_data_for_ticker(ticker)
            if not df.empty:
                # Save individual ticker data to a CSV file
                csv_filename = os.path.join(self.save_dir, f'{ticker}_data.csv')
                df.to_csv(csv_filename, index=False)
                fetched_tickers.append(ticker)  # Add the ticker to the fetched list
                self.save_checkpoint(fetched_tickers)  # Save progress in the checkpoint
            time.sleep(1)  # Small delay between requests to avoid hitting API rate limits

        logging.info("All data fetched. Combining files...")
        self.combine_csv_files()  # Combine all fetched data into a single CSV file

    def load_checkpoint(self):
        """
        Loads the list of fetched tickers from the checkpoint file.

        :return: List of tickers for which data has already been fetched.
        """

        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return []

    def save_checkpoint(self, fetched_tickers):
        """
        Saves the list of fetched tickers to the checkpoint file.

        :param fetched_tickers: List of tickers that have already been fetched.
        """

        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(fetched_tickers, f)

    def combine_csv_files(self):
        """
        Combines individual stock data CSV files into one CSV file and deletes the individual files.
        """

        all_files = [f for f in os.listdir(self.save_dir) if f.endswith('_data.csv')]
        combined_data = []

        # Read and combine data from each individual file
        for filename in all_files:
            df = pd.read_csv(os.path.join(self.save_dir, filename))
            combined_data.append(df)

        # Combine all data into a single DataFrame
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_df.to_csv(os.path.join(self.save_dir, 'sp500_stocks_data.csv'), index=False)
        logging.info(f'Combined S&P 500 stock data saved to sp500_stocks_data.csv')

        # Remove individual CSV files
        for filename in all_files:
            os.remove(os.path.join(self.save_dir, filename))


class YahooFinanceFallbackFetcher:
    """
    This class fetches historical stock data for specified tickers using Yahoo Finance API
    when Tiingo API fails to provide data due to the monthly 500 unique stock ticker request limit.
    """

    def __init__(self, tickers, start_date, end_date, save_dir='/Users/macbook/Desktop/Farrer_Quant_Assignment/data/raw/'):
        """
        Initializes the Yahoo Finance fallback fetcher with tickers, date range, and save directory.

        :param tickers: List of stock ticker symbols to fetch data for.
        :param start_date: Start date for fetching historical data.
        :param end_date: End date for fetching historical data.
        :param save_dir: Directory to save the stock data files.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def fetch_data(self):
        """
        Fetches historical data for all specified tickers and saves them as a single CSV file.
        """
        all_data = []
        for ticker in tqdm(self.tickers, desc="Fetching fallback stock data"):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=self.start_date, end=self.end_date)
                df = df.reset_index()
                df = df[['Date', 'Close']]  # Only keep date and close price
                df.columns = ['date', 'adjClose']  # Rename columns for consistency
                df['ticker'] = ticker
                all_data.append(df)
                logging.info(f"Successfully fetched data for {ticker} using Yahoo Finance API")
            except Exception as e:
                logging.error(f"Error fetching data for {ticker} from Yahoo Finance: {e}")

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            csv_filename = os.path.join(self.save_dir, 'fallback_stocks_data.csv')
            combined_df.to_csv(csv_filename, index=False)
            logging.info(f"Fallback stock data saved to {csv_filename}")
        else:
            logging.warning("No fallback data was successfully fetched.")


class IndexDataFetcher:
    """
    This class fetches historical data for the S&P 500 index (^GSPC) using yfinance and saves it as a CSV file.
    """

    def __init__(self, start_date, end_date, save_dir='/Users/macbook/Desktop/Farrer_Quant_Assignment/data/raw/'):
        """
        Initializes the index data fetcher with a date range and save directory.

        :param start_date: Start date for fetching historical index data.
        :param end_date: End date for fetching historical index data.
        :param save_dir: Directory to save the index data file.
        """

        self.start_date = start_date
        self.end_date = end_date
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)  # Ensure save directory exists

    def fetch_data(self):
        """
        Fetches historical data for the S&P 500 index (^GSPC) and saves it as a CSV file.
        """

        try:
            # Use yfinance to download the data
            df = yf.download('^GSPC', start=self.start_date, end=self.end_date)
            df = df[['Adj Close']].reset_index()
            df.columns = ['date', 'adjClose']  # Rename columns for consistency
            df['ticker'] = 'SPX'  # Add a ticker column for the index
            csv_filename = os.path.join(self.save_dir, 'sp500_index.csv')
            df.to_csv(csv_filename, index=False)  # Save index data to CSV
            logging.info(f'S&P 500 index data saved to {csv_filename}')
        except Exception as e:
            logging.error(f'Error fetching S&P 500 index data: {e}')


class FREDDataFetcher:
    """
    This class fetches economic data from the Federal Reserve Economic Data (FRED) using pandas_datareader.
    """

    def __init__(self, series_id, start_date, end_date, save_dir='/Users/macbook/Desktop/Farrer_Quant_Assignment/data/raw/'):
        """
        Initializes the FRED data fetcher with a FRED series ID, date range, and save directory.

        :param series_id: FRED series ID to fetch (e.g., 'DGS1' for 1-year Treasury rate).
        :param start_date: Start date for fetching the data.
        :param end_date: End date for fetching the data.
        :param save_dir: Directory to save the data file.
        """

        self.series_id = series_id
        self.start_date = start_date
        self.end_date = end_date
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)  # Ensure save directory exists

    def fetch_data(self):
        """
        Fetches data from the FRED API for the given series and saves it as a CSV file.
        """

        try:
            # Use pandas_datareader to fetch data from FRED
            df = web.DataReader(self.series_id, 'fred', self.start_date, self.end_date)
            csv_filename = os.path.join(self.save_dir, f'{self.series_id}.csv')
            df.to_csv(csv_filename)  # Save data to CSV
            logging.info(f'Data for {self.series_id} saved to {csv_filename}')
        except Exception as e:
            logging.error(f'Error fetching data for {self.series_id}: {e}')


def main():
    """
    Main function to scrape S&P 500 tickers, fetch historical stock, index, and FRED data, and save them as CSV files.
    """

    start_date = '2012-01-01'
    end_date = '2024-10-10'  # Can replace it with "datetime.now().strftime('%Y-%m-%d')"
    api_key = '56054c41dd125c5b9a2678c7b34451d999f09d39'  # Replace with your Tiingo API key

    # Step 1: Scrape S&P 500 companies
    ticker_scraper = SP500TickerScraper()
    tickers = ticker_scraper.get_sp500_companies()
    logging.info(f'Total tickers fetched: {len(tickers)}')

    # Step 2: Fetch historical stock data from Tiingo
    tiingo_fetcher = TiingoDataFetcher(api_key, tickers, start_date, end_date)
    tiingo_fetcher.fetch_data()

    # Step 3: Use YahooFinanceFallbackFetcher for remaining stocks
    fallback_tickers = ['ZBRA', 'ZBH', 'ZTS']
    fallback_fetcher = YahooFinanceFallbackFetcher(fallback_tickers, start_date, end_date)
    fallback_fetcher.fetch_data()

    # Step 4: Fetch S&P 500 index data from yfinance
    index_fetcher = IndexDataFetcher(start_date, end_date)
    index_fetcher.fetch_data()

    # Step 5: Fetch Risk Free-Rate data from FRED
    fred_series_id = 'DGS1'  # 1-Year Treasury Constant Maturity Rate
    fred_fetcher = FREDDataFetcher(fred_series_id, start_date, end_date)
    fred_fetcher.fetch_data()

    logging.info("All data fetching completed. You can now proceed with data cleaning and merging in data_cleaning.py.")


if __name__ == '__main__':
    main()
