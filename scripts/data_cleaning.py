import pandas as pd
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataCleaner:
    """
    This class cleans all the relevant raw data files for further analysis.
    """

    def __init__(self, raw_data_path, cleaned_data_path):
        """
        Initializes the DataCleaner with paths to raw and cleaned data.

        :param raw_data_path (str): Path to the directory containing raw data files.
        :param cleaned_data_path (str): Path to the directory where cleaned data will be saved.
        """

        self.raw_data_path = raw_data_path
        self.cleaned_data_path = cleaned_data_path
        os.makedirs(self.cleaned_data_path, exist_ok=True)

    def load_stock_data(self):
        """
        Loads and combines stock data from main and fallback CSV files.

        :return: pandas.DataFrame: A DataFrame containing combined stock data with 'date' and 'ticker' as a MultiIndex.
        """

        stock_data_path = os.path.join(self.raw_data_path, 'sp500_stocks_data.csv')
        fallback_data_path = os.path.join(self.raw_data_path, 'fallback_stocks_data.csv')

        df = pd.read_csv(stock_data_path)
        fallback_df = pd.read_csv(fallback_data_path)

        # Combine main stock data with fallback data
        df = pd.concat([df, fallback_df], ignore_index=True)

        # Parse dates with errors='coerce' and utc=True to handle timezones
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        df['date'] = df['date'].dt.tz_convert(None)  # Convert to naive datetime

        # Drop rows with NaT in 'date'
        df.dropna(subset=['date'], inplace=True)

        # Set 'date' and 'ticker' as a MultiIndex
        df.set_index(['date', 'ticker'], inplace=True)
        df.sort_index(inplace=True)  # Ensure the MultiIndex is sorted

        return df

    def load_index_data(self):
        """
        Loads S&P 500 index data from a CSV file.

        :return: pandas.DataFrame: A DataFrame containing index data with 'date' as the index.
        """

        index_data_path = os.path.join(self.raw_data_path, 'sp500_index.csv')
        df = pd.read_csv(index_data_path)

        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        df['date'] = df['date'].dt.tz_convert(None)  # Convert to naive datetime

        # Drop rows with NaT in 'date'
        df.dropna(subset=['date'], inplace=True)

        # Set 'date' as the index
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        return df

    def load_risk_free_rate(self):
        """
        Loads the risk-free rate data from a CSV file.

        :return: pandas.DataFrame: A DataFrame containing risk-free rate data with 'DATE' as the index.
        """

        rf_data_path = os.path.join(self.raw_data_path, 'DGS1.csv')
        df = pd.read_csv(rf_data_path)

        # Parse dates with errors='coerce' and utc=True to handle timezones
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce', utc=True)
        df['DATE'] = df['DATE'].dt.tz_convert(None)  # Convert to naive datetime

        # Drop rows with NaT in 'DATE'
        df.dropna(subset=['DATE'], inplace=True)

        # Set 'DATE' as the index
        df.set_index('DATE', inplace=True)

        # Rename the column to 'risk_free_rate' and convert percentage to decimal
        df = df.rename(columns={'DGS1': 'risk_free_rate'})
        df['risk_free_rate'] = df['risk_free_rate'] / 100  # Convert percentage to decimal

        df.sort_index(inplace=True)  # Ensure the index is sorted

        return df

    def clean_stock_data(self, df):
        """
        Cleans stock data by removing NaNs, outliers, and insufficient data.

        :param df (pandas.DataFrame): The raw stock data DataFrame.

        :return: tuple:
                 pandas.DataFrame: Cleaned stock data.
                 list: List of tickers removed due to insufficient data.
        """

        # Remove any rows with NaN values in 'adjClose'
        df = df.dropna(subset=['adjClose'])

        # Remove outliers
        df = self.remove_outliers(df, column='adjClose', threshold=5)

        # Ensure all tickers have sufficient historical data (e.g., at least 1 year)
        min_periods = 252  # Approximately 1 year of trading days
        ticker_counts = df.groupby('ticker').size()
        valid_tickers = ticker_counts[ticker_counts >= min_periods].index
        removed_tickers = ticker_counts[ticker_counts < min_periods].index.tolist()
        df = df[df.index.get_level_values('ticker').isin(valid_tickers)]

        return df, removed_tickers

    @staticmethod
    def remove_outliers(df, column, threshold):
        """
        Removes outliers from a DataFrame based on z-score threshold.

        :param df (pandas.DataFrame): The DataFrame from which to remove outliers.
        :param column (str): The column name on which to calculate z-scores.
        :param threshold (float): The z-score threshold to use for identifying outliers.

        :return:    pandas.DataFrame: DataFrame with outliers removed.
        """

        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return df[z_scores < threshold]

    @staticmethod
    def align_dates(stock_data, index_data, risk_free_rate):
        """
        Aligns the dates across stock data, index data, and risk-free rate data.

        :param stock_data (pandas.DataFrame): The stock data DataFrame.
        :param index_data (pandas.DataFrame): The index data DataFrame.
        :param risk_free_rate (pandas.DataFrame): The risk-free rate DataFrame.

        :return: tuple:
                 pandas.DataFrame: Aligned stock data.
                 pandas.DataFrame: Aligned index data.
                 pandas.DataFrame: Aligned risk-free rate data.
        """
        # Find the common date range
        start_date = max(stock_data.index.get_level_values('date').min(),
                         index_data.index.min(),
                         risk_free_rate.index.min())
        end_date = min(stock_data.index.get_level_values('date').max(),
                       index_data.index.max(),
                       risk_free_rate.index.max())

        logging.info(f"Aligning data from {start_date} to {end_date}")

        # Use pd.IndexSlice for slicing MultiIndex
        idx = pd.IndexSlice

        # Filter data to the common date range
        stock_data = stock_data.loc[idx[start_date:end_date, :], :]
        index_data = index_data.loc[start_date:end_date]
        risk_free_rate = risk_free_rate.loc[start_date:end_date]

        return stock_data, index_data, risk_free_rate

    @staticmethod
    def calculate_returns(df):
        """
        Calculates returns from adjusted close prices.

        :param df (pandas.DataFrame): The DataFrame containing 'adjClose' prices.

        :return: pandas.DataFrame: A DataFrame containing the returns.
        """

        if 'ticker' in df.index.names:
            returns = df.groupby('ticker')['adjClose'].pct_change()
        else:
            returns = df['adjClose'].pct_change()
        return returns.to_frame('returns')

    def save_cleaned_data(self, stock_data, stock_returns, index_data, index_returns, risk_free_rate):
        """
        Saves the cleaned data and calculated returns to CSV files.

        :param stock_data (pandas.DataFrame): Cleaned stock data.
        :param stock_returns (pandas.DataFrame): Calculated stock returns.
        :param index_data (pandas.DataFrame): Cleaned index data.
        :param index_returns (pandas.DataFrame): Calculated index returns.
        :param risk_free_rate (pandas.DataFrame): Cleaned risk-free rate data.
        """

        stock_data.to_csv(os.path.join(self.cleaned_data_path, 'cleaned_stock_data.csv'))
        stock_returns.to_csv(os.path.join(self.cleaned_data_path, 'stock_returns.csv'))
        index_data.to_csv(os.path.join(self.cleaned_data_path, 'cleaned_index_data.csv'))
        index_returns.to_csv(os.path.join(self.cleaned_data_path, 'index_returns.csv'))
        risk_free_rate.to_csv(os.path.join(self.cleaned_data_path, 'cleaned_risk_free_rate.csv'))
        logging.info("Cleaned data saved successfully.")

    @staticmethod
    def print_data_shapes(raw_stock_data, cleaned_stock_data, raw_index_data,
                          cleaned_index_data, raw_rf_data, cleaned_rf_data):
        """
        Logs the shapes of raw and cleaned datasets for comparison.

        :param raw_stock_data (pandas.DataFrame): Raw stock data.
        :param cleaned_stock_data (pandas.DataFrame): Cleaned stock data.
        :param raw_index_data (pandas.DataFrame): Raw index data.
        :param cleaned_index_data (pandas.DataFrame): Cleaned index data.
        :param raw_rf_data (pandas.DataFrame): Raw risk-free rate data.
        :param cleaned_rf_data (pandas.DataFrame): Cleaned risk-free rate data.
        """

        logging.info(f"Raw stock data shape: {raw_stock_data.shape}")
        logging.info(f"Cleaned stock data shape: {cleaned_stock_data.shape}")
        logging.info(f"Raw index data shape: {raw_index_data.shape}")
        logging.info(f"Cleaned index data shape: {cleaned_index_data.shape}")
        logging.info(f"Raw risk-free rate data shape: {raw_rf_data.shape}")
        logging.info(f"Cleaned risk-free rate data shape: {cleaned_rf_data.shape}")

    def run_cleaning_process(self):
        """
        Runs the entire data cleaning process: loading data, cleaning, aligning dates,
        calculating returns, printing data shapes, and saving the cleaned data.
        """

        # Load data
        raw_stock_data = self.load_stock_data()
        raw_index_data = self.load_index_data()
        raw_rf_data = self.load_risk_free_rate()

        # Clean stock data
        cleaned_stock_data, removed_tickers = self.clean_stock_data(raw_stock_data)

        # Print removed tickers
        logging.info(f"Tickers removed due to insufficient data: {removed_tickers}")

        # Align dates across all datasets
        cleaned_stock_data, cleaned_index_data, cleaned_rf_data = self.align_dates(
            cleaned_stock_data, raw_index_data, raw_rf_data)

        # Calculate returns
        stock_returns = self.calculate_returns(cleaned_stock_data)
        index_returns = self.calculate_returns(cleaned_index_data)

        # Print data shapes
        self.print_data_shapes(raw_stock_data, cleaned_stock_data, raw_index_data,
                               cleaned_index_data, raw_rf_data, cleaned_rf_data)

        # Save cleaned data
        self.save_cleaned_data(cleaned_stock_data, stock_returns, cleaned_index_data,
                               index_returns, cleaned_rf_data)


def main():
    """
    Main function to execute the data cleaning process.
    """

    raw_data_path = '/Users/macbook/Desktop/Farrer_Quant_Assignment/data/raw'
    cleaned_data_path = '/Users/macbook/Desktop/Farrer_Quant_Assignment/data/cleaned'

    cleaner = DataCleaner(raw_data_path, cleaned_data_path)
    cleaner.run_cleaning_process()


if __name__ == '__main__':
    main()
