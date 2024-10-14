import pandas as pd
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MomentumCalculator:
    """
    This class calculates momentum and lagged momentum for different horizons using cleaned stock data.
    """

    def __init__(self, input_path, output_path):
        """
        Initializes the MomentumCalculator with paths to input and output data.

        :param input_path: Path to the directory containing cleaned data files.
        :param output_path: Path to the directory where calculation results will be saved.
        """
        self.input_path = input_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.data = None
        self.momentum_horizons = [1, 3, 6, 12]  # in months
        self.lagged_horizons = [1, 2]  # in months

    def load_data(self):
        """
        Loads the cleaned stock data from a CSV file.
        """

        file_path = os.path.join(self.input_path, 'cleaned_stock_data.csv')
        self.data = pd.read_csv(file_path)
        self.data = self.data.sort_values(['ticker', 'date'])

    def calculate_momentum(self):
        """
        Calculates momentum for different horizons.
        M_n = P_t - P_(t-n)
        """

        for horizon in self.momentum_horizons:
            lag = horizon * 22  # 22 trading days per month
            column_name = f'momentum_{horizon}m'
            self.data[column_name] = self.data.groupby('ticker')['adjClose'].transform(
                lambda x: x - x.shift(lag)
            )
            logging.info(f"Calculated {column_name}")

    def calculate_lagged_momentum(self):
        """
        Calculates lagged momentum for different horizons.
        LM_(1,t) = P_(t-lag) - P_(t-lag-1)
        """

        for lag in self.lagged_horizons:
            lag_days = lag * 22
            column_name = f'lagged_momentum_1m_{lag}m'
            self.data[column_name] = self.data.groupby('ticker')['adjClose'].transform(
                lambda x: x.shift(lag_days) - x.shift(lag_days + 22)
            )
            logging.info(f"Calculated {column_name}")

    def save_results(self):
        """
        Saves the calculation results to a CSV file.
        """

        output_file = os.path.join(self.output_path, 'momentum_calculations.csv')
        self.data.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")

    def print_data_summary(self):
        """
        Prints a summary of the calculated data.
        """

        logging.info(f"Final data shape: {self.data.shape}")
        logging.info(f"Columns: {', '.join(self.data.columns)}")
        logging.info(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        logging.info(f"Number of unique tickers: {self.data['ticker'].nunique()}")

    def run_calculation_process(self):
        """
        Runs the entire momentum calculation process: loading data, calculating momentum,
        calculating lagged momentum, printing data summary, and saving the results.
        """

        self.load_data()
        self.calculate_momentum()
        self.calculate_lagged_momentum()
        self.print_data_summary()
        self.save_results()


def main():
    """
    Main function to execute the momentum calculation process.
    """

    input_path = '/Users/macbook/Desktop/Farrer_Quant_Assignment/data/cleaned'
    output_path = '/Users/macbook/Desktop/Farrer_Quant_Assignment/data/calculations'

    calculator = MomentumCalculator(input_path, output_path)
    calculator.run_calculation_process()


if __name__ == '__main__':
    main()
