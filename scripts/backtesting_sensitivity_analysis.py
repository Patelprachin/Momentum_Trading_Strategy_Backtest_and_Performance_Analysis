import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats


class MomentumBacktester:
    """
    This class handles the backtesting of a momentum strategy over a specified time period.
    """

    def __init__(self, data_path, start_date, end_date, momentum_column='momentum_1m',
                 rebalance_freq='W', holding_periods=[22, 66, 132, 252],
                 z_score_threshold=1, smoothing=False, smoothing_window=5):
        """
        Initializes the MomentumBacktester with data and strategy parameters.

        :param data_path: Path to the CSV file containing stock data.
        :param start_date: Start date for the backtest.
        :param end_date: End date for the backtest.
        :param momentum_column: Column name for momentum data.
        :param rebalance_freq: Frequency of rebalancing ('W' for weekly, 'M' for monthly).
        :param holding_periods: List of holding periods (in days) for calculating returns.
        :param z_score_threshold: Threshold for z-score filtering during stock selection.
        :param smoothing: Whether to apply smoothing to momentum values.
        :param smoothing_window: Window size for smoothing if smoothing is applied.
        """

        self.data = pd.read_csv(data_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date)]
        self.data.set_index(['date', 'ticker'], inplace=True)
        self.data.sort_index(inplace=True)

        self.momentum_column = momentum_column
        self.rebalance_freq = rebalance_freq
        self.holding_periods = holding_periods
        self.z_score_threshold = z_score_threshold
        self.smoothing = smoothing
        self.smoothing_window = smoothing_window
        self.portfolio = pd.DataFrame()
        self.results = {
            'daily_returns': [],
            'weights': [],
            'turnover': []
        }
        self.holding_period_returns = {period: [] for period in holding_periods}

        # Calculate the lookback period based on the momentum column
        self.lookback_period = self.get_lookback_period()

    def get_lookback_period(self):
        """
        Determines the lookback period based on the momentum column.
        :return: DateOffset object representing the lookback period.
        """

        if self.momentum_column == 'momentum_1m':
            return pd.DateOffset(months=1)
        elif self.momentum_column == 'momentum_3m':
            return pd.DateOffset(months=3)
        elif self.momentum_column == 'momentum_6m':
            return pd.DateOffset(months=6)
        elif self.momentum_column == 'momentum_12m':
            return pd.DateOffset(months=12)
        else:
            raise ValueError(f"Unsupported momentum column: {self.momentum_column}")

    def calculate_zscore(self, group):
        """
        Calculates the z-score for the selected momentum column.

        :param group: DataFrame group for which the z-score is calculated.
        :return: Series of z-score values for the momentum column.
        """

        if self.smoothing:
            # Apply rolling mean smoothing if specified
            group[self.momentum_column] = group[self.momentum_column].rolling(window=self.smoothing_window).mean()
        return stats.zscore(group[self.momentum_column], nan_policy='omit')

    def rebalance_portfolio(self, date):
        """
        Rebalances portfolio based on the selected momentum column and z-score.

        :param date: The date for which the portfolio is rebalanced.
        :return: Series representing the weights of selected stocks.
        """

        # Use the pre-calculated momentum values for the current date
        data_slice = self.data.loc[date].dropna(subset=[self.momentum_column])

        # Calculate z-scores for momentum values and filter based on threshold
        data_slice['zscore'] = self.calculate_zscore(data_slice)
        selected_stocks = data_slice[data_slice['zscore'] > self.z_score_threshold]

        # Calculate equal weights for selected stocks
        if len(selected_stocks) > 0:
            weights = pd.Series(1 / len(selected_stocks), index=selected_stocks.index)
        else:
            weights = pd.Series()

        return weights

    def calculate_returns(self, weights, current_date, next_date):
        """
        Calculates daily returns based on portfolio weights.

        :param weights: Series representing the portfolio weights.
        :param current_date: The current date for return calculation.
        :param next_date: The next date for return calculation.
        :return: Sum of weighted daily returns.
        """

        current_prices = self.data.loc[current_date, 'adjClose'].reindex(weights.index)
        next_prices = self.data.loc[next_date, 'adjClose'].reindex(weights.index)
        valid_stocks = current_prices.notnull() & next_prices.notnull()
        returns = (next_prices[valid_stocks] / current_prices[valid_stocks] - 1) * weights[valid_stocks]
        return returns.sum()

    @staticmethod
    def calculate_turnover(old_weights, new_weights):
        """
        Calculates portfolio turnover, which measures changes in portfolio composition.

        :param old_weights: Series of previous portfolio weights.
        :param new_weights: Series of new portfolio weights.
        :return: Portfolio turnover value.
        """

        return np.abs(old_weights.subtract(new_weights, fill_value=0)).sum() / 2

    def run_backtest(self):
        """
        Runs the backtest over the specified momentum column and rebalancing schedule.
        Calculates daily returns, turnover, and holding period returns.
        """

        dates = self.data.index.get_level_values('date').unique()

        # Determine the first valid date with momentum data
        first_valid_date = self.data[self.momentum_column].first_valid_index()[0]
        adjusted_start_date = max(dates[0], first_valid_date)

        # Generate rebalance dates
        rebalance_dates = pd.date_range(start=adjusted_start_date, end=dates[-1], freq=self.rebalance_freq)

        current_weights = pd.Series()
        portfolio_value = 1.0
        self.portfolio_values = [(adjusted_start_date, portfolio_value)]

        for i, date in enumerate(rebalance_dates[:-1]):
            if date not in dates:
                date = dates[dates > date][0]

            new_weights = self.rebalance_portfolio(date)
            turnover = self.calculate_turnover(current_weights, new_weights)
            self.results['weights'].append((date, new_weights))
            self.results['turnover'].append((date, turnover))

            next_rebalance = rebalance_dates[i + 1]
            holding_dates = pd.date_range(start=date, end=next_rebalance, freq='D')

            for j, holding_date in enumerate(holding_dates[:-1]):
                if holding_date in dates and holding_dates[j + 1] in dates:
                    daily_return = self.calculate_returns(new_weights, holding_date, holding_dates[j + 1])
                    portfolio_value *= (1 + daily_return)
                    self.results['daily_returns'].append((holding_dates[j + 1], daily_return))
                    self.portfolio_values.append((holding_dates[j + 1], portfolio_value))

            # Calculate holding period returns
            for period in self.holding_periods:
                end_date = date + pd.Timedelta(days=period)
                if end_date <= dates[-1]:
                    holding_return = self.calculate_holding_period_return(date, end_date, new_weights)
                    self.holding_period_returns[period].append((date, holding_return))

            current_weights = new_weights

    def calculate_holding_period_return(self, start_date, end_date, weights):
        """
        Calculates holding period returns for a given start and end date.

        :param start_date: Start date of the holding period.
        :param end_date: End date of the holding period.
        :param weights: Series of portfolio weights at the start date.
        :return: Sum of weighted returns for the holding period.
        """

        # Adjust start and end dates to the nearest valid trading dates
        if start_date not in self.data.index.get_level_values('date'):
            start_date = self.data.index.get_level_values('date').searchsorted(start_date, side='left')
            start_date = self.data.index.get_level_values('date')[start_date]

        if end_date not in self.data.index.get_level_values('date'):
            end_date = self.data.index.get_level_values('date').searchsorted(end_date, side='left')
            if end_date >= len(self.data.index.get_level_values('date')):
                return 0  # Return 0 if end_date exceeds available data
            end_date = self.data.index.get_level_values('date')[end_date]

        start_prices = self.data.loc[start_date, 'adjClose'].reindex(weights.index)
        end_prices = self.data.loc[end_date, 'adjClose'].reindex(weights.index)

        valid_stocks = start_prices.notnull() & end_prices.notnull()
        returns = (end_prices[valid_stocks] / start_prices[valid_stocks] - 1) * weights[valid_stocks]

        return returns.sum()

    def calculate_drawdown(self):
        """
        Calculates the maximum drawdown during the backtest.

        :return: DataFrame containing drawdown values for each date.
        """

        portfolio_values_df = pd.DataFrame(self.portfolio_values, columns=['date', 'value']).set_index('date')
        portfolio_values_df['peak'] = portfolio_values_df['value'].cummax()
        portfolio_values_df['drawdown'] = (portfolio_values_df['value'] / portfolio_values_df['peak']) - 1
        return portfolio_values_df

    def calculate_performance_metrics(self):
        """
        Calculates key performance metrics such as Sharpe Ratio, Drawdown, and returns.

        :return: Tuple of performance metrics dictionary and drawdown DataFrame.
        """

        returns_df = pd.DataFrame(self.results['daily_returns'], columns=['date', 'return']).set_index('date')
        cumulative_returns = (1 + returns_df['return']).cumprod() - 1
        annual_return = (1 + cumulative_returns.iloc[-1]) ** (252 / len(returns_df)) - 1
        annual_volatility = returns_df['return'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility

        drawdown_df = self.calculate_drawdown()
        max_drawdown = drawdown_df['drawdown'].min()

        turnover_df = pd.DataFrame(self.results['turnover'], columns=['date', 'turnover']).set_index('date')
        avg_turnover = turnover_df['turnover'].mean()

        metrics = {
            'Cumulative Returns': cumulative_returns.iloc[-1] * 100,
            'Annualized Return': annual_return * 100,
            'Annualized Volatility': annual_volatility * 100,
            'Annualized Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown * 100,
            'Average Turnover': avg_turnover * 100
        }

        # Calculate holding period performance metrics
        for period, returns in self.holding_period_returns.items():
            returns_df = pd.DataFrame(returns, columns=['date', 'return']).set_index('date')
            avg_return = returns_df['return'].mean()
            metrics[f'{period}-day Holding Period Avg Return'] = avg_return * 100

        return metrics, drawdown_df

    def calculate_cumulative_returns(self):
        """
        Calculates cumulative returns for the strategy.

        :return: Series of cumulative returns
        """
        returns_df = pd.DataFrame(self.results['daily_returns'], columns=['date', 'return']).set_index('date')
        return (1 + returns_df['return']).cumprod() - 1

    def print_strategy_setup(self):
        """
        Prints the setup details of the momentum strategy.
        """

        print("\n*Strategy Set Up*")
        print(f"Momentum Horizon: {self.momentum_column}")
        print(f"Rebalancing Frequency: {'Weekly' if self.rebalance_freq == 'W' else 'Monthly'}")
        print(f"Z-score Threshold: {self.z_score_threshold}")
        print(f"Z-score Smoothing: {self.smoothing}")
        print(f"Smoothing Window: {self.smoothing_window} days")

    def generate_output_filename(self):
        """
        Generates a filename for saving the results based on the strategy parameters.

        :return: A string representing the filename for the results.
        """

        rebalance_freq = 'Weekly' if self.rebalance_freq == 'W' else 'Monthly'
        smoothing_str = 'True' if self.smoothing else 'False'

        filename = (
            f"{self.momentum_column}_"
            f"{rebalance_freq}_"
            f"zscore{self.z_score_threshold}_"
            f"smoothing{smoothing_str}_"
            f"{self.smoothing_window}d.csv"
        )

        return filename

    def save_results(self, output_dir):
        """
        Saves the backtest results to a CSV file.

        :param output_dir: Directory path where the results will be saved.
        :return: Path of the saved CSV file.
        """

        returns_df = pd.DataFrame(self.results['daily_returns'], columns=['date', 'return']).set_index('date')
        weights_df = pd.DataFrame([(date, dict(weights)) for date, weights in self.results['weights']])
        weights_df.columns = ['date', 'weights']
        weights_df = weights_df.set_index('date').apply(pd.Series)
        turnover_df = pd.DataFrame(self.results['turnover'], columns=['date', 'turnover']).set_index('date')
        drawdown_df = self.calculate_drawdown()

        results_df = pd.concat([returns_df, weights_df, turnover_df, drawdown_df], axis=1)

        # Add holding period returns to the results
        for period, returns in self.holding_period_returns.items():
            returns_df = pd.DataFrame(returns, columns=['date', f'{period}d_return']).set_index('date')
            results_df = results_df.join(returns_df, how='left')

        filename = self.generate_output_filename()
        output_path = os.path.join(output_dir, filename)
        results_df.to_csv(output_path)
        return output_path

    def run_and_print_results(self):
        """
        Executes the backtest, prints the performance metrics, and displays the strategy setup.
        """

        self.run_backtest()
        self.print_strategy_setup()
        metrics, _ = self.calculate_performance_metrics()

        print("\n*Performance Metrics*")
        for metric, value in metrics.items():
            if metric == 'Annualized Sharpe Ratio':
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value:.2f}%")


def run_sensitivity_analysis(data_path, output_dir, start_date, end_date):
    """
    Conducts a sensitivity analysis for different strategy parameters.

    :param data_path: Path to the cleaned stock data CSV.
    :param output_dir: Directory where the backtest results will be saved.
    :param start_date: Start date for the sensitivity analysis.
    :param end_date: End date for the sensitivity analysis.
    """

    base_params = {
        'data_path': data_path,
        'start_date': start_date,
        'end_date': end_date,
        'momentum_column': 'momentum_1m',
        'rebalance_freq': 'W',
        'holding_periods': [22, 66, 132, 252],
        'z_score_threshold': 1,
        'smoothing': False,
        'smoothing_window': 5
    }

    sensitivity_params = {
        'momentum_horizon': {
            'param': 'momentum_column',
            'values': ['momentum_1m', 'momentum_3m', 'momentum_6m', 'momentum_12m'],
            'dir_name': 'momentum_horizon'
        },
        'rebalance_frequency': {
            'param': 'rebalance_freq',
            'values': ['W', 'M'],
            'dir_name': 'rebalance_frequency'
        },
        'z_score_threshold': {
            'param': 'z_score_threshold',
            'values': [0.5, 1, 2],
            'dir_name': 'z_score_threshold'
        },
        'smoothing': {
            'param': 'smoothing',
            'values': [False, True],
            'dir_name': 'smoothing'
        },
        'smoothing_window': {
            'param': 'smoothing_window',
            'values': [5, 22, 132],
            'dir_name': 'smoothing_window'
        }
    }

    all_results = []
    cumulative_returns = {}

    for param_name, param_info in sensitivity_params.items():
        param_dir = os.path.join(output_dir, param_info['dir_name'])
        os.makedirs(param_dir, exist_ok=True)

        param_cumulative_returns = {}

        for value in param_info['values']:
            current_params = base_params.copy()
            current_params[param_info['param']] = value

            if param_name == 'smoothing_window':
                current_params['smoothing'] = True

            print(f"\n{'=' * 50}")
            print(f"Running backtest for {param_name} = {value}")
            print(f"{'=' * 50}")

            backtester = MomentumBacktester(**current_params)
            backtester.run_backtest()
            backtester.print_strategy_setup()

            output_path = backtester.save_results(param_dir)
            print(f"\nResults saved to {output_path}")

            metrics, _ = backtester.calculate_performance_metrics()
            result = {
                'Parameter': param_name,
                'Value': value,
                'Output Path': output_path,
                **metrics
            }
            all_results.append(result)

            # Store cumulative returns for plotting
            param_cumulative_returns[value] = backtester.calculate_cumulative_returns()

        cumulative_returns[param_name] = param_cumulative_returns

    # Save summarized results
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(output_dir, 'summarised_results.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummarized results saved to {summary_path}")

    # Generate plots
    plot_cumulative_returns(cumulative_returns, output_dir)
    plot_turnover_analysis(summary_df, output_dir)


def plot_cumulative_returns(cumulative_returns, output_dir):
    """
    Plots cumulative returns for different strategy parameters.

    :param cumulative_returns: Dictionary of cumulative returns for each parameter
    :param output_dir: Directory to save the plots
    """
    plt.figure(figsize=(12, 8))
    for param_name, param_returns in cumulative_returns.items():
        if param_name in ['momentum_horizon', 'rebalance_frequency', 'smoothing_window', 'smoothing']:
            plt.figure(figsize=(12, 8))
            for value, returns in param_returns.items():
                plt.plot(returns.index, returns.values, label=f"{value}")

            plt.title(f"Cumulative Returns - {param_name}")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Returns")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"cumulative_returns_{param_name}.png"))
            plt.close()


def plot_turnover_analysis(summary_df, output_dir):
    """
    Plots turnover analysis for different rebalancing frequencies.

    :param summary_df: DataFrame containing summary results
    :param output_dir: Directory to save the plot
    """
    rebalance_freq_data = summary_df[summary_df['Parameter'] == 'rebalance_frequency']

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Value', y='Average Turnover', data=rebalance_freq_data)
    plt.title("Turnover Analysis for Different Rebalancing Frequencies")
    plt.xlabel("Rebalancing Frequency")
    plt.ylabel("Average Turnover (%)")
    plt.savefig(os.path.join(output_dir, "turnover_analysis.png"))
    plt.close()


def main():
    """
    Main function to execute the sensitivity analysis.
    """
    data_path = "/Users/macbook/Desktop/Farrer_Quant_Assignment/data/calculations/momentum_calculations.csv"
    output_dir = "/Users/macbook/Desktop/Farrer_Quant_Assignment/data/backtesting/"
    start_date = "2019-10-09"
    end_date = "2024-10-09"

    run_sensitivity_analysis(data_path, output_dir, start_date, end_date)


if __name__ == "__main__":
    main()
