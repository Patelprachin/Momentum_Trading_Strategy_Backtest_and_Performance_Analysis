import pandas as pd
import numpy as np
from scipy import stats
import os


class LaggedMomentumBacktester:
    def __init__(self, data_path, start_date, end_date,
                 momentum_columns=['lagged_momentum_1m_1m', 'lagged_momentum_1m_2m'],
                 rebalance_freq='W', holding_periods=[22, 66, 132, 252],
                 z_score_threshold=1, smoothing=False, smoothing_window=5):
        self.data = pd.read_csv(data_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data[(self.data['date'] >= start_date) & (self.data['date'] <= end_date)]
        self.data.set_index(['date', 'ticker'], inplace=True)
        self.data.sort_index(inplace=True)

        self.momentum_columns = momentum_columns
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

    def calculate_zscore(self, group):
        if self.smoothing:
            for col in self.momentum_columns:
                group[col] = group[col].rolling(window=self.smoothing_window).mean()
        return group[self.momentum_columns].apply(lambda x: stats.zscore(x, nan_policy='omit'))

    def rebalance_portfolio(self, date):
        data_slice = self.data.loc[date].dropna(subset=self.momentum_columns)
        data_slice['zscore'] = self.calculate_zscore(data_slice).mean(axis=1)
        selected_stocks = data_slice[data_slice['zscore'] > self.z_score_threshold]

        if len(selected_stocks) > 0:
            weights = pd.Series(1 / len(selected_stocks), index=selected_stocks.index)
        else:
            weights = pd.Series()

        return weights

    def calculate_returns(self, weights, current_date, next_date):
        current_prices = self.data.loc[current_date, 'adjClose'].reindex(weights.index)
        next_prices = self.data.loc[next_date, 'adjClose'].reindex(weights.index)
        valid_stocks = current_prices.notnull() & next_prices.notnull()
        returns = (next_prices[valid_stocks] / current_prices[valid_stocks] - 1) * weights[valid_stocks]
        return returns.sum()

    @staticmethod
    def calculate_turnover(old_weights, new_weights):
        return np.abs(old_weights.subtract(new_weights, fill_value=0)).sum() / 2

    def run_backtest(self):
        dates = self.data.index.get_level_values('date').unique()
        first_valid_date = self.data[self.momentum_columns[0]].first_valid_index()[0]
        adjusted_start_date = max(dates[0], first_valid_date)
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

            for period in self.holding_periods:
                end_date = date + pd.Timedelta(days=period)
                if end_date <= dates[-1]:
                    holding_return = self.calculate_holding_period_return(date, end_date, new_weights)
                    self.holding_period_returns[period].append((date, holding_return))

            current_weights = new_weights

    def calculate_holding_period_return(self, start_date, end_date, weights):
        if start_date not in self.data.index.get_level_values('date'):
            start_date = self.data.index.get_level_values('date').searchsorted(start_date, side='left')
            start_date = self.data.index.get_level_values('date')[start_date]

        if end_date not in self.data.index.get_level_values('date'):
            end_date = self.data.index.get_level_values('date').searchsorted(end_date, side='left')
            if end_date >= len(self.data.index.get_level_values('date')):
                return 0
            end_date = self.data.index.get_level_values('date')[end_date]

        start_prices = self.data.loc[start_date, 'adjClose'].reindex(weights.index)
        end_prices = self.data.loc[end_date, 'adjClose'].reindex(weights.index)

        valid_stocks = start_prices.notnull() & end_prices.notnull()
        returns = (end_prices[valid_stocks] / start_prices[valid_stocks] - 1) * weights[valid_stocks]

        return returns.sum()

    def calculate_drawdown(self):
        portfolio_values_df = pd.DataFrame(self.portfolio_values, columns=['date', 'value']).set_index('date')
        portfolio_values_df['peak'] = portfolio_values_df['value'].cummax()
        portfolio_values_df['drawdown'] = (portfolio_values_df['value'] / portfolio_values_df['peak']) - 1
        return portfolio_values_df

    def calculate_performance_metrics(self):
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

        for period, returns in self.holding_period_returns.items():
            returns_df = pd.DataFrame(returns, columns=['date', 'return']).set_index('date')
            avg_return = returns_df['return'].mean()
            metrics[f'{period}-day Holding Period Avg Return'] = avg_return * 100

        return metrics, drawdown_df

    def print_strategy_setup(self):
        print("\n*Strategy Set Up*")
        print(f"Lagged Momentum Columns: {', '.join(self.momentum_columns)}")
        print(f"Rebalancing Frequency: {'Weekly' if self.rebalance_freq == 'W' else 'Monthly'}")
        print(f"Z-score Threshold: {self.z_score_threshold}")
        print(f"Z-score Smoothing: {self.smoothing}")
        print(f"Smoothing Window: {self.smoothing_window} days")

    def generate_output_filename(self):
        rebalance_freq = 'Weekly' if self.rebalance_freq == 'W' else 'Monthly'
        smoothing_str = 'True' if self.smoothing else 'False'

        filename = (
            f"lagged_momentum_"
            f"{rebalance_freq}_"
            f"zscore{self.z_score_threshold}_"
            f"smoothing{smoothing_str}_"
            f"{self.smoothing_window}d.csv"
        )

        return filename

    def save_results(self, output_dir):
        returns_df = pd.DataFrame(self.results['daily_returns'], columns=['date', 'return']).set_index('date')
        weights_df = pd.DataFrame([(date, dict(weights)) for date, weights in self.results['weights']])
        weights_df.columns = ['date', 'weights']
        weights_df = weights_df.set_index('date').apply(pd.Series)
        turnover_df = pd.DataFrame(self.results['turnover'], columns=['date', 'turnover']).set_index('date')
        drawdown_df = self.calculate_drawdown()

        results_df = pd.concat([returns_df, weights_df, turnover_df, drawdown_df], axis=1)

        for period, returns in self.holding_period_returns.items():
            returns_df = pd.DataFrame(returns, columns=['date', f'{period}d_return']).set_index('date')
            results_df = results_df.join(returns_df, how='left')

        filename = self.generate_output_filename()
        output_path = os.path.join(output_dir, filename)
        results_df.to_csv(output_path)
        return output_path

    def run_and_print_results(self):
        self.run_backtest()
        self.print_strategy_setup()
        metrics, _ = self.calculate_performance_metrics()

        print("\n*Performance Metrics*")
        for metric, value in metrics.items():
            if metric == 'Annualized Sharpe Ratio':
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value:.2f}%")


def main():
    data_path = "/Users/macbook/Desktop/Farrer_Quant_Assignment/data/calculations/momentum_calculations.csv"
    output_dir = "/Users/macbook/Desktop/Farrer_Quant_Assignment/data/backtesting/lagged_momentum"
    start_date = "2019-10-09"
    end_date = "2024-10-09"

    backtester = LaggedMomentumBacktester(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        momentum_columns=['lagged_momentum_1m_1m', 'lagged_momentum_1m_2m'],
        rebalance_freq='W',
        holding_periods=[22, 66, 132, 252],
        z_score_threshold=1,
        smoothing=False,
        smoothing_window=5
    )

    backtester.run_and_print_results()
    output_path = backtester.save_results(output_dir)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
