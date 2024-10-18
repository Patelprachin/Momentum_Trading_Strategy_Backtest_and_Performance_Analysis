import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


class MomentumVisualizer:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.strategies = {}
        self.sp500_returns = None

    def load_data(self):
        # Load S&P 500 returns
        self.sp500_returns = pd.read_csv(
            os.path.join(self.data_dir, 'cleaned', 'index_returns.csv'),
            parse_dates=['date'], index_col='date'
        )
        self.sp500_returns = self.sp500_returns.reindex(
            pd.date_range(self.sp500_returns.index.min(), self.sp500_returns.index.max(), freq='D')
        ).ffill()

        # Load strategy data
        strategy_files = {
            'momentum_1m': 'momentum_horizon/momentum_1m_Weekly_zscore1_smoothingFalse_5d.csv',
            'momentum_3m': 'momentum_horizon/momentum_3m_Weekly_zscore1_smoothingFalse_5d.csv',
            'momentum_6m': 'momentum_horizon/momentum_6m_Weekly_zscore1_smoothingFalse_5d.csv',
            'momentum_12m': 'momentum_horizon/momentum_12m_Weekly_zscore1_smoothingFalse_5d.csv',
            'rebalance_monthly': 'rebalance_frequency/momentum_1m_Monthly_zscore1_smoothingFalse_5d.csv',
            'rebalance_weekly': 'rebalance_frequency/momentum_1m_Weekly_zscore1_smoothingFalse_5d.csv',
            'smoothing_5d': 'smoothing_window/momentum_1m_Weekly_zscore1_smoothingTrue_5d.csv',
            'smoothing_22d': 'smoothing_window/momentum_1m_Weekly_zscore1_smoothingTrue_22d.csv',
            'smoothing_132d': 'smoothing_window/momentum_1m_Weekly_zscore1_smoothingTrue_132d.csv',
            'lagged_momentum': 'lagged_momentum/lagged_momentum_Weekly_zscore1_smoothingFalse_5d.csv'
        }

        for strategy, file_path in strategy_files.items():
            df = pd.read_csv(
                os.path.join(self.data_dir, 'backtesting', file_path),
                parse_dates=['date'], index_col='date'
            )

            # Ensure continuous date index and forward fill missing values
            df = df.reindex(
                pd.date_range(df.index.min(), df.index.max(), freq='D')
            ).ffill().fillna(0)

            # Calculate cumulative returns
            df['cumulative_return'] = (1 + df['return']).cumprod()

            self.strategies[strategy] = df

    def plot_cumulative_returns_momentum_horizons(self):
        plt.figure(figsize=(12, 6))
        for strategy in ['momentum_1m', 'momentum_3m', 'momentum_6m', 'momentum_12m']:
            plt.plot(self.strategies[strategy].index, self.strategies[strategy]['cumulative_return'], label=strategy)
        plt.title('Cumulative Returns for Different Momentum Horizons')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'cumulative_returns_momentum_horizons.png'))
        plt.close()

    def plot_cumulative_returns_rebalancing(self):
        plt.figure(figsize=(12, 6))
        for strategy in ['rebalance_monthly', 'rebalance_weekly']:
            plt.plot(self.strategies[strategy].index, self.strategies[strategy]['cumulative_return'], label=strategy)
        plt.title('Cumulative Returns for Different Rebalancing Frequencies')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'cumulative_returns_rebalancing.png'))
        plt.close()

    def plot_cumulative_returns_smoothing(self):
        plt.figure(figsize=(12, 6))
        for strategy in ['smoothing_5d', 'smoothing_22d', 'smoothing_132d']:
            plt.plot(self.strategies[strategy].index, self.strategies[strategy]['cumulative_return'], label=strategy)
        plt.title('Cumulative Returns for Different Smoothing Windows')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'cumulative_returns_smoothing.png'))
        plt.close()

    def plot_cumulative_returns_with_without_smoothing(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.strategies['momentum_1m'].index, self.strategies['momentum_1m']['cumulative_return'],
                 label='Without Smoothing')
        plt.plot(self.strategies['smoothing_5d'].index, self.strategies['smoothing_5d']['cumulative_return'],
                 label='With Smoothing (5d)')
        plt.title('Cumulative Returns With and Without Smoothing')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'cumulative_returns_with_without_smoothing.png'))
        plt.close()

    def plot_turnover_analysis(self):
        plt.figure(figsize=(12, 6))
        for strategy in ['rebalance_monthly', 'rebalance_weekly']:
            plt.plot(self.strategies[strategy].index, self.strategies[strategy]['turnover'], label=strategy)
        plt.title('Turnover Analysis for Different Rebalancing Frequencies')
        plt.xlabel('Date')
        plt.ylabel('Turnover')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'turnover_analysis.png'))
        plt.close()

    def plot_strategy_vs_sp500(self):
        strategy_data = self.strategies['smoothing_132d']

        # Align S&P 500 returns with strategy data
        aligned_sp500 = self.sp500_returns.loc[strategy_data.index[0]:strategy_data.index[-1]].copy()

        # Calculate cumulative returns for S&P 500 starting from the strategy start date
        aligned_sp500.loc[:, 'cumulative_return'] = (1 + aligned_sp500['returns']).cumprod()

        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)

        # Cumulative Returns
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(strategy_data.index, strategy_data['cumulative_return'], label='Strategy')
        ax1.plot(aligned_sp500.index, aligned_sp500['cumulative_return'], label='S&P 500')
        ax1.set_title('Cumulative Returns')
        ax1.legend()
        ax1.set_xlabel('')

        # Drawdown
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(strategy_data.index, strategy_data['drawdown'])
        ax2.set_title('Drawdown')
        ax2.set_xlabel('')

        # Turnover
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(strategy_data.index, strategy_data['turnover'])
        ax3.set_title('Turnover')
        ax3.set_xlabel('Date')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'strategy_vs_sp500.png'))
        plt.close()

    def plot_current_vs_lagged_momentum(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.strategies['momentum_1m'].index, self.strategies['momentum_1m']['cumulative_return'],
                 label='Current Momentum')
        plt.plot(self.strategies['lagged_momentum'].index, self.strategies['lagged_momentum']['cumulative_return'],
                 label='Lagged Momentum')
        plt.title('Current vs Lagged Momentum Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'current_vs_lagged_momentum.png'))
        plt.close()


def main():
    data_dir = '/Users/macbook/Desktop/Farrer_Quant_Assignment/data'
    output_dir = '/Users/macbook/Desktop/Farrer_Quant_Assignment/reports/figures'

    visualizer = MomentumVisualizer(data_dir, output_dir)
    visualizer.load_data()

    visualizer.plot_cumulative_returns_momentum_horizons()
    visualizer.plot_cumulative_returns_rebalancing()
    visualizer.plot_cumulative_returns_smoothing()
    visualizer.plot_cumulative_returns_with_without_smoothing()
    visualizer.plot_turnover_analysis()
    visualizer.plot_strategy_vs_sp500()
    visualizer.plot_current_vs_lagged_momentum()


if __name__ == "__main__":
    main()
