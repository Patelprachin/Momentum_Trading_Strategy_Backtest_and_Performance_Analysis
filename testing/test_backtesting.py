import os
from scripts.backtesting_sensitivity_analysis import MomentumBacktester


def test_momentum_horizons(data_path, output_dir, start_date, end_date):
    """
    Tests different momentum horizons for a momentum-based backtesting strategy.

    :param data_path: Path to the CSV file containing stock data.
    :param output_dir: Directory where backtest results will be saved.
    :param start_date: Start date for the backtest.
    :param end_date: End date for the backtest.
    """

    base_params = {
        'data_path': data_path,
        'start_date': start_date,
        'end_date': end_date,
        'rebalance_freq': 'W',
        'holding_periods': [22, 66, 132, 252],
        'z_score_threshold': 1,
        'smoothing': False,
        'smoothing_window': 5
    }

    momentum_horizons = ['momentum_1m', 'momentum_3m', 'momentum_6m', 'momentum_12m']
    test_results = []

    # Iterate over each momentum horizon and test the strategy.
    for momentum_column in momentum_horizons:
        current_params = base_params.copy()
        current_params['momentum_column'] = momentum_column

        backtester = MomentumBacktester(**current_params)
        backtester.run_backtest()
        metrics, _ = backtester.calculate_performance_metrics()

        # Save individual backtest results
        param_dir = os.path.join(output_dir, 'test_momentum_horizon')
        os.makedirs(param_dir, exist_ok=True)
        output_path = backtester.save_results(param_dir)
        print(f"Results for {momentum_column} saved to {output_path}")

        # Collect results for summary
        result = {
            'Momentum Horizon': momentum_column,
            **metrics
        }
        test_results.append(result)

    # Print summarized results
    print("\nTest Results Summary:")
    for result in test_results:
        print(f"\nMomentum Horizon: {result['Momentum Horizon']}")
        for metric, value in result.items():
            if metric != 'Momentum Horizon':
                print(f"{metric}: {value:.4f}")


def main():
    """
    Main function to execute the testing of momentum horizons.
    """

    data_path = "/Users/macbook/Desktop/Farrer_Quant_Assignment/data/calculations/momentum_calculations.csv"
    output_dir = "/Users/macbook/Desktop/Farrer_Quant_Assignment/data/backtesting/"
    start_date = "2019-10-09"
    end_date = "2024-10-09"

    test_momentum_horizons(data_path, output_dir, start_date, end_date)


if __name__ == "__main__":
    main()
