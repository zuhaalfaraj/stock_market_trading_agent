import argparse
from test import TradingTester
from environment import TradingEnvironment
from data import DatasetClass, TechnicalIndicators
from datetime import datetime
from trading_agent import DQAgent


def predict(args):
    symbol = args.symbol
    start_date = args.start_date
    end_date = args.end_date
    cash = args.cash

    data_obj = DatasetClass(symbol, start_date, end_date)
    stocks_df = data_obj.stocks_df

    indicator = TechnicalIndicators()
    df_ta = indicator.run_all_default(stocks_df)

    env = TradingEnvironment(df_ta, cash)
    states_list = env.get_state()

    agent = DQAgent(input_dim=len(states_list), action_dim=3)

    path = args.model_path
    tester = TradingTester(env, agent, df_ta, symbol, path)

    # Testing the agent
    total_profit, holding_history = tester.test_agent()

    # Plotting signals
    tester.plot_signals(holding_history)

    # Backtesting
    tester.backtest(holding_history)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a trading agent.")
    parser.add_argument('--symbol', type=str, default='VZ', help='Stock symbol to test.')
    parser.add_argument('--start_date', type=str, default='2023-03-01', help='Start date for testing data.')
    parser.add_argument('--end_date', type=str, default=datetime.today().strftime('%Y-%m-%d'), help='End date for testing data.')
    parser.add_argument('--cash', type=float, default=10000, help='Initial cash amount.')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help='Path to the trained model.')

    args = parser.parse_args()
    predict(args)
