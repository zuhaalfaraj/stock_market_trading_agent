import argparse
from environment import TradingEnvironment
from train import TrainAgent
from data import DatasetClass, TechnicalIndicators
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

#python your_script_name.py --symbol "AAPL" --start_date "2017-01-01" --end_date "2022-01-01" --cash 10000 --episodes 500
def main(args):
    symbol = args.symbol
    start_date = args.start_date
    end_date = args.end_date
    cash = args.cash

    data_obj = DatasetClass(symbol, start_date, end_date)
    stocks_df = data_obj.stocks_df

    indicator = TechnicalIndicators()
    df_ta = indicator.run_all_default(stocks_df)
    #org = df_ta[['Adj Close','rsi', 'upper_band', 'lower_band']]
    #scaler = MinMaxScaler()
    #df_ta = pd.DataFrame(scaler.fit_transform(df_ta), columns=df_ta.columns, index=df_ta.index).drop(['Adj Close', 'rsi', 'upper_band', 'lower_band'], axis =1)
    #df_ta = pd.concat([df_ta, org], axis = 1)
    print(df_ta)
    env = TradingEnvironment(df_ta, cash)
    states_list = [env.cr[env.current_step], env.volume_oscillator[env.current_step],
                   env.bollinger_percent[env.current_step], env.macd_signal[env.current_step],
                   env.current_cash, env.stock_owned]

    trading_agent = TrainAgent(env, input_dim=len(states_list), action_dim=3, episodes=args.episodes)
    trading_agent.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a trading agent.")
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol to trade.')
    parser.add_argument('--start_date', type=str, default='2017-01-01', help='Start date for training data.')
    parser.add_argument('--end_date', type=str, default=datetime.today().strftime('%Y-%m-%d'),
                        help='End date for training data.')
    parser.add_argument('--cash', type=float, default=10000, help='Initial cash amount.')
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes.')

    args = parser.parse_args()
    main(args)
