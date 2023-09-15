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
    #print(df_ta)
    env = TradingEnvironment(df_ta, cash)


    trading_agent = TrainAgent(env, action_dim=3, episodes=args.episodes,
                               hidden_dim=args.hidden_dim, gamma=args.gamma,
                               epsilon=args.epsilon, epsilon_min=args.epsilon_min,
                               epsilon_decay=args.epsilon_decay, lr=args.lr)
    trading_agent.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a trading agent.")
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol to trade.')
    parser.add_argument('--start_date', type=str, default='2017-01-01', help='Start date for training data.')
    parser.add_argument('--end_date', type=str, default=datetime.today().strftime('%Y-%m-%d'),
                        help='End date for training data.')
    parser.add_argument('--cash', type=float, default=10000, help='Initial cash amount.')
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes.')

    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='NN hidden states dim')
    parser.add_argument('--gamma', type=float, default=0.99, help='Q Learning gamma (discount rate')
    parser.add_argument('--epsilon', type=float, default=0.9, help='exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.999, help='Decay rate')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='minimum epsilon')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate')

    args = parser.parse_args()
    main(args)
