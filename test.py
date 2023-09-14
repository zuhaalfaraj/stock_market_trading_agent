import torch
import pandas as pd
import matplotlib.pyplot as plt
import os

folder_path = 'path_to_your_directory'


class TradingTester:
    def __init__(self, env, agent, data, name, path=None):
        self.env = env
        self.agent = agent
        if path is not None:
            self.agent.load_model(path)
        self.data = data
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists('results'):
            os.makedirs('results')

        if not os.path.exists('results/' +self.name):
            os.makedirs('results/' +self.name)

    def test_agent(self):
        total_profit = 0
        self.env.reset()
        states_list = [
            self.env.cr.iloc[self.env.current_step],
            self.env.volume_oscillator.iloc[self.env.current_step],
            self.env.bollinger_percent.iloc[self.env.current_step],
            self.env.macd_signal.iloc[self.env.current_step],
            self.env.current_cash,
            self.env.stock_owned
        ]
        state = torch.Tensor(states_list).unsqueeze(0).to(self.device)
        done = False
        holding_history = pd.DataFrame(
            columns=['Date', 'Action', 'Holdings', 'Cash', 'Return', 'total_profit', 'Shares'])
        tracker = pd.DataFrame(
            columns=['Step', 'Buy_reward', 'Sell_reward', 'Hold_reward', 'Total Reward', 'Currnt Port', 'Next Port'])
        returns, action, total_profit = self.env.current_cash, None, 0

        while not done:
            holding_history.loc[len(holding_history), holding_history.columns] = {
                'Date': self.env.date[self.env.current_step],
                'Action': action,
                'Holdings': self.env.stock_owned,
                'Cash': self.env.current_cash,
                'Return': returns,
                'total_profit': total_profit,
                'Shares': self.env.shares
            }
            available_actions = [0, 1, 2]

            # Removing invalid actions
            if self.env.current_cash < self.env.stock_price.iloc[self.env.current_step] * self.env.shares:  # Can't buy
                available_actions.remove(0)
            if self.env.stock_owned < self.env.shares:  # Can't sell
                available_actions.remove(1)

            action = self.agent.act(state, available_actions)
            profit, done = self.env.step(action, tracker)
            total_profit += profit
            state = torch.Tensor(states_list).unsqueeze(0).to(self.device)
            returns = self.env.current_cash + self.env.stock_owned * self.env.stock_price.iloc[self.env.current_step]
        holding_history.to_csv('results/{}/holding_history.csv'.format(self.name))
        return total_profit, holding_history

    def plot_signals(self, holding_history):
        signals = holding_history[['Date', 'Action']].set_index('Date')
        plt.figure(figsize=(14, 8))
        plt.plot(self.data.index, self.data['Adj Close'], label='Price', color='blue')

        buy_dates = signals[signals['Action'] == 0].index
        buy_prices = self.data.loc[buy_dates, 'Adj Close']
        plt.scatter(buy_dates, buy_prices, marker='^', color='green', label='Buy Signal', alpha=1, s=50)

        sell_dates = signals[signals['Action'] == 1].index
        sell_prices = self.data.loc[sell_dates, 'Adj Close']
        plt.scatter(sell_dates, sell_prices, marker='v', color='red', label='Sell Signal', alpha=1, s=50)

        plt.title('Stock Price with Buy/Sell Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/{}/signals.png'.format(self.name))

    def backtest(self, holding_history):
        portfolio = holding_history[['Date', 'Return']].set_index('Date')
        benchmark = self.data[['Adj Close']].rename(columns={'Adj Close': 'Benchmark'}).pct_change().cumsum()
        plt.figure(figsize=(14, 8))
        plt.plot(portfolio.index, portfolio['Return'].pct_change().cumsum(), label='Portfolio')
        plt.plot(benchmark.index, benchmark['Benchmark'], label='Benchmark')
        plt.legend()
        plt.title('Backtest: Portfolio vs. Benchmark')
        plt.grid(True)
        plt.savefig('results/{}/backtest.png'.format(self.name))
