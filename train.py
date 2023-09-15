import numpy as np

from trading_agent import DQAgent
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt

class TrainAgent:
    """
    Trainer class for training a Deep Q-Network (DQN) agent in a trading environment.

    Args:
        env (TradingEnvironment): Trading environment for training.
        input_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the action space.
        episodes (int): Number of training episodes.

    Attributes:
        env (TradingEnvironment): Trading environment for training.
        agent (DQAgent): DQN agent for training.
        episodes (int): Number of training episodes.
        best_reward (float): Best reward achieved during training.
        device (str): Device used for training ('cuda' or 'cpu').
    """

    def __init__(self, env, action_dim, episodes,
                 hidden_dim=128, gamma=0.99, epsilon=0.9, epsilon_min=0.01,
                 epsilon_decay=0.999, lr=0.0000001
                 ):
        self.env = env
        self.agent = DQAgent(input_dim=len(env.get_state()), action_dim=action_dim,
                             hidden_dim=hidden_dim, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min,
                             epsilon_decay=epsilon_decay, lr=lr)
        self.episodes = episodes
        self.best_reward = -float('inf')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists('models'):
            os.makedirs('models')

    def train(self):
        """
        Train the DQN agent in the trading environment.
        """
        for e in range(self.episodes):
            self.env.reset()
            done = False
            total_trades = 0
            total_reward = 0
            print(e)
            action = None
            reward = 0

            holding_history = pd.DataFrame(columns=['Date', 'Holdings', 'Cash', 'Return', 'Action', 'Reward', 'Shares'])
            tracker = pd.DataFrame(
                columns=['Step', 'Buy_reward', 'Sell_reward', 'Hold_reward', 'Total Reward', 'Currnt Port',
                         'Next Port'])

            buy_sell = [0, 0]
            current_return = self.env.current_cash

            states_list = self.env.get_state()
            state = torch.Tensor(states_list).unsqueeze(0).to(self.device)

            while not done:

                holding_history.loc[len(holding_history), holding_history.columns] = {
                    'Date': self.env.date[self.env.current_step], 'Holdings': self.env.stock_owned, 'Cash': self.env.current_cash,
                    'Return': current_return, 'Action': action, 'Reward': reward, 'Shares': self.env.shares}

                available_actions = [0, 1, 2]

                # Removing invalid actions
                if self.env.current_cash < self.env.stock_price.iloc[self.env.current_step] * self.env.shares:  # Can't buy
                    available_actions.remove(0)
                if self.env.stock_owned <= self.env.shares:  # Can't sell
                    available_actions.remove(1)

                action = self.agent.act(state, available_actions)
                reward, done = self.env.step(action, tracker)

                total_reward += reward
                if action == 0 or action == 1:
                    total_trades += 1
                    if action == 0:
                        buy_sell[0] += 1
                    if action == 1:
                        buy_sell[1] += 1

                next_state = torch.Tensor(self.env.get_state()).unsqueeze(0).to(self.device)
                self.agent.remember(state, action, reward, next_state, done)

                state = next_state



                self.agent.replay()

            print({'total_trades': total_trades, 'buy': buy_sell[0], 'sell': buy_sell[1],
                   'total_reward': total_reward / len(self.env.stock_price)})
            self.agent.loss_history_avg.append(np.mean(self.agent.loss_history))
            self.agent.loss_history = []
            self.post_episode_actions(total_reward, e)

        print("Training done!")
        self.agent.save_model('last_models.pth')

        if not os.path.exists('results'):
            os.makedirs('results')

        plt.plot(self.agent.loss_history_avg)
        plt.xlabel('episodes')
        plt.ylabel('avg loss')
        plt.savefig('results/loss_hist.png', bbox_inches='tight')

    def post_episode_actions(self, total_reward, e):
        """
        Perform post-episode actions, such as saving the best model.

        Args:
            total_reward (float): Total reward achieved in the episode.
            e (int): Current episode number.
        """
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.agent.save_model()
            print('Model Updated')

        if e % 5 == 0:
            self.agent.update_target_net()
