class TradingEnvironment:
    """
    A class representing a trading environment for reinforcement learning.

    Args:
        stock_df (pandas.DataFrame): DataFrame containing stock price and technical indicators data.
        starting_cash (float, optional): Initial cash available for trading. Defaults to 10000.
    """

    def __init__(self, stock_df, starting_cash=10000):
        self.current_cash = starting_cash
        self.stock_owned = 0
        self.stock_price = stock_df['Adj Close']
        self.cr = stock_df['cr']
        self.volume_oscillator = stock_df['volume_oscillator']
        self.bollinger_percent = stock_df['bollinger_percent']
        self.macd_signal = stock_df['macd_signal']

        self.bollinger_upper = stock_df['upper_band']
        self.bollinger_lower = stock_df['lower_band']
        self.rsi = stock_df['rsi']
        self.date = stock_df.index

        self.starting_cash = starting_cash

        self.current_step = 1
        self.shares = 0

        self.portfolio_value = starting_cash
        self.previous_price = self.stock_price.iloc[0]

        self.share_optimizer = DynamicShareCalculator()

    def reset(self):
        """Reset the trading environment to the initial state."""
        self.current_step = 0
        self.current_cash = self.starting_cash
        self.stock_owned = 0

    def step(self, action, tracker):
        """
        Take a trading action and compute the reward.

        Args:
            action (int): The action to take (0: Buy, 1: Sell, 2: Hold).
            tracker (pandas.DataFrame): DataFrame for tracking trading actions and rewards.

        Returns:
            float: The reward for the current step.
            bool: Whether the episode is done.
        """
        TRANSACTION_COST = 1.0
        HOLD_PENALTY = -0.01
        reward = 0

        current_portfolio_value = self.current_cash + self.stock_owned * self.stock_price.iloc[self.current_step - 1]
        potential_change = self.stock_price.iloc[self.current_step] - self.stock_price.iloc[self.current_step - 1]

        buy_reward, sell_reward, hold_reward = None, None, None

        if action == 0:  # Buy
            self.shares = self.share_optimizer.shares_to_buy(
                self.stock_price.iloc[self.current_step], self.current_cash,
                self.rsi.iloc[self.current_step], self.bollinger_upper.iloc[self.current_step],
                self.bollinger_lower.iloc[self.current_step]
            )
            if self.current_cash >= self.stock_price.iloc[self.current_step] * self.shares:
                self.stock_owned += self.shares
                self.current_cash -= self.stock_price.iloc[self.current_step] * self.shares - TRANSACTION_COST

                if potential_change > 0:
                    buy_reward = 5 + (5 * potential_change / self.stock_price.iloc[self.current_step - 1])
                else:
                    buy_reward = -2

                reward += buy_reward* 10

            else:
                reward = -10  # Penalty for invalid action

        elif action == 1:  # Sell
            self.shares = self.share_optimizer.shares_to_sell(
                self.stock_price.iloc[self.current_step], self.current_cash,
                self.rsi.iloc[self.current_step], self.bollinger_upper.iloc[self.current_step],
                self.bollinger_lower.iloc[self.current_step]
            )
            if self.stock_owned >= self.shares:
                self.stock_owned -= self.shares
                self.current_cash += self.stock_price.iloc[self.current_step] * self.shares - TRANSACTION_COST

                if potential_change < 0:
                    sell_reward = 5 - (5 * potential_change / self.stock_price.iloc[self.current_step - 1])
                else:
                    sell_reward = -2

                reward += sell_reward* 10

            else:
                reward = -10  # Penalty for invalid action

        elif action == 2:  # Hold
            hold_reward = (1 - abs(potential_change) / self.stock_price.iloc[self.current_step - 1])
            reward += hold_reward* 10

        next_portfolio_value = self.current_cash + self.stock_owned * self.stock_price.iloc[self.current_step]

        reward += 10 * (self.stock_price.iloc[self.current_step] - self.stock_price.iloc[self.current_step - 1]) / self.stock_price.iloc[self.current_step - 1]

        tracker.loc[len(tracker), tracker.columns] = {'Step': self.current_step, 'Buy_reward': buy_reward,
                                                      'Sell_reward': sell_reward,
                                                      'Hold_reward': hold_reward, 'Total Reward': reward,
                                                      'Currnt Port': current_portfolio_value,
                                                      'Next Port': next_portfolio_value}

        self.portfolio_value = next_portfolio_value
        self.current_step += 1
        done = (self.current_step == len(self.stock_price) - 1)

        return reward, done

class DynamicShareCalculator:
    """
    A class for dynamically calculating the number of shares to buy or sell based on trading signals.

    Args:
        buy_percentage (float): Percentage of available cash to use for buying. Defaults to 0.6.
        sell_percentage (float): Percentage of current holding to sell. Defaults to 0.2.
        rsi_overbought (int): RSI level considered overbought. Defaults to 70.
        rsi_oversold (int): RSI level considered oversold. Defaults to 30.
    """

    def __init__(self, buy_percentage=0.6, sell_percentage=0.2, rsi_overbought=70, rsi_oversold=30):
        self.buy_percentage = buy_percentage
        self.sell_percentage = sell_percentage
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def shares_to_buy(self, price, available_cash, rsi, bollinger_upper, bollinger_lower):
        """
        Calculate the number of shares to buy based on trading signals.

        Args:
            price (float): Current stock price.
            available_cash (float): Available cash for buying.
            rsi (float): Relative Strength Index (RSI) value.
            bollinger_upper (float): Upper Bollinger Band value.
            bollinger_lower (float): Lower Bollinger Band value.

        Returns:
            int: Number of shares to buy.
        """
        cash_to_use = available_cash * self.buy_percentage

        if rsi < self.rsi_oversold and price <= bollinger_lower:
            cash_to_use *= 1.5

        elif rsi > self.rsi_overbought or price >= bollinger_upper:
            cash_to_use *= 0.5

        number_of_shares = cash_to_use // price
        return int(number_of_shares)

    def shares_to_sell(self, current_holding, rsi, bollinger_upper, bollinger_lower, current_price):
        """
        Calculate the number of shares to sell based on trading signals.

        Args:
            current_holding (int): Current number of shares held.
            rsi (float): Relative Strength Index (RSI) value.
            bollinger_upper (float): Upper Bollinger Band value.
            bollinger_lower (float): Lower Bollinger Band value.
            current_price (float): Current stock price.

        Returns:
            int: Number of shares to sell.
        """
        shares_to_sell = current_holding * self.sell_percentage

        if rsi > self.rsi_overbought and current_price >= bollinger_upper:
            shares_to_sell *= 1.5

        elif rsi < self.rsi_oversold or current_price <= bollinger_lower:
            shares_to_sell *= 0.5

        return int(shares_to_sell)
