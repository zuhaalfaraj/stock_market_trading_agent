import yfinance as yf
import matplotlib.pyplot as plt
import ta
from datetime import datetime
import pandas as pd

class DatasetClass:
    """A class for downloading and managing stock price data."""

    def __init__(self, symbol, start_date, end_date):
        """
        Initialize the DatasetClass.

        Args:
            symbol (str): The stock symbol.
            start_date (str): The start date for data retrieval.
            end_date (str): The end date for data retrieval.
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.stocks_df = self.query()

    def query(self):
        """
        Download stock price data.

        Returns:
            pandas.DataFrame: A DataFrame containing stock price data.
        """
        stock_data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        return stock_data

class TechnicalIndicators:
    """A class for calculating technical indicators on stock price data."""

    def run_all_default(self, stocks_df):
        """
        Calculate various technical indicators on stock price data.

        Args:
            stocks_df (pandas.DataFrame): A DataFrame containing stock price data.

        Returns:
            pandas.DataFrame: A DataFrame containing technical indicators.
        """
        df_ta = stocks_df[['Adj Close']].copy()
        df_ta['ma15'] = self.moving_avg(stocks_df)
        df_ta['cr'] = self.cumulative_return(stocks_df)
        df_ta['cmf'] = self.chaikin_money_flow(stocks_df)
        df_ta['volume_oscillator'] = self.volume_oscillator(stocks_df)
        upper_band, lower_band, bollinger_percent = self.bollingerband_percent(stocks_df)
        df_ta['upper_band'] = upper_band
        df_ta['lower_band'] = lower_band
        df_ta['bollinger_percent'] = bollinger_percent
        df_ta['stoch_oscillator'] = self.stoch_oscillator(stocks_df)
        df_ta['macd_signal'] = self.macd_signal(stocks_df)
        df_ta['rsi'] = self.rsi(stocks_df)
        return df_ta.dropna()

    def show(self, df_ta, ind):
        """
        Plot stock price and a technical indicator.

        Args:
            df_ta (pandas.DataFrame): A DataFrame containing stock price and technical indicators.
            ind (str): The technical indicator to plot.
        """
        fig, axs = plt.subplots(2)
        fig.suptitle(f'Close price vs {ind.upper()} Indicator')
        axs[0].plot(df_ta.index, df_ta['Adj Close'], label='Price')
        axs[0].legend()
        axs[0].grid()
        axs[1].plot(df_ta.index, df_ta[ind], label=ind, color='red')
        axs[1].legend()
        axs[1].grid()

    def moving_avg(self, stocks_df, window=15):
        """
        Calculate the Moving Average (MA) indicator.

        Args:
            stocks_df (pandas.DataFrame): A DataFrame containing stock price data.
            window (int): The window size for the moving average.

        Returns:
            pandas.Series: The Moving Average (MA) indicator.
        """
        ma = stocks_df['Adj Close'].rolling(window=window).mean()
        return ma

    def cumulative_return(self, stocks_df):
        """
        Calculate the Cumulative Return (CR) indicator.

        Args:
            stocks_df (pandas.DataFrame): A DataFrame containing stock price data.

        Returns:
            pandas.Series: The Cumulative Return (CR) indicator.
        """
        cr = 1 - stocks_df['Adj Close'] / stocks_df['Adj Close'].iloc[0]
        return cr

    def chaikin_money_flow(self, stocks_df):
        """
        Calculate the Chaikin Money Flow (CMF) indicator.

        Args:
            stocks_df (pandas.DataFrame): A DataFrame containing stock price data.

        Returns:
            pandas.Series: The Chaikin Money Flow (CMF) indicator.
        """
        cmf = ta.volume.ChaikinMoneyFlowIndicator(
            high=stocks_df['High'],
            low=stocks_df['Low'],
            close=stocks_df['Adj Close'],
            volume=stocks_df['Volume']
        ).chaikin_money_flow()
        return cmf

    def volume_oscillator(self, stocks_df):
        """
        Calculate the Volume Oscillator indicator.

        Args:
            stocks_df (pandas.DataFrame): A DataFrame containing stock price data.

        Returns:
            pandas.Series: The Volume Oscillator indicator.
        """
        short_volume_ma = stocks_df['Volume'].rolling(window=5).mean()
        long_volume_ma = stocks_df['Volume'].rolling(window=10).mean()
        volume_oscillator = ((short_volume_ma - long_volume_ma) / long_volume_ma) * 100
        return volume_oscillator

    def bollingerband_percent(self, stocks_df):
        """
        Calculate the Bollinger Bands (BB) and Bollinger Bands Percent (BBP) indicators.

        Args:
            stocks_df (pandas.DataFrame): A DataFrame containing stock price data.

        Returns:
            pandas.Series: The Upper Bollinger Band, Lower Bollinger Band, and Bollinger Bands Percent indicators.
        """
        bollinger = ta.volatility.BollingerBands(close=stocks_df['Close'], window=20, window_dev=2)
        upper_band = bollinger.bollinger_hband()
        lower_band = bollinger.bollinger_lband()
        bollinger_percent = (stocks_df['Adj Close'] - lower_band) / (upper_band - lower_band)
        return upper_band, lower_band, bollinger_percent

    def stoch_oscillator(self, stocks_df):
        """
        Calculate the Stochastic Oscillator indicator.

        Args:
            stocks_df (pandas.DataFrame): A DataFrame containing stock price data.

        Returns:
            pandas.Series: The Stochastic Oscillator indicator.
        """
        stoch_oscillator = (stocks_df['Adj Close'] - min(stocks_df['Low'])) / (max(stocks_df['High']) - min(stocks_df['Low']))
        stoch_oscillator = stoch_oscillator * 100
        return stoch_oscillator

    def macd_signal(self, stocks_df):
        """
        Calculate the Moving Average Convergence Divergence (MACD) and Signal Line indicators.

        Args:
            stocks_df (pandas.DataFrame): A DataFrame containing stock price data.

        Returns:
            pandas.Series: The MACD and Signal Line indicators.
        """
        exp1 = stocks_df['Adj Close'].ewm(span=12, adjust=False).mean()
        exp2 = stocks_df['Adj Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        return macd_signal

    def rsi(self, stocks_df):
        """
        Calculate the Relative Strength Index (RSI) indicator.

        Args:
            stocks_df (pandas.DataFrame): A DataFrame containing stock price data.

        Returns:
            pandas.Series: The RSI indicator.
        """
        delta = stocks_df['Adj Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

if __name__ == '__main__':
    symbol = 'TSLA'
    start_date = '2020-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    cash = 10000

    data_obj = DatasetClass(symbol, start_date, end_date)
    stocks_df = data_obj.stocks_df

    indicator = TechnicalIndicators()
    df_ta = indicator.run_all_default(stocks_df)
    indicator.show(df_ta, 'ma15')
