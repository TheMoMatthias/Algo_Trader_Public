import os
import json
import pandas as pd
import numpy as np 
import kucoin
import sys
import requests
import hashlib
import base64
import hmac
import time
import datetime as dt
from datetime import datetime, timedelta
import san 
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import seaborn 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import platform
# from data_download_entire_history  import *
from io import StringIO
import os
import sys
import pandas_ta as ta
from scipy.signal import argrelextrema, find_peaks
from scipy.ndimage import gaussian_filter1d 
# from numba import njit

def get_running_environment():
    if 'microsoft-standard' in platform.uname().release:
        return 'wsl'
    elif platform.system() == 'Windows':
        return 'windows'
    else:
        return 'unknown'

# Detect environment
env = get_running_environment()

base_path = os.path.realpath(__file__)

if env == 'wsl':
    # crypto_bot_path = "/mnt/c/Users/mauri/Documents/Trading Bot/Python/AlgoTrader" 
    crypto_bot_path = os.path.dirname(os.path.dirname(__file__))
    crypto_bot_path_windows = "/mnt/c/Users/mauri/Documents/Trading Bot/Python/AlgoTrader"
    Python_path = os.path.dirname(crypto_bot_path_windows)
    Trading_bot_path = os.path.dirname(Python_path)
    Trading_path = os.path.join(Trading_bot_path, "Trading")
else:
    crypto_bot_path = os.path.dirname(os.path.dirname(base_path))
    Python_path = os.path.dirname(crypto_bot_path)
    Trading_bot_path = os.path.dirname(Python_path)
    Trading_path = os.path.join(Trading_bot_path, "Trading")

logging_path = os.path.join(Trading_bot_path, "Logging")
# crypto_bot_path = r"C:\Users\mauri\Documents\Trading Bot\Python\AlgoTrader"

trade_api_path = os.path.join(crypto_bot_path, "API Trader")
AWS_path = os.path.join(crypto_bot_path, "AWS")
GOOGLE_path = os.path.join(crypto_bot_path, "GOOGLE")
backtest_path = os.path.join(crypto_bot_path, "Backtesting")
config_path = os.path.join(crypto_bot_path, "Config")
data_loader = os.path.join(crypto_bot_path, "Data Loader")

# Strategy and Trading API Paths
strategy_path = os.path.join(crypto_bot_path, "Trading Strategies")
trade_api_path = os.path.join(crypto_bot_path, "API Trader")
backtest_path = os.path.join(crypto_bot_path, "Backtesting")
kucoin_api = os.path.join(crypto_bot_path, "Kucoin API")

# Data Paths
data_path_crypto = os.path.join(Trading_bot_path, "Data", "Cryptocurrencies")
histo_data_path = os.path.join(data_path_crypto, "Historical Data")
datasets_path = os.path.join(data_path_crypto, "Datasets")
transformer_path = os.path.join(datasets_path, "transformer")
csv_dataset_path = os.path.join(datasets_path, "crypto datasets", "csv")
hdf_dataset_path = os.path.join(datasets_path, "crypto datasets", "hdf5")
hist_data_download_path = os.path.join(crypto_bot_path, "Hist Data Download")
hist_data_download_kucoin = os.path.join(hist_data_download_path, "Kucoin")
san_api_data_path = os.path.join(hist_data_download_path, "SanApi Data")
main_data_files_path = os.path.join(san_api_data_path, "Main data files")

# S3 / G Drive paths
Trading_path_s3 = "Trading"
logging_path_s3 = "Logging"

# List of all paths to be added
paths_to_add = [ crypto_bot_path, trade_api_path, 
                backtest_path, Trading_path,
                config_path, logging_path, 
                data_path_crypto, datasets_path,
                main_data_files_path, san_api_data_path,
                hist_data_download_path, kucoin_api,
                csv_dataset_path, hdf_dataset_path,
                AWS_path, GOOGLE_path, histo_data_path,
                data_loader, hist_data_download_kucoin, strategy_path]

# Add paths to sys.path and verify
for path in paths_to_add:
    if path not in sys.path:
        sys.path.append(path)

import mo_utils as utils
from utilsAWS import S3Utility
from utilsGoogleDrive import GoogleDriveUtility

# def assign_values(df, timestamp, values):
#     """
#     Assigns values to the DataFrame `df` at `timestamp`.
#     Handles both single values and multiple values (tuples) dynamically.
#     """
#     if isinstance(values, tuple):
#         # If multiple values are returned, assign them to all available columns from left to right
#         columns = df.columns[:len(values)]  # Select as many columns as needed
#         for col, val in zip(columns, values):
#             df.loc[timestamp, col] = val
#     else:
#         # Single value, assign to the first available column
#         df.iloc[df.index.get_loc(timestamp), 0] = values


#################################################################################################################################################
#
#                         TA Indicator Creator Class
#
#################################################################################################################################################

class TAIndicatorCreator():
    def __init__(self, logger=None):
        self.logger = logger if logger else self.setup_default_logger()

        #config
        config_path = utils.find_config_path() 
        config = utils.read_config_file(os.path.join(config_path,"AlgoTrader_config.ini"))
        
        # Retrieve all assets and create mapping with SAN coin names 
        
        all_asset_file_path = f"{main_data_files_path}/all_assets.xlsx" 
        self.all_assets = pd.read_excel(os.path.join(main_data_files_path, "all_assets.xlsx"), header=0)
        self.ticker_to_slug_mapping = dict(zip(self.all_assets['ticker'], self.all_assets['slug']))
        
        self.logger.info(f"TAIndicatorCreator initialized")
        
    def setup_default_logger(self):
        logger.add("TAIndicatorCreator.log", rotation="1 MB")
        return logger
    
    def load_price_data(self, coin, fiat, interval, metric, ohlc="close"):
        if metric in ['price_usdt',"price_usd"]:
            file_path = os.path.join(data_path_crypto, "Historical data", metric, interval, f"{metric}_{interval}.csv")
            if not os.path.exists(file_path):
                self.logger.error(f"FILE NON EXISTING: The data is not available for {metric} under {file_path}. Please investigate")
                return None
            df = pd.read_csv(file_path, index_col=0)
        else:
            file_path = os.path.join(data_path_crypto, "Historical data", metric, interval, f"{metric}_{coin}_{fiat}_{interval}.csv")
            if not os.path.exists(file_path):
                self.logger.error(f"FILE NON EXISTING: The data is not available for {metric} under {file_path}. Please investigate")
                return None
            df = pd.read_csv(file_path, index_col=0)
            df[self.ticker_to_slug_mapping[coin]] = df[ohlc].copy()   #creating coin slug from sanpy for better adoption
            
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    
    
    #################################################################################################################################################
    #
    #                            Indicator Calculation Functions
    #
    #################################################################################################################################################

    #Volume:
    def calculate_volume(self,df):
        return df['volume']
    
    def calculate_rolling_volatility(self, df, window=14):
        """Calculates the rolling volatility of a given data series."""
        return (df["close"].pct_change().rolling(window=window).std().round(5) *100)
    
    #EMA
    def ema_calculation(self, price, last_ema, span):
        return self.calculate_ema_with_ema(price, last_ema, span)
    
    def calculate_ema(self, df, span, last_ema=None):
        """
        Calculate EMA of given data series with specified length.
        """
        ema = ta.ema(df["price"], length=span).round(2)
        return ema

    def calculate_ema_with_ema(self, price, last_ema, span):
        """
        Simple EMA calculation.
        """
        return (price * (2 / (span + 1))) + (last_ema * (1 - (2 / (span + 1))))
    
    #MA (standard moving average)
    def calculate_ma(self, df, span):
        """Calculates the moving average of a given data series."""
        ma = ta.sma(df['price'], length=span).round(2)
        return ma

    #weighted moving average
    def calculate_vwma(self, df, span):
        """Calculates the volume-weighted moving average of a given data series."""
        volume = df['volume']
        vwma = ta.vwma(df['price'], volume, length=span).round(2)
        return vwma

    #MACD
    def macd_calculation(self, price_series, short_ema_span, long_ema_span, signal_span):
        macd = ta.macd(price_series["close"], fast=short_ema_span, slow=long_ema_span, signal=signal_span).round(2)
        
        return macd
    
    #Bollinger
    def bollinger_calculation(self, price_series, window, num_std_dev):
        bollinger = ta.bbands(price_series["close"], length=window, std=num_std_dev).round(2)
        return bollinger
    
    def calculate_ao(self, df, fast=5, slow=34):
        ao = ta.ao(df['high'], df['low'], fast, slow).round(2)
        return ao
    
    def calculate_cti(self, df, length=None, offset=None):
        """Calculates Chande Trend Index."""
        cti = ta.cti(df['close'], length=length, offset=offset).round(2)
        return cti

    def calculate_dm(self, df, length=None, mamode=None, talib=None, drift=None, offset=None):
        """Calculates Directional Movement (DM)."""
        dm = ta.dm(df['high'], df['low'], length=length, mamode=mamode, drift=drift, offset=offset).round(2)
        return dm

    def calculate_fisher(self, df, length=None, signal=None, offset=None):
        """Calculates Fisher Transform."""
        fisher = ta.fisher(df['high'], df['low'], length=length, signal=signal, offset=offset).round(2)
        return fisher

    def calculate_willr(self, df, length=None, offset=None):
        """Calculates Williams %R."""
        willr = ta.willr(df['high'], df['low'], df['close'], length=length, offset=offset).round(2)
        return willr
    
    def calculate_chop(self, df, length=14, atr_length=1, ln=False, scalar=100, drift=1, offset=0):
        """Calculates Choppiness Index."""
        chop = ta.chop(df['high'], df['low'], df['close'], length=length, atr_length=atr_length, ln=ln, scalar=scalar, drift=drift, offset=offset).round(2)
        return chop

    def calculate_hurst(self, data, lag_max=20, lag_type="linear", window=200):
        """Calculates the Hurst exponent over rolling windows of the time series."""

        # Ensure we are working with a 1D Series
        if isinstance(data, pd.DataFrame):
            if 'price' in data.columns:
                data = data['price']
            else:
                raise ValueError("DataFrame must contain a 'price' column or pass a Series/1D array.")
        
        if isinstance(data, pd.Series):
            date_index = data.index
            data = data.values  # Convert to numpy array for efficiency

        hurst_values = []

        # Ensure we have enough data points for the given window
        if len(data) < window:
            raise ValueError("The data length is smaller than the rolling window size.")

        # Loop over the data with a rolling window approach
        for start in range(len(data) - window + 1):
            end = start + window
            window_data = data[start:end]

            # Calculate the Hurst exponent for this window
            tau = []
            if lag_type == "linear":
                lagvec = np.arange(2, lag_max)  # Linear lags (2, 3, 4, ..., lag_max-1)
            elif lag_type == "log":
                lagvec = np.logspace(np.log10(2), np.log10(lag_max), num=lag_max-2).astype(int)  # Logarithmic spaced lags
            else:
                raise ValueError("Invalid lag_type. Choose either 'linear' or 'log'.")

            # Calculate pairwise differences and their standard deviation
            for lag in lagvec:
                pp = np.subtract(window_data[lag:], window_data[:-lag])
                if np.std(pp) == 0:
                    tau.append(np.nan)
                else:
                    tau.append(np.sqrt(np.std(pp, ddof=0)))  # Use ddof=0 for population std dev

            # Remove NaN values from tau for polyfit to work
            valid_tau = [t for t in tau if not np.isnan(t)]
            valid_lagvec = [lag for lag, t in zip(lagvec, tau) if not np.isnan(t)]

            if len(valid_tau) < 2:
                hurst_values.append(np.nan)  # Not enough valid data points
            else:
                # Perform linear regression on the log-log plot of lag vs tau
                poly = np.polyfit(np.log(valid_lagvec), np.log(valid_tau), 1)
                hurst_values.append(poly[0])  # Append the Hurst exponent for this window

        extending_hurst_values = np.full(window - 1, np.nan).tolist() + hurst_values
        
        # Return a series of Hurst exponent values with proper indexing
        series = pd.Series(data=extending_hurst_values, index=date_index, name="hurst")
        
        return series   #range(window, len(data) + 1)

    # VWAP
    def calculate_vwap(self, df):
        """Calculates VWAP from scratch."""
        vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume']).round(2)
        return vwap

    def calculate_vwap_with_last_values(self, price, last_vwap, volume, last_volume):
        """Updates VWAP based on the last available value."""
        return (last_vwap * last_volume + price * volume) / (last_volume + volume)

    # Supertrend
    def calculate_supertrend(self, df, period=10, multiplier=3):
        """Calculates Supertrend from scratch."""
        supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=period, multiplier=multiplier).round(2)
        return supertrend


    # ATR
    def calculate_atr(self, df, period=14):
        """Calculates ATR using the TradingView formula, with Relative Moving Average (RMA) of the True Range."""
        atr = ta.atr(df['high'], df['low'], df['close'], length=period).round(2)
        return atr


    def calculate_adx(self, df, di_length=14):
        """Calculates ADX using the Investopedia methodology with Wilder's smoothing."""

        # Ensure all OHLC column names are lowercase
        adx_series = ta.adx(df['high'], df['low'], df['close'], length=di_length).round(2)
        return adx_series[f"ADX_{di_length}"]

    # RSI
    def calculate_rsi(self, df, period=14):
        """Calculates RSI from scratch."""
        rsi_df = ta.rsi(df['close'], length=period).round(2)
        return rsi_df
    

    # def calculate_kama(self, df, length=14, fast=2, slow=30, drift=1):
    #     """Calculates KAMA from scratch."""
    #     kama = ta.kama(df["close"], length=length, fast=fast, slow=slow, drift=drift)

    #     return kama
    

    @staticmethod
    def verify_series(series, min_length=1):
        # Simple check for series validity
        if not isinstance(series, (pd.Series, pd.DataFrame)):
            series = pd.Series(series)
        if len(series) < min_length:
            return None
        return series

    @staticmethod
    def get_drift(drift):
        return int(drift) if isinstance(drift, int) and drift != 0 else 1

    @staticmethod
    def get_offset(offset):
        return int(offset) if isinstance(offset, int) else 0

    @staticmethod
    def non_zero_range(high: pd.Series, low: pd.Series) -> pd.Series:
        """
        Returns the difference high - low, replacing any zero results with a small epsilon.
        Matches the logic from Pandas TA.
        """
        diff = high - low
        # Epsilon to avoid division by zero issues
        epsilon = 1e-14
        diff = diff.replace(0, epsilon)
        return diff

    def calculate_kama(self, df, length=14, fast=2, slow=30, drift=1, offset=0, fillna=None, fill_method=None):
        """
        Kaufman's Adaptive Moving Average (KAMA) custom implementation.
        Matches logic and results with the provided Pandas TA code.
        """
        # Validate input
        close = df["close"]
        # Validate parameters and series
        length = int(length) if length and length > 0 else 10
        fast = int(fast) if fast and fast > 0 else 2
        slow = int(slow) if slow and slow > 0 else 30
        drift = self.get_drift(drift)
        offset = self.get_offset(offset)

        close = self.verify_series(close, max(length, fast, slow))
        if close is None:
            return None

        # Define helper function for weighting
        def weight(l: int) -> float:
            return 2 / (l + 1)

        fr = weight(fast)
        sr = weight(slow)

        # Compute the same intermediate steps as Pandas TA:
        # abs_diff = |close - close.shift(length)|
        # peer_diff = |close - close.shift(drift)|
        abs_diff_s = self.non_zero_range(close, close.shift(length)).abs()
        peer_diff_s = self.non_zero_range(close, close.shift(drift)).abs()
        peer_diff_sum_s = peer_diff_s.rolling(length).sum()

        # ER = abs_diff / peer_diff_sum
        er_s = abs_diff_s / peer_diff_sum_s

        # x = er*(fr - sr) + sr
        x_s = er_s * (fr - sr) + sr
        sc_s = x_s * x_s

        # Convert to numpy arrays for the loop calculation
        close_values = close.values
        sc = sc_s.values

        n = close_values.size
        kama_values = np.full(n, np.nan)

        # Following original code logic:
        # result = [nan for _ in range(length-1)] + [0]
        # Then start recursive calculation at index = length
        if length - 1 < n:
            kama_values[length - 1] = 0.0

        # Calculate KAMA recursively starting from 'length'
        # From Pandas TA logic:
        # for i in range(length, m):
        #     result.append(sc[i]*close[i] + (1 - sc[i])*result[i - 1])
        for i in range(length, n):
            if not np.isnan(sc[i]) and not np.isnan(kama_values[i - 1]):
                kama_values[i] = sc[i] * close_values[i] + (1 - sc[i]) * kama_values[i - 1]

        kama_series = pd.Series(kama_values, index=close.index)

        # Offset
        if offset != 0:
            kama_series = kama_series.shift(offset)

        # Handle fills
        if fillna is not None:
            kama_series.fillna(fillna, inplace=True)
        if fill_method is not None:
            kama_series.fillna(method=fill_method, inplace=True)

        kama_series.name = f"KAMA_{length}_{fast}_{slow}"
        kama_series.category = "overlap"

        return kama_series.round(3)
    

#################################################################################################################################################
#
#                         Special trading indicator calculation functions
#
#################################################################################################################################################
    def calculate_fibonacci_retracement_levels(self, data, window=10, ema_short=3, ema_long=9):
        """
        Calculate Fibonacci retracement levels based on recent significant points and trend direction.
        In an uptrend, levels are projected upwards from the recent high. In a downtrend, levels are projected downwards.
        
        Parameters:
        - data (pd.DataFrame): The input data with 'high', 'low', and 'close' columns.
        - window (int): The lookback window to calculate rolling highs and lows.
        
        Returns:
        - pd.DataFrame: A DataFrame with columns for each Fibonacci level, trend, and key levels.
        """
        # Ensure there are enough rows to perform the calculations
        if len(data) < window:
            return pd.DataFrame(np.nan, index=data.index, columns=['Fib_261.8', 'Fib_200.0', 'Fib_161.8', 'Fib_138.2', 'Fib_100.0', 'Fib_78.6', 'Fib_61.8', 'Fib_50.0', 'Fib_38.2', 'Fib_23.6', 'Trend'])

        # Calculate recent high and low within the specified window
        recent_data = data.iloc[-window:].copy()
        recent_high = recent_data['high'].max()
        recent_low = recent_data['low'].min()

        # Determine trend based on the index of recent high and low
        recent_high_index = recent_data['high'].idxmax()
        recent_low_index = recent_data['low'].idxmin()
        
        # Calculate EMAs for additional trend confirmation
        data['EMA_short'] = data['close'].ewm(span=ema_short, adjust=False).mean()
        data['EMA_long'] = data['close'].ewm(span=ema_long, adjust=False).mean()

        # Confirm the trend using EMA cross along with high-low index check
        if recent_high_index > recent_low_index:
            trend = "uptrend"
        elif recent_high_index < recent_low_index:
            trend = "downtrend"
        

        # Calculate Fibonacci levels based on trend direction
        fib_levels_df = pd.DataFrame(index=recent_data.index)
        diff = recent_high - recent_low

        if trend == "uptrend":
            # Uptrend: project levels above the high
            fib_levels_df['Fib_100.0'] = recent_high
            fib_levels_df['Fib_78.6'] = recent_high - diff * 0.786
            fib_levels_df['Fib_61.8'] = recent_high - diff * 0.618
            fib_levels_df['Fib_50.0'] = recent_high - diff * 0.5
            fib_levels_df['Fib_38.2'] = recent_high - diff * 0.382
            fib_levels_df['Fib_23.6'] = recent_high - diff * 0.236
            fib_levels_df['Fib_0.0'] = recent_high

        elif trend == "downtrend":
            # Downtrend: project levels downwards from the high
            fib_levels_df['Fib_100.0'] = recent_high
            fib_levels_df['Fib_78.6'] = recent_low + diff * 0.786
            fib_levels_df['Fib_61.8'] = recent_low + diff * 0.618
            fib_levels_df['Fib_50.0'] = recent_low + diff * 0.5
            fib_levels_df['Fib_38.2'] = recent_low + diff * 0.382
            fib_levels_df['Fib_23.6'] = recent_low + diff * 0.236
            fib_levels_df['Fib_0.0'] = recent_low
        

        # Populate high, low, and trend information
        fib_levels_df['High'] = recent_high
        fib_levels_df['Low'] = recent_low
        fib_levels_df['Trend'] = trend

        # Return only the last `window` rows to focus on recent data
        return fib_levels_df.iloc[-window:]

    def calculate_pivot_points(self, data, window=20):
        df = pd.DataFrame(index=data.index)
        high = data['high'].rolling(window=window).max()
        low = data['low'].rolling(window=window).min()
        close = data['close']
        df['Pivot'] = (high + low + close) / 3
        df['R1'] = 2 * df['Pivot'] - low
        df['S1'] = 2 * df['Pivot'] - high
        df['R2'] = df['Pivot'] + (high - low)
        df['S2'] = df['Pivot'] - (high - low)
        df['R3'] = high + 2 * (df['Pivot'] - low)
        df['S3'] = low - 2 * (high - df['Pivot'])
        return df

    def calculate_fibonacci_fan(self, data, window=20):
        df = pd.DataFrame(index=data.index)
        high = data['high'].rolling(window=window).max()
        low = data['low'].rolling(window=window).min()
        df['Fib_38.2'] = high - (high - low) * 0.382
        df['Fib_50.0'] = high - (high - low) * 0.5
        df['Fib_61.8'] = high - (high - low) * 0.618
        return df

    def calculate_volume_profile(self, data, window=100, num_bins=20, lookback=200):
        """
        Optimized function to calculate an approximated volume profile over a rolling window for OHLCV data,
        limited to a specified lookback window for performance.

        Parameters:
        - data (pd.DataFrame): The input data with columns 'High', 'Low', and 'Volume'.
        - window (int): The lookback window to calculate the volume profile.
        - num_bins (int): The number of bins to divide the price range.
        - lookback (int): The number of recent rows to consider for the volume profile calculation.

        Returns:
        - pd.DataFrame: A DataFrame with the volume profile for each rolling window period,
                        with each bin as a separate column and volume distributed across intersecting bins.
        """
        # Limit data to the most recent `lookback` rows for faster calculations
        if lookback < len(data):
            recent_data = data.iloc[-lookback:].copy()
        else:
            recent_data = data.copy()

        # Initialize the DataFrame to store volume profiles with each bin as a separate column (float type)
        bin_columns = [f'Bin_{i+1}' for i in range(num_bins)]
        volume_profile_df = pd.DataFrame(np.nan, index=recent_data.index, columns=bin_columns)  # Initialized as NaN for uncalculated rows
        bin_edges_list = [np.nan] * len(recent_data)  # Placeholder for bin edges with NaN for uncalculated rows

        # Vectorized calculation of volume profiles for each rolling window
        for end in range(window, len(recent_data) + 1):
            # Define the rolling window data
            window_data = recent_data.iloc[end - window:end]

            # Determine the price range for this window and create bins
            price_min = window_data['low'].min()
            price_max = window_data['high'].max()
            bins = np.linspace(price_min, price_max, num_bins + 1)

            # Initialize volume profile for this window
            volume_profile = np.zeros(num_bins)

            # Vectorized bin assignment for high and low values within the window
            low_bin_indices = np.searchsorted(bins, window_data['low'].values, side='right') - 1
            high_bin_indices = np.searchsorted(bins, window_data['high'].values, side='left')

            # Vectorized volume distribution over bins
            for i in range(len(window_data)):
                volume = window_data['volume'].iloc[i]
                start_bin = low_bin_indices[i]
                end_bin = high_bin_indices[i]

                if start_bin <= end_bin:  # Proceed if the price range spans bins
                    num_bins_in_range = end_bin - start_bin + 1
                    volume_per_bin = volume / num_bins_in_range
                    volume_profile[start_bin:end_bin+1] += volume_per_bin

            # Store a deep copy of the calculated volume profile in the DataFrame
            volume_profile_df.loc[window_data.index[-1], bin_columns] = volume_profile.copy()
            bin_edges_list[end - 1] = bins[:-1].tolist()  # Store bin edges as list for the current row only

        # Assign the bin edges list to the DataFrame, with NaN for rows without calculated profiles
        volume_profile_df['BinEdges'] = bin_edges_list
        return volume_profile_df

    def identify_hvn_lvn(self, volume_profile_df, bin_edges_col='BinEdges', volume_cols_prefix='Bin'):
        """
        Identifies HVN (high volume nodes) and LVN (low volume nodes) by mapping bin names to actual price levels
        and returns price levels for HVN and LVN zones.
        
        Parameters:
        - volume_profile_df (pd.DataFrame): DataFrame with volume bins and bin edges.
        - bin_edges_col (str): Name of the column containing bin edges.
        - volume_cols_prefix (str): Prefix for volume columns (e.g., 'Bin' for columns named 'Bin_1', 'Bin_2', etc.)

        Returns:
        - pd.DataFrame: DataFrame with HVN_Zones and LVN_Zones columns as lists of price levels.
        """
        hvn_zones = []
        lvn_zones = []

        # Iterate over each row in the DataFrame to calculate HVN and LVN
        for idx, row in volume_profile_df.iterrows():
            bin_edges = row[bin_edges_col]
            
            # Check if bin_edges is a list with the correct number of elements
            if not isinstance(bin_edges, list) or len(bin_edges) < 2:
                # If bin_edges is not valid, append empty lists for HVN and LVN zones and skip to the next row
                hvn_zones.append([])
                lvn_zones.append([])
                continue
            
            # Get volumes for each bin and identify the HVN and LVN bins based on volume thresholds
            bin_volumes = [row.get(f"{volume_cols_prefix}_{i+1}", 0) for i in range(len(bin_edges) - 1)]
            
            # Find high volume and low volume zones
            sorted_bins = sorted(enumerate(bin_volumes), key=lambda x: x[1], reverse=True)
            hvn_indices = [idx for idx, vol in sorted_bins[:5]]  # Top 5 HVNs
            lvn_indices = [idx for idx, vol in sorted_bins[-5:]]  # Bottom 5 LVNs

            # Map HVN and LVN indices to price ranges
            hvn_zones_prices = [(bin_edges[i], bin_edges[i+1]) for i in hvn_indices]
            lvn_zones_prices = [(bin_edges[i], bin_edges[i+1]) for i in lvn_indices]

            hvn_zones.append(hvn_zones_prices)
            lvn_zones.append(lvn_zones_prices)
        
        # Add HVN_Zones and LVN_Zones columns as lists of price levels
        volume_profile_df['HVN_Zones'] = hvn_zones
        volume_profile_df['LVN_Zones'] = lvn_zones
        
        return volume_profile_df

    
    def find_local_extrema_filtered(self, df, window=1000, prominence=0.02, min_distance=3, window_size=5, save_file=False, interval=None, coin=None, fiat=None, indicator_name=None):
        """
        Identify local minima and maxima in a time series using scipy.signal.argrelextrema.
        
        Parameters:
        df (pd.DataFrame): A DataFrame with two columns, 'timestamp' and 'value'.
        
        Returns:
        dict: A dictionary containing the last 5 peaks and troughs with timestamps.
        """
        # Limit the data to the most recent 1000 rows for performance
        values = df[-window:].values
        timestamps = df[-window:].index.values
        
        # Step 1: Apply rolling window to identify local maxima and minima
        rolling_max = pd.Series(values).rolling(window=window_size, center=True).max()
        rolling_min = pd.Series(values).rolling(window=window_size, center=True).min()

        # Find indices of local maxima and minima
        peaks, _ = find_peaks(values, prominence=prominence, distance=min_distance)
        troughs, _ = find_peaks(-values, prominence=prominence, distance=min_distance)
        
        # Step 3: Filter peaks and troughs to ensure they are local extrema within the window
        filtered_peaks = [i for i in peaks if values[i] == rolling_max[i]]
        filtered_troughs = [i for i in troughs if values[i] == rolling_min[i]]
        
        # Create a DataFrame with the combined data
        df_extrema = pd.DataFrame({'timestamp': timestamps})
        
        # Add columns for peaks and troughs
        df_extrema['peak'] = None
        df_extrema['trough'] = None
        
        df_extrema.loc[filtered_peaks, 'peak'] = values[filtered_peaks]
        df_extrema.loc[filtered_troughs, 'trough'] = values[filtered_troughs]

        # Step 4: Return only the last 5 peaks and troughs
        last_5_peaks = [(timestamps[i], values[i]) for i in filtered_peaks[-5:]]
        last_5_troughs = [(timestamps[i], values[i]) for i in filtered_troughs[-5:]]
        
        local_maxima = last_5_peaks
        local_minima = last_5_troughs
        
        if save_file:
            data_name = f"extrema_{indicator_name}"
            
            if not os.path.exists(os.path.join(histo_data_path, data_name, interval)):
                os.makedirs(os.path.join(histo_data_path, data_name, interval))
            
            df_extrema.set_index('timestamp', inplace=True)
            df_extrema.to_csv(os.path.join(histo_data_path, data_name, interval, f"{data_name}_{coin}_{fiat}_{interval}.csv"))

        return local_minima, local_maxima
    

    def cdl_pattern(self, price_data, name="all", offset=0):
        """
        Custom Candle Pattern Detection.

        Parameters:
            open_ (pd.Series): Open prices.
            high (pd.Series): High prices.
            low (pd.Series): Low prices.
            close (pd.Series): Close prices.
            name (str or list): Name(s) of patterns to calculate. Defaults to "all".

        Returns:
            dict: Dictionary containing results of pattern matches for each requested pattern.
        """
        # Define available patterns
        available_patterns = {
            "morningstar": TAIndicatorCreator.is_morning_star,
            "morningstardoji": TAIndicatorCreator.is_morning_star_doji,
            "eveningstar": TAIndicatorCreator.is_evening_star,
            "eveningstardoji": TAIndicatorCreator.is_evening_star_doji,
            "3whitesoldiers": TAIndicatorCreator.is_three_white_soldiers,
            "3blackcrows": TAIndicatorCreator.is_three_black_crows,
            "engulfing_bullish": TAIndicatorCreator.is_engulfing_bullish,
            "engulfing_bearish": TAIndicatorCreator.is_engulfing_bearish,
            "flag_bullish": TAIndicatorCreator.is_flag_bullish,
            "flag_bearish": TAIndicatorCreator.is_flag_bearish,
            "inside_candle": TAIndicatorCreator.is_inside_candle,
            "special_engulfing_bullish": TAIndicatorCreator.is_special_engulfing_bullish,
            "special_engulfing_bearish": TAIndicatorCreator.is_special_engulfing_bearish,
            "retest_bullish": TAIndicatorCreator.is_retest_bullish,
            "retest_bearish": TAIndicatorCreator.is_retest_bearish,
        }
        
        if offset != 0:
            price_data = price_data.shift(offset)

        price_data.columns = price_data.columns.str.lower()
        open_ = price_data['open']
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']


        if name == "all":
            name = list(available_patterns.keys())
        elif isinstance(name, list):
            name = [pattern for pattern in name if pattern in available_patterns]

        # Verify valid patterns
        if not name:
            raise ValueError("No valid patterns specified.")

        # Evaluate patterns
        results = {}
        for pattern_name in name:
            pattern_func = available_patterns[pattern_name]
            results[pattern_name] = pattern_func(open_, high, low, close)

        return results

    def is_morning_star(open_, high, low, close):
        return (
            len(open_) > 2 and
            close.iloc[-3] < open_.iloc[-3] and
            abs(close.iloc[-2] - open_.iloc[-2]) < (abs(high.iloc[-2] - low.iloc[-2]) - abs(close.iloc[-2] - open_.iloc[-2])) and # Small second candle
            close.iloc[-1] > open_.iloc[-1] and close.iloc[-1] > (close.iloc[-3] + open_.iloc[-3]) / 2 and 
            open_.iloc[-3] > (close.iloc[-1] + open_.iloc[-1]) / 2
        )


    def is_morning_star_doji(open_, high, low, close):
        return (
            len(open_) > 2 and
            close.iloc[-3] < open_.iloc[-3] and
            (abs(close.iloc[-2] - open_.iloc[-2]) < (abs(high.iloc[-2] - low.iloc[-2]) - abs(close.iloc[-2] - open_.iloc[-2])) or 
            (abs(close.iloc[-2] - open_.iloc[-2]) < abs(close.iloc[-3] - open_.iloc[-3])*0.2)) and # Doji second candle
            close.iloc[-1] > open_.iloc[-1] and close.iloc[-1] > (close.iloc[-3] + open_.iloc[-3]) / 2  and #last candle close above half of first candle
            abs(close.iloc[-3] - open_.iloc[-3]) > 2*abs(close.iloc[-2] - open_.iloc[-2]) and
            open_.iloc[-3] > (close.iloc[-1] + open_.iloc[-1]) / 2
        )


    def is_evening_star(open_, high, low, close):
        return (
            len(open_) > 2 and
            close.iloc[-3] > open_.iloc[-3] and  # Large bullish first candle
            abs(close.iloc[-2] - open_.iloc[-2]) < (abs(high.iloc[-2] - low.iloc[-2]) - abs(close.iloc[-2] - open_.iloc[-2])) and # Small second candle
            close.iloc[-1] < open_.iloc[-1] and close.iloc[-1] < (close.iloc[-3] + open_.iloc[-3]) / 2 and # Large bearish third candle
            open_.iloc[-3] < (close.iloc[-1] + open_.iloc[-1]) / 2
        )


    def is_evening_star_doji(open_, high, low, close):
        return (
            len(open_) > 2 and
            close.iloc[-3] > open_.iloc[-3] and
            (abs(close.iloc[-2] - open_.iloc[-2]) < (abs(high.iloc[-2] - low.iloc[-2]) - abs(close.iloc[-2] - open_.iloc[-2])) or 
            (abs(close.iloc[-2] - open_.iloc[-2]) < abs(close.iloc[-3] - open_.iloc[-3])*0.2))  and  # Doji second candle
            close.iloc[-1] < open_.iloc[-1] and close.iloc[-1] < (close.iloc[-3] + open_.iloc[-3]) / 2 and 
            abs(close.iloc[-3] - open_.iloc[-3]) > 2*abs(close.iloc[-2] - open_.iloc[-2]) and 
            open_.iloc[-3] < (close.iloc[-1] + open_.iloc[-1]) / 2
        )


    def is_three_white_soldiers(open_, high, low, close):
        threshold = 0.1  # 10% threshold for small lower wicks
        return (
            len(open_) > 2 and
            all(close.iloc[i] > open_.iloc[i] for i in range(-3, 0)) and
            close.iloc[-3] < close.iloc[-2] < close.iloc[-1] and
            all(
                (open_.iloc[i] - low.iloc[i]) < (high.iloc[i] - low.iloc[i]) * threshold
                for i in range(-3, 0)
            )
        )


    def is_three_black_crows(open_, high, low, close):
        threshold = 0.1  # 10% threshold for small upper wicks
        return (
            len(open_) > 2 and
            all(close.iloc[i] < open_.iloc[i] for i in range(-3, 0)) and
            close.iloc[-3] > close.iloc[-2] > close.iloc[-1] and
            all(
                (high.iloc[i] - open_.iloc[i]) < (high.iloc[i] - low.iloc[i]) * threshold
                for i in range(-3, 0)
            )
        )


    def is_engulfing_bullish(open_, high, low, close):
        return (
            len(open_) > 1 and
            # Previous candle is bearish
            close.iloc[-2] < open_.iloc[-2] and
            # Current candle is bullish
            close.iloc[-1] > open_.iloc[-1] and
            # Current candle's body length is greater than previous candle's body length
            open_.iloc[-1] <= close.iloc[-2] and close.iloc[-1] >= open_.iloc[-2]
        )


    def is_engulfing_bearish(open_, high, low, close):
        return (
            len(open_) > 1 and
            # Previous candle is bullish
            close.iloc[-2] > open_.iloc[-2] and
            # Current candle is bearish
            close.iloc[-1] < open_.iloc[-1] and
            # Current candle's body length is greater than previous candle's body length
            open_.iloc[-1] >= close.iloc[-2] and close.iloc[-1] <= open_.iloc[-2] 
        )


    def is_flag_bullish(open_, high, low, close):
        return (
            len(open_) > 3 and
            close.iloc[-4] > open_.iloc[-4] and  # Strong upward trend (flagpole)
            min(close.iloc[-3], open_.iloc[-3]) > min(close.iloc[-2], open_.iloc[-2]) > min(close.iloc[-1], open_.iloc[-1]) and
            max(close.iloc[-3], open_.iloc[-3]) > max(close.iloc[-2], open_.iloc[-2]) > max(close.iloc[-1], open_.iloc[-1]) and
            open_.iloc[-4] < close.iloc[-1] and   # Downward slope
            close.iloc[-1] > (high.iloc[-4] + low.iloc[-4]) / 2  # Close above the middle of the flag
        )


    def is_flag_bearish(open_, high, low, close):
        return (
            len(open_) > 3 and
            close.iloc[-4] < open_.iloc[-4] and  # Strong downward trend (flagpole)
            min(close.iloc[-3], open_.iloc[-3]) < min(close.iloc[-2], open_.iloc[-2]) < min(close.iloc[-1], open_.iloc[-1]) and
            max(close.iloc[-3], open_.iloc[-3]) < max(close.iloc[-2], open_.iloc[-2]) < max(close.iloc[-1], open_.iloc[-1]) and
            open_.iloc[-4] > close.iloc[-1] and    # Upward slope
            close.iloc[-1] < (high.iloc[-4] + low.iloc[-4]) / 2  # Close below the middle of the flag
        )

    def is_inside_candle(open_, high, low, close):
        return (len(open_) > 1 and
            low.iloc[-1] > low.iloc[-2] and high.iloc[-1] < high.iloc[-2] 
        )

    def is_special_engulfing_bullish(open_, high, low, close):
        return (
            len(open_) > 1 and
            close.iloc[-2] < open_.iloc[-2] and
            close.iloc[-1] < open_.iloc[-1] and
            low.iloc[-1] < low.iloc[-2] and
            high.iloc[-1] > high.iloc[-2] and
            close.iloc[-1] > high.iloc[-2]  # Special bullish engulfing with high breakout
        )

    def is_special_engulfing_bearish(open_, high, low, close):
        return (
            len(open_) > 1 and
            close.iloc[-2] > open_.iloc[-2] and
            close.iloc[-1] > open_.iloc[-1] and
            low.iloc[-1] < low.iloc[-2] and
            high.iloc[-1] > high.iloc[-2] and
            close.iloc[-1] < low.iloc[-2]  # Special bearish engulfing with low breakout
        )

    def is_retest_bullish(open_, high, low, close):
        return (
            len(open_) > 3 and
            high.iloc[-1] > close.iloc[-1] and
            close.iloc[-1] > high.iloc[-3] and 
            high.iloc[-3] > high.iloc[-2] and
            high.iloc[-2] > low.iloc[-1] and
            low.iloc[-1] > low.iloc[-3] and 
            low.iloc[-3] > low.iloc[-2]  # Retest bullish pattern
        )   

    def is_retest_bearish(open_, high, low, close):
        return (
            len(open_) > 3 and
            low.iloc[-1] < close.iloc[-1] and
            close.iloc[-1] < low.iloc[-3] and 
            low.iloc[-3] < low.iloc[-2] and
            low.iloc[-2] < high.iloc[-1] and
            high.iloc[-1] < high.iloc[-3] and
            high.iloc[-3] < high.iloc[-2]  # Retest bearish pattern
        )   

    def find_local_extrema(self, df, prominence=0.02, min_distance=3):
        """
        Identify local minima and maxima in a time series using scipy.signal.argrelextrema.
        
        Parameters:
        df (pd.DataFrame): A DataFrame with two columns, 'timestamp' and 'value'.
        
        Returns:
        dict: A dictionary containing the last 5 peaks and troughs with timestamps.
        """
        df = df.iloc[-1000:]  # Limit the data to the most recent 1000 rows for performance

        values = df.values
        timestamps = df.index.values
  
        # Find indices of local maxima and minima
        peak_indices, _ = find_peaks(values, prominence=prominence, distance=min_distance)
        troughs, _ = find_peaks(-values, prominence=prominence, distance=min_distance)
        
        last_5_peaks = [(timestamps[i], values[i]) for i in peak_indices[-5:]]
        last_5_troughs = [(timestamps[i], values[i]) for i in troughs[-5:]]
    
        local_maxima = last_5_peaks
        local_minima = last_5_troughs

        return local_minima, local_maxima

    def calculate_weighted_values(self, value):
        if value < 50:
            # Downtrend weighting: Amplify lower RSI values
            value = (50 - value) +50
            return value
        else:
            return value

#################################################################################################################################################
#
#                         Individual indicator calculation functions
#
#################################################################################################################################################

    
    def fill_missing_values(self, df, price_data, indicator, start_timestamp, current_timestamp, interval, **kwargs):
        """
        Fills missing values for a given indicator between the last recorded timestamp and the current timestamp.
        This version recalculates the entire indicator time series using both new and old price data if necessary.
        """

        # Function mappings for the new indicator calculations
        indicator_functions = {
            'ma': self.calculate_ma,
            'ema': self.calculate_ema,
            'vwma': self.calculate_vwma,
            'macd': self.macd_calculation,
            'bollinger': self.bollinger_calculation,
            'ao': self.calculate_ao,
            'cti': self.calculate_cti,
            'dm': self.calculate_dm,
            'fisher': self.calculate_fisher,
            'willr': self.calculate_willr,
            'vwap': self.calculate_vwap,
            'supertrend': self.calculate_supertrend,
            'atr': self.calculate_atr,
            'adx': self.calculate_adx,
            'rsi': self.calculate_rsi,
            'volume': self.calculate_volume,
            'chop': self.calculate_chop,
            'kama': self.calculate_kama,
            'vol' : self.calculate_rolling_volatility
        }
        
        if "ema" in indicator:
            indicator_func = "ema"
        elif "ma" in indicator and indicator not in ['macd', 'vwma','kama']:
            indicator_func = "ma" 
        elif "vwma" in indicator:
            indicator_func = "vwma"
        else:
            indicator_func = indicator

        calc_func = indicator_functions.get(indicator_func)

        # Get the last valid index for the current indicator data
        last_valid_index = df[indicator].last_valid_index()

        # If no valid data is found, start from scratch
        if last_valid_index is None:
            last_valid_index = start_timestamp
        else:
            # Adjust the last valid index to be a few steps back (for smoother recalculation)
            if interval == '1h':
                three_steps_back = last_valid_index - pd.DateOffset(hours=3)
            elif interval == '1d':
                three_steps_back = last_valid_index - pd.DateOffset(days=3)
            elif interval == '5m':
                three_steps_back = last_valid_index - pd.DateOffset(minutes=15)
            elif interval == '15m':
                three_steps_back = last_valid_index - pd.DateOffset(minutes=45)

        prices_existing_start = df['price'].first_valid_index()
        prices_existing_end = df['price'].last_valid_index()
        
        # If prices provided are insufficient, extend them using existing indicator data
        if df[df.columns[1]].notna().sum() > 0 and prices_existing_start < price_data.first_valid_index() and prices_existing_end >= price_data.first_valid_index():
            existing_prices = df.loc[prices_existing_start:price_data.first_valid_index(), 'price'].dropna()
            combined_prices = pd.concat([existing_prices, price_data], axis=0).drop_duplicates()
            combined_prices = combined_prices.sort_values('timestamp')
        else:
            combined_prices = price_data

        # Recalculate the entire indicator using the extended price data
        recalculated_values = calc_func(combined_prices, **kwargs)

        # Check for duplicate indices in combined_prices
        if combined_prices.index.duplicated().any():
            combined_prices = combined_prices[~combined_prices.index.duplicated(keep='last')]
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='last')]
        if recalculated_values.index.duplicated().any():
            recalculated_values = recalculated_values[~recalculated_values.index.duplicated(keep='last')]

        # Reindex df to include indices from combined_prices
        df = df.reindex(combined_prices.index)
        
        # Update the dataframe with the recalculated indicator values
        if indicator in ['macd']:
            df.loc[combined_prices.index, 'macd'] = recalculated_values.iloc[:,0]
            df.loc[combined_prices.index, 'macd_signal'] = recalculated_values.iloc[:,2]
        elif indicator in ['bollinger']:
            df.loc[combined_prices.index, 'bb_lower'] = recalculated_values.iloc[:,0]
            df.loc[combined_prices.index, 'bb_mid'] = recalculated_values.iloc[:,1]
            df.loc[combined_prices.index, 'bb_upper'] = recalculated_values.iloc[:,2]
            df.loc[combined_prices.index, 'bb_bandwith'] = recalculated_values.iloc[:,3]
            df.loc[combined_prices.index, 'bb_percent_col'] = recalculated_values.iloc[:,4]
        elif indicator in ['supertrend']:
            df.loc[combined_prices.index, 'sp_supertrend'] = recalculated_values.iloc[:,0]
            df.loc[combined_prices.index, 'sp_direction'] = recalculated_values.iloc[:,1]
        elif indicator in ['dm']:
            df.loc[combined_prices.index, 'dm_plus'] = recalculated_values.iloc[:,0]
            df.loc[combined_prices.index, 'dm_minus'] = recalculated_values.iloc[:,1]
        elif indicator in ['fisher']:
            df.loc[combined_prices.index, 'fisher'] = recalculated_values.iloc[:,0]
            df.loc[combined_prices.index, 'fisher_signal'] = recalculated_values.iloc[:,1]
        else:
            df.loc[combined_prices.index, indicator] = recalculated_values

        # Ensure no duplicate indices
        df = df[~df.index.duplicated(keep='last')]

        return df

    def calculate_indicator(self, price="usdt_kucoin", coin=None, fiat=None, interval=None, timestamp_input=None, indicator_name=None, overwrite_file=False, plot=False, ohlc="close", save_file=False, **kwargs):
        """
        Calculate a specified indicator and save to file.
        
        This function calculates a specified financial indicator for a given cryptocurrency and saves the result to a CSV file. 
            It can also plot the indicator if required.
            Parameters:
            - price (str): The type of price data to load. Options are "usdt", "usd", or "usdt_kucoin". Default is "usdt_kucoin".
            - coin (str): The cryptocurrency ticker symbol (e.g., 'BTC').
            - fiat (str): The fiat currency ticker symbol (e.g., 'USD').
            - interval (str): The time interval for the price data (e.g., '1d' for daily data).
            - timestamp_input (datetime): The timestamp to start the calculation from.
            - indicator_name (str): The name of the indicator to calculate.
            - overwrite_file (bool): Whether to overwrite the existing file if it exists. Default is False.
            - plot (bool): Whether to plot the indicator after calculation. Default is False.
            - ohlc (str): The type of OHLC data to use (e.g., 'close', 'open'). Default is 'close'.
            - **kwargs: Additional keyword arguments for specific indicators.
            Returns:
            - pd.DataFrame: A DataFrame containing the calculated indicator values.
        
        
        """
        if isinstance(price, pd.DataFrame):
            price_data = price
            price_data[self.ticker_to_slug_mapping[coin]] = price_data[ohlc].copy()
        elif price == "usdt":
            price_data = self.load_price_data(coin, fiat, interval, 'price_usdt')
        elif price == "usd":
            price_data = self.load_price_data(coin, fiat, interval, 'price_usd')
        elif price == "usdt_kucoin":
            price_data = self.load_price_data(coin, fiat, interval, "price_usdt_kucoin", ohlc=ohlc)
        

        price_data["price"] = price_data[self.ticker_to_slug_mapping[coin]]
        
        slug = self.ticker_to_slug_mapping[coin]
        filename = f"{indicator_name}_{coin.upper()}_{fiat.upper()}_{interval}.csv"
        path = os.path.join(data_path_crypto, "Historical data", indicator_name, interval)
        # price_data = price_data[slug]
        
        if not os.path.exists(path):
            os.makedirs(path)

        file_path = os.path.join(path, filename)
        
        if os.path.exists(file_path) and not overwrite_file:
            try:
                df = pd.read_csv(file_path, index_col=0)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                # df = pd.read_excel(file_path, index_col=0)
            except Exception as e:
                self.logger.error(f"Error reading {filename}: {e}")
                return None
        else:
            data_available = price_data.loc[:price_data[slug].last_valid_index()]    #[slug]
            # data_available = price_data.loc[:price_data.last_valid_index()]
            df = pd.DataFrame(index=data_available.index)
            df.index = df.index.rename('timestamp')
            df['price'] = price_data[f'{slug.lower()}']
            # if indicator_name == 
            for indicator in kwargs.keys():
                df[indicator] = np.nan
            
            if save_file:
                df.to_csv(file_path, index_label='timestamp')
            df.index = pd.to_datetime(df.index).tz_localize(None)
            # df.to_excel(file_path, index_label='timestamp')
            
        for indicator in kwargs.keys():
            df = self.fill_missing_values(df, price_data, indicator, df.index[0], timestamp_input, interval, **kwargs[indicator])

        if save_file:
            df.to_csv(file_path, index_label='timestamp')
    
        # df.to_excel(file_path, index_label='timestamp')
        if plot:
            self.plot_indicator(coin, df[-250:], indicator_name, path)
        
        return df

#################################################################################################################################################
#
#                         Indicator Aggregate Calculation Functions
#
#################################################################################################################################################

    def get_indicator_list(self):
        """Add the new indicators to the list."""
        indicator_list = [
            'ma', 'vwma', 'ema', 'short_ema', 'long_ema', 'high_ema', 'low_ema', 'close_ema', 'open_ema', 'volume_ema', 'macd', 
            'bollinger', 'bbandwith', 'hurst', 'vwap', 'supertrend', 'atr', 'adx', 'rsi', 'volume', 'ao', 'cti', 'dm', 'fisher', 'willr', "candlestick_patterns", "extrema", "chop", "rollvol", "kama"
        ]
        return indicator_list


    def calculate_indicator_data(self, price_df, indicators_to_calc, coin, fiat, interval, end_date):
        indicators = self.get_indicator_list()
        indicator_to_query = [["_".join(parts)] + parts for parts in (i.split("_") for i in indicators_to_calc.tolist()) if parts[0] in indicators]
        
        """"
        0 = complete metric name
        1 = indicator name
        2 = ohlc
        3 = params        
        """
        
        for indicator in indicator_to_query:
            
            self.logger.info(f"Calculating indicator: {indicator[0]}")
            
            if "ma" in indicator[1] and not "extrema" in indicator[1] and not "kama" in indicator[1] and not "macd" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval, 
                                    timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False, ohlc=indicator[2],   
                                    ma={'span': int(indicator[3])})
            
            elif "ema" in indicator[1] and not "extrema" in indicator[1] and not "kama" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval, 
                                    timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False, ohlc=indicator[2],   
                                    ema={'span': int(indicator[3])})
            
            elif "vwma" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval, 
                                    timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False, ohlc=indicator[2],   
                                    vwma={'span': int(indicator[3])})

            elif "macd" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval, 
                                    timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False, ohlc="close",   
                                    macd={'short_ema_span': int(indicator[3]), 'long_ema_span': int(indicator[4]), 'signal_span': int(indicator[5])})

            elif "bollinger" in indicator[1] or "bb" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval, 
                                    timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False, ohlc="close",   
                                    bollinger={'window': int(indicator[3]), 'num_std_dev': float(indicator[4])})
            
            elif "vwap" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval, 
                                    timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False, ohlc="close",   
                                    vwap={})
            
            elif "supertrend" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval, 
                                            timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False, ohlc="close",   
                                            supertrend={'period': int(indicator[3]), 'multiplier': float(indicator[4])})

            elif "rsi" in indicator[1] and not "extrema" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval, 
                                            timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False, ohlc="close",   
                                            rsi={'period': int(indicator[3])})

            elif "adx" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval, 
                                            timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False, ohlc="close",   
                                            adx={'di_length': int(indicator[3])})
            
            elif "atr" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval, 
                                            timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False, ohlc="close",   
                                            atr={'period': int(indicator[3])})
            
            elif "hurst" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval, 
                                            timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False, ohlc="close",   
                                            hurst={'lag_max': int(indicator[3])})
            
            elif "volume" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval, 
                                            timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False, ohlc="volume",   
                                            volume={})
            
            elif "ao" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval,
                                            timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False,
                                            ohlc="close", ao={'fast': int(indicator[3]), 'slow':int(indicator[4])})
            
            elif "cti" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval,
                                            timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False,
                                            ohlc="close", cti={'length': int(indicator[3])})

            elif "dm" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval,
                                            timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False,
                                            ohlc="close", dm={'length': int(indicator[3])})

            elif "fisher" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval,
                                            timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False,
                                            ohlc="high", fisher={'length': int(indicator[3]), 'signal': int(indicator[4])})

            elif "willr" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval,
                                            timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False,
                                            ohlc="close", willr={'length': int(indicator[3])})  
            elif "chop" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval,
                                            timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False,
                                            ohlc="close", chop={'length': int(indicator[3])})
            
            elif "kama" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval,
                                            timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False,
                                            ohlc="close", kama={'length': int(indicator[3])})

            # elif "candlestick_patterns" in indicator[1]:
            #     self.calculate_candlestick_patters("usdt_kucoin", patterns = "all", coin=coin, fiat=fiat, interval=interval,
            #                                 timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, ohlc="close")

            elif "rollvol" in indicator[1]:
                self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval,
                                            timestamp_input=end_date, indicator_name=indicator[0], overwrite_file=True, save_file=True, plot=False,
                                            ohlc="close", vol={'window': int(indicator[3])})
            
            elif "extrema" in indicator[1]:
                if indicator[2] == "close":
                    self.find_local_extrema_filtered(price_df["close"], window=price_df.reset_index().index[0], save_file=True, interval=interval, coin=coin, fiat=fiat, indicator_name=indicator[2])
                    
                elif indicator[2] == "rsi":
                    rsi = self.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval, 
                                            timestamp_input=end_date, indicator_name="rsi", overwrite_file=True, plot=False, ohlc="close",   
                                            rsi={'period': int(14)})
                    self.find_local_extrema_filtered(rsi["rsi"], window=price_df.reset_index().index[0], save_file=True, interval=interval, coin=coin, fiat=fiat, indicator_name=indicator[2])
                    
#################################################################################################################################################
#
#                         Plotting Functions        
#
#################################################################################################################################################

    def plot_indicator(self, coin, df, indicator_name, path):
        plt.figure(figsize=(12, 8))
        
        if indicator_name == 'ema':
            fig, ax1 = plt.subplots()
            ax1.plot(df.index, df['price'], label='Price')
            ax1.plot(df.index, df['short_ema'], label='Short EMA', color='orange')
            ax1.plot(df.index, df['long_ema'], label='Long EMA', color='red')
            ax1.set_title(f'{coin} {indicator_name.upper()}')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True)
            fig.autofmt_xdate()
            fig.savefig(os.path.join(path, f'{indicator_name}_plot.pdf'))
            plt.close(fig)
        
        elif indicator_name == 'macd':
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            
            ax1.plot(df.index, df['price'], label='Price')
            ax1.set_ylabel('Price')
            ax1.legend(loc='upper left')
            ax1.grid(True)
            
            ax2.plot(df.index, df['macd'], label='MACD', color='orange')
            ax2.plot(df.index, df['signal'], label='Signal', color='red')
            ax2.set_ylabel('MACD')
            ax2.legend(loc='upper left')
            ax2.grid(True)
            
            fig.suptitle(f'{coin} {indicator_name.upper()}')
            ax2.set_xlabel('Time')
            fig.autofmt_xdate()
            fig.savefig(os.path.join(path, f'{indicator_name}_plot.pdf'))
            plt.close(fig)
        
        elif indicator_name == 'bollinger':
            fig, ax1 = plt.subplots()
            ax1.plot(df.index, df['price'], label='Price')
            ax1.plot(df.index, df['upper_band'], label='Upper Band', color='orange')
            ax1.plot(df.index, df['lower_band'], label='Lower Band', color='red')
            ax1.set_title(f'{coin} {indicator_name.upper()}')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True)
            fig.autofmt_xdate()
            file_path = f"{path}/{indicator_name}_plot.pdf"
            fig.savefig(os.path.join(path, f'{indicator_name}_plot.pdf'))
            plt.close(fig)
            
# Example usage:
# logger = ...  # Your logger setup
# calculator = TAIndicatorCreator()
# data = calculator.price_usdt  # Example usage of loaded price data
# timestamp_input = pd.Timestamp.now(tz="UTC").floor('1h')
# timestamp_input = pd.Timestamp.now().floor('1h')  
# calculator.calculate_indicator("usdt_kucoin", timestamp_input=timestamp_input, indicator_name='ema', overwrite_file=True, short_ema={'span': 21}, long_ema={'span': 50})
# calculator.calculate_indicator("usdt_kucoin", coin="BTC",fiat="USDT", interval='1h', timestamp_input=timestamp_input, indicator_name='macd', overwrite_file=False, macd={'short_ema_span': 12, 'long_ema_span': 26, 'signal_span': 9}, plot=True)
# calculator.calculate_indicator("usdt_kucoin", timestamp_input=timestamp_input, indicator_name='bollinger', overwrite_file=True, bollinger={'window': 20, 'num_std_dev': 2})
# VWAP Calculation Example

# calculator.calculate_indicator("usdt_kucoin", coin="BTC", fiat="USDT", interval='1h', 
#                                timestamp_input=timestamp_input, 
#                                indicator_name='vwap', overwrite_file=True, plot=True)

# calculator.calculate_indicator("usdt_kucoin", coin="BTC", fiat="USDT", interval='1h', 
#                                timestamp_input=timestamp_input, 
#                                indicator_name='supertrend', overwrite_file=True, plot=True, 
#                                supertrend={'period': 10, 'multiplier': 3})


# # RSI Calculation Example
# timestamp_input = pd.Timestamp.now().floor('1h')
# calculator.calculate_indicator("usdt_kucoin", coin="BTC", fiat="USDT", interval='1h', 
#                                timestamp_input=timestamp_input, 
#                                indicator_name='rsi', overwrite_file=True, plot=True, 
#                                rsi={'period': 14})

# ADX Calculation Example
# timestamp_input = pd.Timestamp.now().floor('1h')
# calculator.calculate_indicator("usdt_kucoin", coin="BTC", fiat="USDT", interval='1h', 
#                                timestamp_input=timestamp_input, 
#                                indicator_name='adx', overwrite_file=True, plot=True, 
#                                adx={'period': 14})

# calculator.calculate_indicator("usdt_kucoin", coin="BTC", fiat="USDT", interval='1h', 
#                                timestamp_input=timestamp_input, 
#                                indicator_name='atr', overwrite_file=True, plot=True, 
#                                atr={'period': 14})

# calculator.calculate_indicator("usdt_kucoin", coin="BTC", fiat="USDT", interval='1h', 
#                                timestamp_input=timestamp_input, 
#                                indicator_name='hurst', overwrite_file=True, plot=True, 
#                                hurst={'lag_max': 40})

# calculator.calculate_indicator("usdt_kucoin", coin="BTC", fiat="USDT", interval='1h', 
#                                timestamp_input=timestamp_input, 
#                                indicator_name='volume', overwrite_file=True, plot=True, 
#                                volume={})