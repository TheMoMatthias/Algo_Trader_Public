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
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import platform
# from data_download_entire_history  import *

import os
import sys

def get_running_environment():
    if 'microsoft-standard' in platform.uname().release:
        return 'wsl'
    elif platform.system() == 'Windows':
        return 'windows'
    else:
        return 'unknown'

# Detect environment
env = get_running_environment()

base_path = os.path.dirname(os.path.realpath(__file__))


if env == 'wsl':
    crypto_bot_path = "/mnt/c/Users/mauri/Documents/Trading Bot/Python/AlgoTrader"
else:
    crypto_bot_path = os.path.dirname(base_path)

Python_path = os.path.dirname(crypto_bot_path)
hist_data_download_path = os.path.join(crypto_bot_path, "Hist Data Download")
san_api_data_path = os.path.join(hist_data_download_path, "SanApi Data")
main_data_files_path = os.path.join(san_api_data_path, "Main data files")

Trading_bot_path = os.path.dirname(Python_path)
Trading_path = os.path.join(Trading_bot_path, "Trading")
data_path_crypto = os.path.join(Trading_bot_path, "Data", "Cryptocurrencies")
histo_data_path_crypto = os.path.join(data_path_crypto, "Historical Data")
dataset_path_crypto = os.path.join(data_path_crypto, "Datasets")
csv_dataset_path = os.path.join(dataset_path_crypto, "crypto datasets", "csv")
hdf_dataset_path = os.path.join(dataset_path_crypto, "crypto datasets", "hdf5")
transformer_path = os.path.join(dataset_path_crypto, 'transformer')

trade_api_path =  os.path.join(crypto_bot_path,"API Trader")
backtest_path = os.path.join(crypto_bot_path, "Backtesting")
config_path = os.path.join(crypto_bot_path,"Config")
utils_path = os.path.join(Python_path, "Tools")
logging_path = os.path.join(Trading_bot_path, "Logging")
kucoin_api = os.path.join(crypto_bot_path,"Kucoin API")

sys.path.append(Python_path)
sys.path.append(crypto_bot_path)
sys.path.append(trade_api_path)
sys.path.append(backtest_path)
sys.path.append(utils_path)
sys.path.append(Trading_path)
sys.path.append(config_path)
sys.path.append(logging_path)
sys.path.append(data_path_crypto)
sys.path.append(histo_data_path_crypto)
sys.path.append(dataset_path_crypto)
sys.path.append(main_data_files_path)
sys.path.append(san_api_data_path)
sys.path.append(hist_data_download_path)
sys.path.append(kucoin_api)
sys.path.append(csv_dataset_path)
sys.path.append(hdf_dataset_path)

import mo_utils as utils

class StrategyEvaluatorBacktest():
    def __init__(self,logger=None):
    ############################################################################################################
    #                                             INIT CLASS
    ############################################################################################################
        self.logger = logger 
        
        # Retrieve all assets and create mapping with SAN coin names 
        self.all_assets = pd.read_excel(os.path.join(main_data_files_path, "all_assets.xlsx"), header=0)
        self.ticker_to_slug_mapping = dict(zip(self.all_assets['ticker'], self.all_assets['slug']))
        # self.slug  = self.ticker_to_slug_mapping[self.coin]
                
        self.logger.info("StrategyEvaluatorBacktest initialized")
    ############################################################################################################
    #                                             BASE FUNCTIONS
    ############################################################################################################
    
    def configure_dataset_logger(self):

        #logger
        current_datetime = dt.datetime.now()
        timestamp = current_datetime.strftime('%d-%m-%Y_%H-%M')
        log_file_name = f"dataset_logger_{timestamp}.txt"
        log_file_path = os.path.join(logging_path, log_file_name)

        if not os.path.exists(logging_path):
            os.makedirs(logging_path)

        self.logger_data.add(log_file_path, rotation="500 MB", level="INFO")
    
    # def load_price_data(self, metric):
    #     if metric in ['price_usdt',"price_usd"]:
    #         file_path = os.path.join(data_path_crypto, "Historical data", metric, self.frequency_backtest, f"{metric}_{self.frequency_backtest}.csv")
    #         if not os.path.exists(file_path):
    #             self.logger.error(f"FILE NON EXISTING: The data is not available for {metric} under {file_path}. Please investigate")
    #             return None
    #         df = pd.read_csv(file_path, index_col=0)
    #     else:
    #         file_path = os.path.join(data_path_crypto, "Historical data", metric, self.frequency_backtest, f"{metric}_{self.coin}_{self.fiat}_{self.frequency_backtest}.csv")
    #         if not os.path.exists(file_path):
    #             self.logger.error(f"FILE NON EXISTING: The data is not available for {metric} under {file_path}. Please investigate")
    #             return None
    #         df = pd.read_csv(file_path, index_col=0)
    #         df = df.rename(columns={"close":self.slug})    #renaming close value to coin slug from sanpy for better adoption
    #     df.index = pd.to_datetime(df.index).tz_localize(None)
    #     return df

    ############################################################################################################
    #                                             EMA STRATEGY
    ############################################################################################################

    def macd_cross_and_ema_trend_checker(self, coin, fiat, interval, trend_interval, timestamp_input):
        """
        EMA Cross strategy
        #uses tm1 timestamp to check if values crossed on hourly and todays timestamps for multi timeframe strategy
        """
        
        if isinstance(timestamp_input, dt.datetime):
            utc_time = pd.Timestamp(timestamp_input)
        
        utc_time = utc_time.tz_localize('Europe/Berlin', ambiguous=True).tz_convert('UTC')
        utc_time = utc_time.replace(tzinfo=None)
        
        if trend_interval == '1d':
            utc_time = utc_time.replace(hour=0, minute=0, second=0, microsecond=0) # - timedelta(days=1) uncomment to get data
            
        # Utility function to get offset datetime
        trend_metric_name = "ema"
        trend_path = os.path.join(histo_data_path_crypto, trend_metric_name, trend_interval)
        trend_filename = f"{trend_metric_name}_{coin}_{fiat}_{trend_interval}.csv"
        
        # Load trend signal
        if os.path.exists(os.path.join(trend_path,trend_filename)):
            df_trend = pd.read_csv(os.path.join(trend_path,trend_filename), index_col='timestamp', parse_dates=True)
            df_trend.index = pd.to_datetime(df_trend.index)
            if utc_time not in df_trend.index:
                # self.logger.info(f"Timestamp {timestamp_input} not in DataFrame.")
                current_trend = "neutral"
        else:
            self.logger.error(f"File {filename} does not exist.")
            return
        
        current_higher_ema = 'short_ema' if df_trend.loc[utc_time, 'short_ema'] > df_trend.loc[utc_time, 'long_ema'] else 'long_ema'
        
        if df_trend.loc[utc_time, 'short_ema'] > df_trend.loc[utc_time, 'long_ema']:
            current_trend = "bullish"
        elif df_trend.loc[utc_time, 'short_ema'] < df_trend.loc[utc_time, 'long_ema']:
            current_trend = "bearish"
        else:
            current_trend = "neutral"
            
        self.logger.info(f"current trend is {current_trend}. Current higher ema is {current_higher_ema} at time {utc_time}")
        
        # Load signal
        if interval == '1h':
            t_1 = (timestamp_input - pd.Timedelta(hours=1))
            t_2 = timestamp_input - pd.Timedelta(hours=2)
        else:
            t_1 = timestamp_input - pd.Timedelta(days=1)
            t_2 = timestamp_input - pd.Timedelta(days=2)
        
        metric_name = "macd"
        path = os.path.join(histo_data_path_crypto, metric_name, interval)
        filename = f"{metric_name}_{coin}_{fiat}_{interval}.csv"
        
        if os.path.exists(os.path.join(path,filename)):
            df = pd.read_csv(os.path.join(path,filename), index_col='timestamp', parse_dates=True)
            df.index = pd.to_datetime(df.index)
            if t_1 not in df.index or t_2 not in df.index:
                self.logger.error(f"Timestamp {t_1} or {t_2} not in DataFrame.")
                return "neutral"
        else:
            self.logger.error(f"File {filename} does not exist.")
            return

        current_highest = 'macd' if df.loc[t_1, 'macd'] > df.loc[t_1, 'signal'] else 'signal'
        previous_highest = 'macd' if df.loc[t_2, 'macd'] > df.loc[t_2, 'signal'] else 'signal'
        self.logger.info(f"current macd is {df.loc[t_1, 'macd']:.2f} and current signal is {df.loc[t_1, 'signal']:.2f}. Current higher value is {current_highest} at time {t_1}")
        self.logger.info(f"previous macd is {df.loc[t_2, 'macd']:.2f} and previous signal is {df.loc[t_2, 'signal']:.2f}. Previous highest is {previous_highest} at time {t_2}") 

        # strategy conditions
        # Calculate positions based on EMA crossovers
        if df.loc[t_1, 'macd'] > df.loc[t_1, 'signal'] and df.loc[t_2, 'macd'] < df.loc[t_2, 'signal'] and current_trend == "bullish":
            signal = "buy"
        elif df.loc[t_1, 'macd'] < df.loc[t_1, 'signal'] and df.loc[t_2, 'macd'] > df.loc[t_2, 'signal'] and current_trend == "bearish":
            signal = "sell"
        else:
            signal = "neutral"

        return signal

    
    def ema_cross_strategy(self, coin, fiat, interval, timestamp_input, logger=None):
        """
        EMA Cross strategy
        """
        
        # Utility function to get offset datetime
        if interval == '1h':
            t_1 = timestamp_input - pd.Timedelta(hours=1)
        elif interval == '1d':
            t_1 = timestamp_input - pd.Timedelta(days=1)
        
        metric_name = "ema"
        path = os.path.join(histo_data_path_crypto, metric_name, interval)
        filename = f"{metric_name}_{coin}_{fiat}_{interval}.csv"
        
        # Load or initialize DataFrame
        if os.path.exists(os.path.join(path,filename)):
            df = pd.read_csv(os.path.join(path,filename), index_col='timestamp', parse_dates=True)
            if timestamp_input not in df.index or t_1 not in df.index:
                    logger.error(f"Timestamp {timestamp_input} not in DataFrame.")
                    return "neutral", None
        else:
            logger.error(f"File {filename} does not exist.")
            return

        current_higher_ema = 'short_ema' if df.loc[timestamp_input, 'short_ema'] > df.loc[timestamp_input, 'long_ema'] else 'long_ema'
        previous_higher_ema = 'short_ema' if df.loc[t_1, 'short_ema'] > df.loc[t_1, 'long_ema'] else 'long_ema'
        logger.info(f"current short ema is {df.loc[timestamp_input, 'short_ema']} and current long ema is {df.loc[timestamp_input, 'long_ema']}. Current higher ema is {current_higher_ema} at time {timestamp_input}")
        logger.info(f"previous short ema is {df.loc[t_1, 'short_ema']} and previous long ema is {df.loc[t_1, 'long_ema']}. Previous higher ema is {previous_higher_ema} at time {t_1}") 

        # strategy conditions
        # Calculate positions based on EMA crossovers
        if df.loc[timestamp_input, 'short_ema'] > df.loc[timestamp_input, 'long_ema'] and df.loc[t_1, 'short_ema'] < df.loc[t_1, 'long_ema']:
            signal = "buy"
        elif df.loc[timestamp_input, 'short_ema'] < df.loc[timestamp_input, 'long_ema'] and df.loc[t_1, 'short_ema'] > df.loc[t_1, 'long_ema']:
            signal = "sell"
        else:
            signal = "neutral"

        # df.loc[timestamp_input, 'position'] = signal
          # Save updated DataFrame

        return signal, df
    
    def macd_cross_strategy(self, coin, fiat, interval, timestamp_input, logger=None):
        """
        MACD Cross strategy
        """
        # Utility function to get offset datetime
        
        if interval == '1h':
            t_1 = timestamp_input - pd.Timedelta(hours=1)
        elif interval == '1d':
            t_1 = timestamp_input - pd.Timedelta(days=1)
        
        metric_name = "macd"
        path = os.path.join(histo_data_path_crypto, metric_name, interval)
        filename = f"{metric_name}_{coin}_{fiat}_{interval}.csv"
        
        # Load or initialize DataFrame
        if os.path.exists(os.path.join(path,filename)):
            df = pd.read_csv(os.path.join(path,filename), index_col='timestamp', parse_dates=True)
            if timestamp_input not in df.index or t_1 not in df.index:
                logger.error(f"Timestamp {timestamp_input} not in DataFrame.")
                return "neutral", None
        else:
            logger.error(f"File {filename} does not exist.")
            return

        current_highest = 'macd' if df.loc[timestamp_input, 'macd'] > df.loc[timestamp_input, 'signal'] else 'signal'
        previous_highest = 'macd' if df.loc[t_1, 'macd'] > df.loc[t_1, 'signal'] else 'signal'
        logger.info(f"current macd is {df.loc[timestamp_input, 'macd']:.2f} and current signal is {df.loc[timestamp_input, 'signal']:.2f}. Current higher value is {current_highest} at time {timestamp_input}")
        logger.info(f"previous macd is {df.loc[t_1, 'macd']:.2f} and previous signal is {df.loc[t_1, 'signal']:.2f}. Previous highest is {previous_highest} at time {t_1}") 

        # strategy conditions
        # Calculate positions based on EMA crossovers
        if df.loc[timestamp_input, 'macd'] > df.loc[timestamp_input, 'signal'] and df.loc[t_1, 'macd'] < df.loc[t_1, 'signal']:
            signal = "buy"
        elif df.loc[timestamp_input, 'macd'] < df.loc[timestamp_input, 'signal'] and df.loc[t_1, 'macd'] > df.loc[t_1, 'signal']:
            signal = "sell"
        else:
            signal = "neutral"

        # df.loc[timestamp_input, 'position'] = signal
          # Save updated DataFrame

        return signal, df




    