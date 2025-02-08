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
import pytz
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
    # crypto_bot_path = "/mnt/c/Users/mauri/Documents/Trading Bot/Python/AlgoTrader" 
    crypto_bot_path = os.path.dirname(base_path)
else:
    crypto_bot_path = os.path.dirname(base_path)

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
                AWS_path, GOOGLE_path, 
                data_loader, hist_data_download_kucoin, strategy_path]

# Add paths to sys.path and verify
for path in paths_to_add:
    if path not in sys.path:
        sys.path.append(path)


import mo_utils as utils
from utilsAWS import S3Utility
from utilsGoogleDrive import GoogleDriveUtility
from KuCoin_Prices import KuCoinDataDownloader
from TAIndicatorCreator import TAIndicatorCreator


class StrategyEvaluator():
    def __init__(self,logger=None, coin=None, fiat=None, interval=None):
    ############################################################################################################
    #                                             INIT CLASS
    ############################################################################################################
        self.logger = logger 
        
        #config
        config_path = utils.find_config_path() 
        config = utils.read_config_file(os.path.join(config_path,"AlgoTrader_config.ini"))
        
        # #init aws
        # self.key = config["AWS"]["access_key"]
        # self.sec_key = config["AWS"]["secret_access_key"]
        # self.bucket = config["AWS"]["bucket_name"]
        # self.arn_role = config["AWS"]["arn_role"]
        # self.region_name = config["AWS"]["region_name"]
        
        #init kucoin price loader
        self.kucoin_price_loader = KuCoinDataDownloader(created_logger=self.logger)
    
        #init TA creator
        self.TA_creator = TAIndicatorCreator(logger=self.logger)
        
        
        # Retrieve all assets and create mapping with SAN coin names 
        self.all_assets = pd.read_excel(os.path.join(main_data_files_path, "all_assets.xlsx"), header=0)
        self.ticker_to_slug_mapping = dict(zip(self.all_assets['ticker'], self.all_assets['slug']))
        # self.slug  = self.ticker_to_slug_mapping[self.coin]
                
        self.logger.info("StrategyEvaluator initialized")
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

    def macd_cross_and_ema_trend_checker(self, coin, fiat, interval, trend_interval, timestamp_input_interval_adj, timestamp_input_actual, logger=None, simulate_live_data=False, verbose=False):
        """
        EMA Cross strategy
        #uses tm1 timestamp to check if values crossed on hourly and todays timestamps for multi timeframe strategy
        """
        
        price_data, minute_data = self.kucoin_price_loader.download_data(coin, fiat, interval, start_date="2017-10-05 00:00:00", end_date=timestamp_input_actual, use_local_timezone=True, drop_not_complete_timestamps=False, simulate_live_data=simulate_live_data, overwrite_file=False)   #use timestamp input and t_1 when you want to use uncompleted current timestep values
        
        df_macd = self.TA_creator.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval, 
                                            timestamp_input=timestamp_input_interval_adj, indicator_name='macd', overwrite_file=True, plot=False,    
                                            macd={'short_ema_span': 12, 'long_ema_span': 26, 'signal_span': 9})
        
        self.kucoin_price_loader.download_data(coin, fiat, trend_interval, start_date="2018-01-01 00:00:00", end_date=timestamp_input_actual, use_local_timezone=True, drop_not_complete_timestamps=False, simulate_live_data=simulate_live_data, overwrite_file=False)
        
        df_trend = self.TA_creator.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=trend_interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ema', overwrite_file=True, plot=False, 
                                        short_ema={'span': 3}, long_ema={'span': 9})
        
        trend_timestamp = timestamp_input_interval_adj
        
        if isinstance(trend_timestamp, dt.datetime):
            trend_timestamp = pd.Timestamp(trend_timestamp)
        
        if not np.all(price_data.localized):
            trend_timestamp = trend_timestamp.tz_localize('Europe/Berlin', ambiguous=True).tz_convert('UTC')
            trend_timestamp = trend_timestamp.replace(tzinfo=None)
        
        if trend_interval == '1d':
            trend_timestamp = trend_timestamp.replace(tzinfo=None)
            trend_timestamp = trend_timestamp.replace(hour=0, minute=0, second=0, microsecond=0) # - timedelta(days=1) uncomment to get data
        elif trend_interval == '1h':
            trend_timestamp = trend_timestamp.replace(tzinfo=None)
            trend_timestamp = trend_timestamp.replace(minute=0, second=0, microsecond=0)  #- timedelta(hours=1) #Using tm1 as hour is not completed yet       uncomment to get data
        elif trend_interval == '15m':
            trend_timestamp = trend_timestamp.replace(tzinfo=None)      
            trend_timestamp = trend_timestamp.replace(second=0, microsecond=0)
            trend_timestamp = trend_timestamp.replace(minute=15 * (trend_timestamp.minute // 15)) # - timedelta(minutes=15) #Using tm1 as 15m is not completed yet       uncomment to get data
        elif trend_interval == '5m':
            trend_timestamp = trend_timestamp.replace(tzinfo=None)      
            trend_timestamp = trend_timestamp.replace(second=0, microsecond=0)
            trend_timestamp = trend_timestamp.replace(minute=5 * (trend_timestamp.minute // 5))  # - timedelta(minutes=5) #Using tm1 as 5m is not completed yet       uncomment to get data
            
        
        # df_trend = df_trend.set_index('timestamp')
        df_trend.index = pd.to_datetime(df_trend.index)
        
        # Define the timezones
        utc_timezone = pytz.timezone('UTC')
        berlin_timezone = pytz.timezone('Europe/Berlin')
        utc_now = datetime.now(utc_timezone)
        berlin_now = utc_now.astimezone(berlin_timezone)
        offset_hours = int(berlin_now.utcoffset().total_seconds() // 3600)

        if not offset_hours >= trend_timestamp.hour >=0:
            trend_timestamp = trend_timestamp
        else:
            if trend_interval=="1d":
                trend_timestamp = trend_timestamp - pd.Timedelta(days=1)
            elif trend_interval == "1h":
                trend_timestamp  = trend_timestamp - pd.Timedelta(hours=offset_hours)
    

        if trend_timestamp not in df_trend.index:
            # self.logger.info(f"Timestamp {timestamp_input} not in DataFrame.")
            current_trend = "neutral"
            return current_trend
        
        current_higher_ema = 'short_ema' if df_trend.loc[trend_timestamp, 'short_ema'] > df_trend.loc[trend_timestamp, 'long_ema'] else 'long_ema'
        
        if df_trend.loc[trend_timestamp, 'short_ema'] > df_trend.loc[trend_timestamp, 'long_ema']:
            current_trend = "bullish"
        elif df_trend.loc[trend_timestamp, 'short_ema'] < df_trend.loc[trend_timestamp, 'long_ema']:
            current_trend = "bearish"
        else:
            current_trend = "neutral"
        
        if verbose:
            self.logger.info(f"current trend is {current_trend}. Current higher ema is {current_higher_ema} at time {trend_timestamp}")
        
        # Load signal
        if interval == '1h':
            t_1 = (timestamp_input_interval_adj - pd.Timedelta(hours=1))
            t_2 = timestamp_input_interval_adj - pd.Timedelta(hours=2)
        elif interval == '1d':
            t_1 = timestamp_input_interval_adj - pd.Timedelta(days=1)
            t_2 = timestamp_input_interval_adj - pd.Timedelta(days=2)
        elif interval == '15m':
            t_1 = timestamp_input_interval_adj - pd.Timedelta(minutes=15)
            t_2 = timestamp_input_interval_adj - pd.Timedelta(minutes=30)
        elif interval == '5m':
            t_1 = timestamp_input_interval_adj - pd.Timedelta(minutes=5)
            t_2 = timestamp_input_interval_adj - pd.Timedelta(minutes=10)
        
        
        # df_macd = df_macd.set_index('timestamp')
        df_macd.index = pd.to_datetime(df_macd.index)
        
        if timestamp_input_interval_adj not in df_macd.index or t_1 not in df_macd.index:
            self.logger.error(f"Timestamp {timestamp_input_interval_adj} or {t_1} not in DataFrame.")
            return "neutral"
        
        current_highest = 'macd' if df_macd.loc[timestamp_input_interval_adj, 'macd'] > df_macd.loc[timestamp_input_interval_adj, 'macd_signal'] else 'signal'     #use timestamp input and t_1 when you want to use uncompleted current timestep values
        previous_highest = 'macd' if df_macd.loc[t_1, 'macd'] > df_macd.loc[t_1, 'macd_signal'] else 'signal'
        
        
        #logging of strategy
        
        if simulate_live_data:
            minute_data = minute_data.set_index('time')
            minute_data.index = pd.to_datetime(minute_data.index)
            
        prev_price = price_data.loc[t_1, "close"]
        
        if verbose:

            self.logger.info(f"{'#'*1} Current interval time is {timestamp_input_interval_adj} with current interval price at {price_data.loc[timestamp_input_interval_adj, 'close']:.2f}. {'#'*1} ")
            
            if simulate_live_data:
                try:
                    self.logger.info(f"{'#'*1} Current  actual time is {timestamp_input_actual} with current price at {minute_data.loc[timestamp_input_actual, 'close']:.2f}. {'#'*1} ")
                except:
                    pass
            
            self.logger.info(f"{'#'*1} Previous time is {t_1} with previous price of {prev_price}. {'#'*1} ")
            
            self.logger.info(f"current macd is {df_macd.loc[timestamp_input_interval_adj, 'macd']:.2f} and current signal is {df_macd.loc[timestamp_input_interval_adj, 'macd_signal']:.2f}. Current higher value is {current_highest} at time {timestamp_input_interval_adj}")
            self.logger.info(f"previous macd is {df_macd.loc[t_1, 'macd']:.2f} and previous signal is {df_macd.loc[t_1, 'macd_signal']:.2f}. Previous highest is {previous_highest} at time {t_1}") 

        # strategy conditions
        # Calculate positions based on EMA crossovers
        if df_macd.loc[timestamp_input_interval_adj, 'macd'] > df_macd.loc[timestamp_input_interval_adj, 'macd_signal'] and df_macd.loc[t_1, 'macd'] < df_macd.loc[t_1, 'macd_signal'] and current_trend == "bullish":
            signal = "buy"
        elif df_macd.loc[timestamp_input_interval_adj, 'macd'] < df_macd.loc[timestamp_input_interval_adj, 'macd_signal'] and df_macd.loc[t_1, 'macd'] > df_macd.loc[t_1, 'macd_signal'] and current_trend == "bearish":
            signal = "sell"
        else:
            signal = "neutral"

        return signal
    
    
    def ma_hl_div_vol(self, coin, fiat, interval, trend_interval, timestamp_input_interval_adj, timestamp_input_actual, logger=None, simulate_live_data=False, verbose=False):
        """
        EMA Cross strategy
        #uses tm1 timestamp to check if values crossed on hourly and todays timestamps for multi timeframe strategy

        if trend interval is not specified the function does not use multitimeframe approach
        """

        price_data, minute_data = self.kucoin_price_loader.download_data(coin, fiat, interval, end_date=timestamp_input_actual, 
                                                            use_local_timezone=True, drop_not_complete_timestamps=False, simulate_live_data=simulate_live_data, overwrite_file=False)    #start_date="2017-10-05 00:00:00"  #use timestamp input and t_1 when you want to use uncompleted current timestep values
        
        ma_high = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                            timestamp_input=timestamp_input_interval_adj, indicator_name='ma_high', overwrite_file=True, plot=False, ohlc="high",   
                                            high_ma={'span': 20})
        
        ma_low = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ma_low', overwrite_file=True, plot=False, ohlc="low", 
                                        low_ma={'span': 20})
        
        adx = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='adx', overwrite_file=True, plot=False, ohlc="close", 
                                        adx={'di_length': 14})
        
        rsi = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='rsi', overwrite_file=True, plot=False, ohlc="close", 
                                        rsi={'period': 14})

        rsi_7 = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='rsi', overwrite_file=True, plot=False, ohlc="close", 
                                        rsi={'period': 7})

        atr = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='atr', overwrite_file=True, plot=False, ohlc="close", 
                                        atr={'period': 7})

        volume_ma = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ma', overwrite_file=True, plot=False, ohlc="turnover", 
                                        ma={'span': 50})
        
        relative_volume = (price_data['turnover'] / volume_ma["ma"])

        ema_long = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ema', overwrite_file=True, plot=False, ohlc="close", 
                                        ema={'span': 38})

        chop_index = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='chop', overwrite_file=True, plot=False, ohlc="close", 
                                        chop={'length': 14})

        # local_minima_price, local_maxima_price  = self.TA_creator.find_local_extrema(price_data["close"])
        local_minima_price, local_maxima_price = self.TA_creator.find_local_extrema_filtered(price_data["close"])

        # local_minima_rsi, local_maxima_rsi = self.TA_creator.find_local_extrema(rsi["rsi"])
        local_minima_rsi, local_maxima_rsi = self.TA_creator.find_local_extrema_filtered(rsi["rsi"])
        minima_extrema_diff = abs(rsi.loc[timestamp_input_interval_adj, "rsi"] - local_minima_rsi[-1][1])
        maxima_extrema_diff = abs(local_maxima_rsi[-1][1] - rsi.loc[timestamp_input_interval_adj, "rsi"])

        trend_score_simple = 0.4 * rsi["rsi"] + 0.3 * adx["adx"] + 0.3 * relative_volume

        ma_low.index = pd.to_datetime(ma_low.index)
        ma_high.index = pd.to_datetime(ma_high.index)
        atr.index = pd.to_datetime(atr.index)
        rsi.index = pd.to_datetime(rsi.index)
        rsi_7.index = pd.to_datetime(rsi_7.index)
        adx.index = pd.to_datetime(adx.index)
        relative_volume.index = pd.to_datetime(relative_volume.index)
        ema_long.index = pd.to_datetime(ema_long.index)
        chop_index.index = pd.to_datetime(chop_index.index)
       
        if timestamp_input_interval_adj not in ma_low.index or timestamp_input_interval_adj not in ma_high.index:
            # self.logger.info(f"Timestamp {timestamp_input} not in DataFrame.")
            current_trend = "neutral"
            return current_trend
        
        last_used_minute_data = minute_data.loc[minute_data['time'] <= timestamp_input_actual, 'time'].iloc[-1]
        minute_data = minute_data.set_index('time') 

        last_available_timestep = price_data.index[-2]
        trend_window_index = price_data.index[-4] 

        buy_tm1_price = min(price_data.loc[last_available_timestep,["open","close"]])
        sell_tm1_price = max(price_data.loc[last_available_timestep,["open","close"]])
        prev_price = price_data.loc[last_available_timestep,'close']

        #check if last 3 prices were above low ema 
        buy_trend_values = [min(price_data.loc[step, ["open","close"]]) for step in price_data[trend_window_index:last_available_timestep].index]
        
        #check if last 3 prices were below high ema
        sell_trend_values = [max(price_data.loc[step, ["open","close"]]) for step in price_data[trend_window_index:last_available_timestep].index]
        
        
        if verbose:
            # self.logger.info(f"{'#'*1} current trend is {current_trend} {'#'*1} with macd {macd.loc[utc_time, 'macd']:.2f} and signal {macd.loc[utc_time, 'signal']:.2f} at time {utc_time}")
            
            
            if simulate_live_data:
                try:
                    
                    self.logger.info(f"{'#'*1} Interval time:{timestamp_input_interval_adj}, interval price: {price_data.loc[timestamp_input_interval_adj, 'close']:.2f} # actual time: {timestamp_input_actual} minute price: {minute_data.loc[last_used_minute_data, 'close']:.2f} # Previous time: {last_available_timestep}, previous price: {prev_price} ")
                except:
                    pass
            else:
                
                self.logger.info(f"{'#'*1} Interval time:{timestamp_input_interval_adj}, interval price: {price_data.loc[timestamp_input_interval_adj, 'close']:.2f} # Previous time: {last_available_timestep}, previous price: {prev_price} #")
            
            self.logger.info(f"{'#'*1} high_ma: {ma_high.loc[timestamp_input_interval_adj, 'high_ma']:.2f} # low_ma: {ma_low.loc[timestamp_input_interval_adj, 'low_ma']:.2f} # adx: {adx.loc[timestamp_input_interval_adj,'adx']} # rsi: {rsi.loc[timestamp_input_interval_adj,'rsi']} # rel. volume: {np.round(relative_volume.loc[timestamp_input_interval_adj],4)} # atr: {atr.loc[timestamp_input_interval_adj,'atr']} # trend sc.: {np.round(trend_score_simple.loc[timestamp_input_interval_adj],3)} # chop: {chop_index.loc[timestamp_input_interval_adj,'chop']} {'#'*1} ")
            
        # price_buy_zone = ma_high.loc[timestamp_input_interval_adj, 'high_ma'] * 0.995 
        # price_sell_zone = ma_low.loc[timestamp_input_interval_adj, 'low_ma'] * 1.005
        
        #ma condition buy
        # price_cond_buy_1 =   (price_data.loc[timestamp_input_interval_adj, "close"] < ma_high.loc[timestamp_input_interval_adj, 'high_ma'] and 
        #                       buy_tm1_price >= ma_high.loc[last_available_timestep, 'high_ma'] and 
        #                       rsi.loc[timestamp_input_interval_adj, 'rsi'] > 50 and 
        #                       adx.loc[timestamp_input_interval_adj, 'adx'] > 25  and
        #                       price_data.loc[timestamp_input_interval_adj, "close"] > price_buy_zone)     #price under high ma & prev  low above ma
        
        #strong break through high and low ma
        price_cond_buy_1 =  (price_data.loc[timestamp_input_interval_adj, "close"] < ma_low.loc[timestamp_input_interval_adj, 'low_ma'] and 
                              price_data.loc[timestamp_input_interval_adj, "high"] > ma_low.loc[timestamp_input_interval_adj, 'low_ma'] and
                              np.all(buy_trend_values >= ma_low.loc[last_available_timestep, 'low_ma']) and 
                              rsi.loc[timestamp_input_interval_adj, 'rsi'] > 50 and adx.loc[timestamp_input_interval_adj, 'adx'] > 25  and 
                              chop_index.loc[timestamp_input_interval_adj, "chop"] < 50)     #price under high ma & prev  low above ma
        
        # #strong break through high and low ma                       #remove condition as causing too many losses
        # price_cond_buy_2 = (relative_volume[timestamp_input_interval_adj] > 1.5 and 
        #                     price_data.loc[timestamp_input_interval_adj, "close"] > ema_long.loc[timestamp_input_interval_adj, 'ema'] and 
        #                     price_data.loc[timestamp_input_interval_adj, "low"] < ma_low.loc[timestamp_input_interval_adj, 'low_ma'] and 
        #                     price_data.loc[timestamp_input_interval_adj, "close"] > ma_high.loc[timestamp_input_interval_adj, 'high_ma'] and
        #                     price_data.loc[timestamp_input_interval_adj, "open"] > ma_low.loc[timestamp_input_interval_adj, 'low_ma'])
        
        #volume indicator
        price_cond_buy_2 = (#adx.loc[timestamp_input_interval_adj,"adx"] < 25 and 
                            rsi.loc[timestamp_input_interval_adj,"rsi"] > 50 and 
                            rsi.loc[timestamp_input_interval_adj,"rsi"] < 80 and 
                            price_data.loc[timestamp_input_interval_adj, "close"] >  price_data.loc[timestamp_input_interval_adj, "open"] and
                            relative_volume.loc[timestamp_input_interval_adj] > 2 and
                            chop_index.loc[timestamp_input_interval_adj, "chop"] < 50)

        #rsi divergence
        price_cond_buy_3 = (local_minima_price[-1][1] < local_minima_price[-2][1] and 
                            local_minima_rsi[-1][1] > local_minima_rsi[-2][1] and
                            local_minima_price[-1][0] == local_minima_rsi[-1][0] and
                            local_minima_price[-2][0] == local_minima_rsi[-2][0] and
                            local_minima_price[-1][0] >= price_data.index[-3] and 
                            30 > trend_score_simple.loc[timestamp_input_interval_adj] > 25 and
                            timestamp_input_actual.minute >= 57 and
                            minima_extrema_diff < 10 and
                            chop_index.loc[timestamp_input_interval_adj, "chop"] < 50)  #adjust so that signal can max be two candles away, prevent delayed trigger mechanism
        
        #rsi hidden divergence
        price_cond_buy_4 = (local_minima_price[-1][1] > local_minima_price[-2][1] and 
                            local_minima_rsi[-1][1] < local_minima_rsi[-2][1] and
                            local_minima_price[-1][0] == local_minima_rsi[-1][0] and
                            local_minima_price[-2][0] == local_minima_rsi[-2][0] and
                            local_minima_price[-1][0] >= price_data.index[-3] and
                            30 > trend_score_simple.loc[timestamp_input_interval_adj] > 25 and
                            timestamp_input_actual.minute >= 57 and
                            minima_extrema_diff < 10 and 
                            chop_index.loc[timestamp_input_interval_adj, "chop"] < 50)   #adjust so that signal can max be two candles away, prevent delayed trigger mechanism)

        #ma condition sell
        price_cond_sell_1 = (price_data.loc[timestamp_input_interval_adj, "close"] > ma_high.loc[timestamp_input_interval_adj, 'high_ma'] and 
                             price_data.loc[timestamp_input_interval_adj, "low"] < ma_high.loc[timestamp_input_interval_adj, 'high_ma'] and
                             np.all(sell_trend_values <= ma_high.loc[last_available_timestep, 'high_ma']) and   #price above low ma & prev high below ma 
                             rsi.loc[timestamp_input_interval_adj, 'rsi'] < 50 and adx.loc[timestamp_input_interval_adj, 'adx'] > 25  and
                             chop_index.loc[timestamp_input_interval_adj, "chop"] < 50) 
        
        # #strong break through high and low ma
        # price_cond_sell_2 = (relative_volume[timestamp_input_interval_adj] > 1.5 and
        #                     price_data.loc[timestamp_input_interval_adj, "close"] < ema_long.loc[timestamp_input_interval_adj, 'ema'] and
        #                     price_data.loc[timestamp_input_interval_adj, "high"] > ma_high.loc[timestamp_input_interval_adj, 'high_ma'] and
        #                     price_data.loc[timestamp_input_interval_adj, "close"] < ma_low.loc[timestamp_input_interval_adj, 'low_ma'] and
        #                     price_data.loc[timestamp_input_interval_adj, "open"] > ma_high.loc[timestamp_input_interval_adj, 'high_ma'])

        #volume indicator 
        price_cond_sell_2 = (#adx.loc[timestamp_input_interval_adj,"adx"] < 25 and 
                             rsi.loc[timestamp_input_interval_adj,"rsi"] < 50 and 
                             rsi.loc[timestamp_input_interval_adj,"rsi"] > 20 and 
                             price_data.loc[timestamp_input_interval_adj, "close"] <  price_data.loc[timestamp_input_interval_adj, "open"] and
                             relative_volume.loc[timestamp_input_interval_adj] > 2 and
                             chop_index.loc[timestamp_input_interval_adj, "chop"] < 50)
        #rsi divergence
        price_cond_sell_3 = (local_maxima_price[-1][1] > local_maxima_price[-2][1] and
                            local_maxima_rsi[-1][1] < local_maxima_rsi[-2][1] and 
                            local_maxima_price[-1][0] == local_maxima_rsi[-1][0] and 
                            local_maxima_price[-2][0] == local_maxima_rsi[-2][0] and
                            local_maxima_price[-1][0] >= price_data.index[-3] and 
                            30 > trend_score_simple.loc[timestamp_input_interval_adj] > 25 and
                            timestamp_input_actual.minute >= 57 and
                            maxima_extrema_diff < 10 and
                            chop_index.loc[timestamp_input_interval_adj, "chop"] < 50)   #adjust so that signal can max be two candles away, prevent delayed trigger mechanism)
        
        #rsi hidden divergence
        price_cond_sell_4 = (local_maxima_price[-1][1] < local_maxima_price[-2][1] and
                            local_maxima_rsi[-1][1] > local_maxima_rsi[-2][1] and
                            local_maxima_price[-1][0] == local_maxima_rsi[-1][0] and
                            local_maxima_price[-2][0] == local_maxima_rsi[-2][0] and
                            local_maxima_price[-1][0] >= price_data.index[-3] and
                            30 > trend_score_simple.loc[timestamp_input_interval_adj] > 25 and
                            timestamp_input_actual.minute >= 57 and
                            maxima_extrema_diff < 10 and
                            chop_index.loc[timestamp_input_interval_adj, "chop"] < 50)   #adjust so that signal can max be two candles away, prevent delayed trigger mechanism) 
        
        price_buy_conditions = [price_cond_buy_1, price_cond_buy_2, price_cond_buy_3, price_cond_buy_4]
        price_sell_conditions = [price_cond_sell_1, price_cond_sell_2, price_cond_sell_3, price_cond_sell_4]

        # Determine the signal and the index of the condition that triggered it
        if any(price_buy_conditions):
            signal = "buy"
            condition_index = price_buy_conditions.index(True)
            if condition_index == 0:
                condition = "cond_buy_1_ma"
            elif condition_index == 1:
                condition = "cond_buy_2_volume"
            elif condition_index == 2:
                condition = "cond_buy_3_divergence"
            elif condition_index == 3:
                condition = "cond_buy_4_hidden_divergence"
            self.logger.info(f"{'#'*1} Buy signal triggered by condition {condition} {'#'*1}")
        elif any(price_sell_conditions):
            signal = "sell"
            condition_index = price_sell_conditions.index(True)
            if condition_index == 0:
                condition = "cond_sell_1_ma"
            elif condition_index == 1:
                condition = "cond_sell_2_volume"
            elif condition_index == 2:
                condition = "cond_sell_3_divergence"
            elif condition_index == 3:
                condition = "cond_sell_4_hidden_divergence"
            self.logger.info(f"{'#'*1} Sell signal triggered by condition {condition} {'#'*1}")
        else:
            signal = "neutral"
            condition_index = None

        strategy_data = pd.DataFrame(index=[timestamp_input_actual], columns=["price","signal","rsi","rsi_7","adx","atr","rel_vol","ema_long","ma_high","ma_low","buy_tm1","sell_tm1"])
        strategy_data.loc[timestamp_input_actual, "price"] = price_data.loc[timestamp_input_interval_adj, "close"]
        strategy_data.loc[timestamp_input_actual, "signal"] = signal
        strategy_data.loc[timestamp_input_actual, "trend_score"] = trend_score_simple.loc[timestamp_input_interval_adj]
        strategy_data.loc[timestamp_input_actual, "condition"] = condition_index +1 if condition_index is not None else None
        strategy_data.loc[timestamp_input_actual, "rsi"] = rsi.loc[timestamp_input_interval_adj, "rsi"]
        strategy_data.loc[timestamp_input_actual, "rsi_7"] = rsi_7.loc[timestamp_input_interval_adj, "rsi"]
        strategy_data.loc[timestamp_input_actual, "adx"] = adx.loc[timestamp_input_interval_adj, "adx"]
        strategy_data.loc[timestamp_input_actual, "atr"] = atr.loc[timestamp_input_interval_adj, "atr"]
        strategy_data.loc[timestamp_input_actual, "rel_vol"] = relative_volume.loc[timestamp_input_interval_adj]
        strategy_data.loc[timestamp_input_actual, "ema_long"] = ema_long.loc[timestamp_input_interval_adj, "ema"]
        strategy_data.loc[timestamp_input_actual, "ma_high"] = ma_high.loc[timestamp_input_interval_adj, "high_ma"]
        strategy_data.loc[timestamp_input_actual, "ma_low"] = ma_low.loc[timestamp_input_interval_adj, "low_ma"]
        strategy_data.loc[timestamp_input_actual, "chop"] = chop_index.loc[timestamp_input_interval_adj, "chop"]
        strategy_data.loc[timestamp_input_actual, "buy_tm1"] = buy_tm1_price
        strategy_data.loc[timestamp_input_actual, "sell_tm1"] = sell_tm1_price
        strategy_data.loc[timestamp_input_actual, "maxima_t"] = pd.Timestamp(local_maxima_price[-1][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "maxima_t1"] = pd.Timestamp(local_maxima_price[-2][0]).strftime('%Y-%m-%d %H:%M:%S')      
        strategy_data.loc[timestamp_input_actual, "maxima_rsi_t"] = pd.Timestamp(local_maxima_rsi[-1][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "maxima_rsi_t1"] = pd.Timestamp(local_maxima_rsi[-2][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "minima_t"] = pd.Timestamp(local_minima_price[-1][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "minima_t1"] = pd.Timestamp(local_minima_price[-2][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "minima_rsi_t"] = pd.Timestamp(local_minima_rsi[-1][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "minima_rsi_t1"] = pd.Timestamp(local_minima_rsi[-2][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "maxima_t_value"] = local_maxima_price[-1][1]
        strategy_data.loc[timestamp_input_actual, "maxima_t1_value"] = local_maxima_price[-2][1]
        strategy_data.loc[timestamp_input_actual, "maxima_rsi_t_value"] = local_maxima_rsi[-1][1]
        strategy_data.loc[timestamp_input_actual, "maxima_rsi_t1_value"] = local_maxima_rsi[-2][1]
        strategy_data.loc[timestamp_input_actual, "minima_t_value"] = local_minima_price[-1][1]
        strategy_data.loc[timestamp_input_actual, "minima_t1_value"] = local_minima_price[-2][1]
        strategy_data.loc[timestamp_input_actual, "minima_rsi_t_value"] = local_minima_rsi[-1][1]
        strategy_data.loc[timestamp_input_actual, "minima_rsi_t1_value"] = local_minima_rsi[-2][1]
        strategy_data.loc[timestamp_input_actual, "condition_1_buy"] = price_cond_buy_1
        strategy_data.loc[timestamp_input_actual, "condition_2_buy"] = price_cond_buy_2
        strategy_data.loc[timestamp_input_actual, "condition_3_buy"] = price_cond_buy_3
        strategy_data.loc[timestamp_input_actual, "condition_4_buy"] = price_cond_buy_4
        strategy_data.loc[timestamp_input_actual, "condition_1_sell"] = price_cond_sell_1
        strategy_data.loc[timestamp_input_actual, "condition_2_sell"] = price_cond_sell_2
        strategy_data.loc[timestamp_input_actual, "condition_3_sell"] = price_cond_sell_3
        strategy_data.loc[timestamp_input_actual, "condition_4_sell"] = price_cond_sell_4

        return signal, atr.loc[timestamp_input_interval_adj, 'atr'], condition_index, strategy_data
    
    def ma_hl_div(self, coin, fiat, interval, trend_interval, timestamp_input_interval_adj, timestamp_input_actual, logger=None, simulate_live_data=False, verbose=False):
        """
        EMA Cross strategy
        #uses tm1 timestamp to check if values crossed on hourly and todays timestamps for multi timeframe strategy

        if trend interval is not specified the function does not use multitimeframe approach
        """

        price_data, minute_data = self.kucoin_price_loader.download_data(coin, fiat, interval, end_date=timestamp_input_actual, 
                                                            use_local_timezone=True, drop_not_complete_timestamps=False, simulate_live_data=simulate_live_data, overwrite_file=False)    #start_date="2017-10-05 00:00:00"  #use timestamp input and t_1 when you want to use uncompleted current timestep values
        
        price_data_long, minute_data_long = self.kucoin_price_loader.download_data(coin, fiat, "1d", end_date=timestamp_input_actual,
                                                                 use_local_timezone=True, drop_not_complete_timestamps=False, simulate_live_data=simulate_live_data, overwrite_file=False)    #start_date="2017-10-05 00:00:00"  #use timestamp input and t_1 when you want to use uncompleted current timestep values
        
        roll_vol = price_data_long['close'].pct_change().rolling(2).std().round(5)
        
        ma_high = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                            timestamp_input=timestamp_input_interval_adj, indicator_name='ma_high', overwrite_file=True, plot=False, ohlc="high",   
                                            high_ma={'span': 20})
        
        ma_low = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ma_low', overwrite_file=True, plot=False, ohlc="low", 
                                        low_ma={'span': 20})
        
        adx = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='adx', overwrite_file=True, plot=False, ohlc="close", 
                                        adx={'di_length': 14})
        
        rsi = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='rsi', overwrite_file=True, plot=False, ohlc="close", 
                                        rsi={'period': 14})

        rsi_7 = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='rsi', overwrite_file=True, plot=False, ohlc="close", 
                                        rsi={'period': 7})

        atr = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='atr', overwrite_file=True, plot=False, ohlc="close", 
                                        atr={'period': 7})

        volume_ma = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ma', overwrite_file=True, plot=False, ohlc="turnover", 
                                        ma={'span': 50})
        
        relative_volume = (price_data['turnover'] / volume_ma["ma"])

        ema_long = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ema', overwrite_file=True, plot=False, ohlc="close", 
                                        ema={'span': 38})
        
        chop_index = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='chop', overwrite_file=True, plot=False, ohlc="close", 
                                        chop={'length': 14})

        # local_minima_price, local_maxima_price  = self.TA_creator.find_local_extrema(price_data["close"])
        local_minima_price, local_maxima_price = self.TA_creator.find_local_extrema_filtered(price_data["close"])

        # local_minima_rsi, local_maxima_rsi = self.TA_creator.find_local_extrema(rsi["rsi"])
        local_minima_rsi, local_maxima_rsi = self.TA_creator.find_local_extrema_filtered(rsi["rsi"])
        minima_extrema_diff = abs(rsi.loc[timestamp_input_interval_adj, "rsi"] - local_minima_rsi[-1][1])
        maxima_extrema_diff = abs(local_maxima_rsi[-1][1] - rsi.loc[timestamp_input_interval_adj, "rsi"])

        trend_score_simple = 0.4 * rsi["rsi"] + 0.3 * adx["adx"] + 0.3 * relative_volume

        ma_low.index = pd.to_datetime(ma_low.index)
        ma_high.index = pd.to_datetime(ma_high.index)
        atr.index = pd.to_datetime(atr.index)
        rsi.index = pd.to_datetime(rsi.index)
        rsi_7.index = pd.to_datetime(rsi_7.index)
        adx.index = pd.to_datetime(adx.index)
        relative_volume.index = pd.to_datetime(relative_volume.index)
        ema_long.index = pd.to_datetime(ema_long.index)
        chop_index.index = pd.to_datetime(chop_index.index)
       
        if timestamp_input_interval_adj not in ma_low.index or timestamp_input_interval_adj not in ma_high.index:
            # self.logger.info(f"Timestamp {timestamp_input} not in DataFrame.")
            current_trend = "neutral"
            return current_trend
        
        last_used_minute_data = minute_data.loc[minute_data['time'] <= timestamp_input_actual, 'time'].iloc[-1]
        minute_data = minute_data.set_index('time') 

        last_available_timestep = price_data.index[-2]
        trend_window_index = price_data.index[-4] 

        buy_tm1_price = min(price_data.loc[last_available_timestep,["open","close"]])
        sell_tm1_price = max(price_data.loc[last_available_timestep,["open","close"]])
        prev_price = price_data.loc[last_available_timestep,'close']

        #check if last 3 prices were above low ema 
        buy_trend_values = [min(price_data.loc[step, ["open","close"]]) for step in price_data[trend_window_index:last_available_timestep].index]
        
        #check if last 3 prices were below high ema
        sell_trend_values = [max(price_data.loc[step, ["open","close"]]) for step in price_data[trend_window_index:last_available_timestep].index]
        
        
        if verbose:
            # self.logger.info(f"{'#'*1} current trend is {current_trend} {'#'*1} with macd {macd.loc[utc_time, 'macd']:.2f} and signal {macd.loc[utc_time, 'signal']:.2f} at time {utc_time}")
            
            
            if simulate_live_data:
                try:
                    
                    self.logger.info(f"{'#'*1} Interval time:{timestamp_input_interval_adj}, interval price: {price_data.loc[timestamp_input_interval_adj, 'close']:.2f} # actual time: {timestamp_input_actual} minute price: {minute_data.loc[last_used_minute_data, 'close']:.2f} # Previous time: {last_available_timestep}, previous price: {prev_price} ")
                except:
                    pass
            else:
                
                self.logger.info(f"{'#'*1} Interval time:{timestamp_input_interval_adj}, interval price: {price_data.loc[timestamp_input_interval_adj, 'close']:.2f} # Previous time: {last_available_timestep}, previous price: {prev_price} #")
            
            self.logger.info(f"{'#'*1} high_ma: {ma_high.loc[timestamp_input_interval_adj, 'high_ma']:.2f} # low_ma: {ma_low.loc[timestamp_input_interval_adj, 'low_ma']:.2f} # adx: {adx.loc[timestamp_input_interval_adj,'adx']} # rsi: {rsi.loc[timestamp_input_interval_adj,'rsi']} # rel. volume: {np.round(relative_volume.loc[timestamp_input_interval_adj],4)} # atr: {atr.loc[timestamp_input_interval_adj,'atr']} # trend sc.: {np.round(trend_score_simple.loc[timestamp_input_interval_adj],3)} # chop: {chop_index.loc[timestamp_input_interval_adj,'chop']} {'#'*1} ")
            
        price_buy_zone = ma_high.loc[timestamp_input_interval_adj, 'high_ma'] * 0.995 
        price_sell_zone = ma_low.loc[timestamp_input_interval_adj, 'low_ma'] * 1.005
        
        
        #strong break through high and low ma
        price_cond_buy_1 =  (price_data.loc[timestamp_input_interval_adj, "close"] < ma_low.loc[timestamp_input_interval_adj, 'low_ma'] and 
                              price_data.loc[timestamp_input_interval_adj, "high"] > ma_low.loc[timestamp_input_interval_adj, 'low_ma'] and
                              np.all(buy_trend_values >= ma_low.loc[last_available_timestep, 'low_ma']) and 
                              rsi.loc[timestamp_input_interval_adj, 'rsi'] > 50 and adx.loc[timestamp_input_interval_adj, 'adx'] > 25  and
                              chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and
                              roll_vol.iloc[-1] > 0.015
                             )     #price under high ma & prev  low above ma
     
        # # ma condition buy
        # price_cond_buy_1 =   (price_data.loc[timestamp_input_interval_adj, "close"] < ma_high.loc[timestamp_input_interval_adj, 'high_ma'] and 
        #                       buy_tm1_price >= ma_high.loc[last_available_timestep, 'high_ma'] and 
        #                       rsi.loc[timestamp_input_interval_adj, 'rsi'] > 50 and 
        #                       price_data.loc[timestamp_input_interval_adj, "close"] > price_buy_zone and 
        #                       adx.loc[timestamp_input_interval_adj, 'adx'] > 25)   #   and  #price under high ma & prev  low above ma
        
        #rsi divergence
        price_cond_buy_2 = (local_minima_price[-1][1] < local_minima_price[-2][1] and 
                            local_minima_rsi[-1][1] > local_minima_rsi[-2][1] and
                            local_minima_price[-1][0] == local_minima_rsi[-1][0] and
                            local_minima_price[-2][0] == local_minima_rsi[-2][0] and
                            local_minima_price[-1][0] >= price_data.index[-3] and 
                            35 > trend_score_simple.loc[timestamp_input_interval_adj] > 25 and
                            timestamp_input_actual.minute >= 55 and
                            minima_extrema_diff < 10 and 
                            chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and
                            roll_vol.iloc[-1] > 0.015)  #adjust so that signal can max be two candles away, prevent delayed trigger mechanism
        
        #rsi hidden divergence
        price_cond_buy_3 = (local_minima_price[-1][1] > local_minima_price[-2][1] and 
                            local_minima_rsi[-1][1] < local_minima_rsi[-2][1] and
                            local_minima_price[-1][0] == local_minima_rsi[-1][0] and
                            local_minima_price[-2][0] == local_minima_rsi[-2][0] and
                            local_minima_price[-1][0] >= price_data.index[-3] and
                            35 > trend_score_simple.loc[timestamp_input_interval_adj] > 25 and
                            timestamp_input_actual.minute >= 55 and
                            minima_extrema_diff < 10 and   #adjust so that signal can max be two candles away, prevent delayed trigger mechanism)
                            chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and
                            roll_vol.iloc[-1] > 0.015)

        #ma condition sell 
        price_cond_sell_1 = (price_data.loc[timestamp_input_interval_adj, "close"] > ma_high.loc[timestamp_input_interval_adj, 'high_ma'] and 
                             price_data.loc[timestamp_input_interval_adj, "low"] < ma_high.loc[timestamp_input_interval_adj, 'high_ma'] and
                             np.all(sell_trend_values <= ma_high.loc[last_available_timestep, 'high_ma']) and   #price above low ma & prev high below ma 
                             rsi.loc[timestamp_input_interval_adj, 'rsi'] < 50 and adx.loc[timestamp_input_interval_adj, 'adx'] > 25 and
                             chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and
                             roll_vol.iloc[-1] > 0.015) 
        
        # #ma condition sell
        # price_cond_sell_1 = (price_data.loc[timestamp_input_interval_adj, "close"] > ma_low.loc[timestamp_input_interval_adj, 'low_ma'] and 
        #                      sell_tm1_price <= ma_low.loc[last_available_timestep, 'low_ma'] and
        #                      rsi.loc[timestamp_input_interval_adj, 'rsi'] < 50 and 
        #                      price_data.loc[timestamp_input_interval_adj, "close"] < price_sell_zone and
        #                      adx.loc[timestamp_input_interval_adj, 'adx'] > 25)  
       
        #rsi divergence
        price_cond_sell_2 = (local_maxima_price[-1][1] > local_maxima_price[-2][1] and
                            local_maxima_rsi[-1][1] < local_maxima_rsi[-2][1] and 
                            local_maxima_price[-1][0] == local_maxima_rsi[-1][0] and 
                            local_maxima_price[-2][0] == local_maxima_rsi[-2][0] and
                            local_maxima_price[-1][0] >= price_data.index[-3] and 
                            35 > trend_score_simple.loc[timestamp_input_interval_adj] > 25 and
                            timestamp_input_actual.minute >= 55 and   #adjust so that signal can max be two candles away, prevent delayed trigger mechanism)
                            maxima_extrema_diff < 10 and  
                            chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and
                            roll_vol.iloc[-1] > 0.015)  
        
        #rsi hidden divergence
        price_cond_sell_3 = (local_maxima_price[-1][1] < local_maxima_price[-2][1] and
                            local_maxima_rsi[-1][1] > local_maxima_rsi[-2][1] and
                            local_maxima_price[-1][0] == local_maxima_rsi[-1][0] and
                            local_maxima_price[-2][0] == local_maxima_rsi[-2][0] and
                            local_maxima_price[-1][0] >= price_data.index[-3] and
                            35 > trend_score_simple.loc[timestamp_input_interval_adj] > 25 and
                            timestamp_input_actual.minute >= 55 and
                            maxima_extrema_diff < 10 and
                            chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and
                            roll_vol.iloc[-1] > 0.015)   #adjust so that signal can max be two candles away, prevent delayed trigger mechanism) 
        
        price_buy_conditions = [price_cond_buy_1, price_cond_buy_2, price_cond_buy_3]
        price_sell_conditions = [price_cond_sell_1, price_cond_sell_2, price_cond_sell_3]

        # Determine the signal and the index of the condition that triggered it
        if any(price_buy_conditions):
            signal = "buy"
            condition_index = price_buy_conditions.index(True)
            if condition_index == 0:
                condition = "cond_buy_1_ma"
            elif condition_index == 1:
                condition = "cond_buy_2_divergence"
            elif condition_index == 2:
                condition = "cond_buy_3_hidden_divergence"
            self.logger.info(f"{'#'*1} Buy signal triggered by condition {condition} {'#'*1}")
        elif any(price_sell_conditions):
            signal = "sell"
            condition_index = price_sell_conditions.index(True)
            if condition_index == 0:
                condition = "cond_sell_1_ma"
            elif condition_index == 1:
                condition = "cond_sell_2_divergence"
            elif condition_index == 2:
                condition = "cond_sell_3_hidden_divergence"
            self.logger.info(f"{'#'*1} Sell signal triggered by condition {condition} {'#'*1}")
        else:
            signal = "neutral"
            condition_index = None

        strategy_data = pd.DataFrame(index=[timestamp_input_actual], columns=["price","signal","rsi","rsi_7","adx","atr","rel_vol","ema_long","ma_high","ma_low","buy_tm1","sell_tm1"])
        strategy_data.loc[timestamp_input_actual, "price"] = price_data.loc[timestamp_input_interval_adj, "close"]
        strategy_data.loc[timestamp_input_actual, "signal"] = signal
        strategy_data.loc[timestamp_input_actual, "trend_score"] = trend_score_simple.loc[timestamp_input_interval_adj]
        strategy_data.loc[timestamp_input_actual, "condition"] = condition_index +1 if condition_index is not None else None
        strategy_data.loc[timestamp_input_actual, "long_close"] = price_data_long.loc[timestamp_input_interval_adj.replace(hour=0, minute=0, second=0), "close"]
        strategy_data.loc[timestamp_input_actual, "roll_vol"] = roll_vol.loc[timestamp_input_interval_adj.replace(hour=0, minute=0, second=0)]
        strategy_data.loc[timestamp_input_actual, "rsi"] = rsi.loc[timestamp_input_interval_adj, "rsi"]
        strategy_data.loc[timestamp_input_actual, "rsi_7"] = rsi_7.loc[timestamp_input_interval_adj, "rsi"]
        strategy_data.loc[timestamp_input_actual, "adx"] = adx.loc[timestamp_input_interval_adj, "adx"]
        strategy_data.loc[timestamp_input_actual, "atr"] = atr.loc[timestamp_input_interval_adj, "atr"]
        strategy_data.loc[timestamp_input_actual, "rel_vol"] = relative_volume.loc[timestamp_input_interval_adj]
        strategy_data.loc[timestamp_input_actual, "ema_long"] = ema_long.loc[timestamp_input_interval_adj, "ema"]
        strategy_data.loc[timestamp_input_actual, "ma_high"] = ma_high.loc[timestamp_input_interval_adj, "high_ma"]
        strategy_data.loc[timestamp_input_actual, "ma_low"] = ma_low.loc[timestamp_input_interval_adj, "low_ma"]
        strategy_data.loc[timestamp_input_actual, "chop"] = chop_index.loc[timestamp_input_interval_adj, "chop"]
        strategy_data.loc[timestamp_input_actual, "buy_tm1"] = buy_tm1_price
        strategy_data.loc[timestamp_input_actual, "sell_tm1"] = sell_tm1_price
        strategy_data.loc[timestamp_input_actual, "maxima_t"] = pd.Timestamp(local_maxima_price[-1][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "maxima_t1"] = pd.Timestamp(local_maxima_price[-2][0]).strftime('%Y-%m-%d %H:%M:%S')      
        strategy_data.loc[timestamp_input_actual, "maxima_rsi_t"] = pd.Timestamp(local_maxima_rsi[-1][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "maxima_rsi_t1"] = pd.Timestamp(local_maxima_rsi[-2][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "minima_t"] = pd.Timestamp(local_minima_price[-1][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "minima_t1"] = pd.Timestamp(local_minima_price[-2][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "minima_rsi_t"] = pd.Timestamp(local_minima_rsi[-1][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "minima_rsi_t1"] = pd.Timestamp(local_minima_rsi[-2][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "maxima_t_value"] = local_maxima_price[-1][1]
        strategy_data.loc[timestamp_input_actual, "maxima_t1_value"] = local_maxima_price[-2][1]
        strategy_data.loc[timestamp_input_actual, "maxima_rsi_t_value"] = local_maxima_rsi[-1][1]
        strategy_data.loc[timestamp_input_actual, "maxima_rsi_t1_value"] = local_maxima_rsi[-2][1]
        strategy_data.loc[timestamp_input_actual, "minima_t_value"] = local_minima_price[-1][1]
        strategy_data.loc[timestamp_input_actual, "minima_t1_value"] = local_minima_price[-2][1]
        strategy_data.loc[timestamp_input_actual, "minima_rsi_t_value"] = local_minima_rsi[-1][1]
        strategy_data.loc[timestamp_input_actual, "minima_rsi_t1_value"] = local_minima_rsi[-2][1]
        strategy_data.loc[timestamp_input_actual, "condition_1_buy"] = price_cond_buy_1
        strategy_data.loc[timestamp_input_actual, "condition_2_buy"] = price_cond_buy_2
        strategy_data.loc[timestamp_input_actual, "condition_3_buy"] = price_cond_buy_3
        strategy_data.loc[timestamp_input_actual, "condition_1_sell"] = price_cond_sell_1
        strategy_data.loc[timestamp_input_actual, "condition_2_sell"] = price_cond_sell_2
        strategy_data.loc[timestamp_input_actual, "condition_3_sell"] = price_cond_sell_3

        return signal, atr.loc[timestamp_input_interval_adj, 'atr'], condition_index, strategy_data


    def support_resistance(self, coin, fiat, interval, trend_interval, timestamp_input_interval_adj, timestamp_input_actual, logger=None, simulate_live_data=False, verbose=False):
        """
        EMA Cross strategy
        #uses tm1 timestamp to check if values crossed on hourly and todays timestamps for multi timeframe strategy

        if trend interval is not specified the function does not use multitimeframe approach
        """

        price_data, minute_data = self.kucoin_price_loader.download_data(coin, fiat, interval, end_date=timestamp_input_actual, 
                                                            use_local_timezone=True, drop_not_complete_timestamps=False, simulate_live_data=simulate_live_data, overwrite_file=False)    #start_date="2017-10-05 00:00:00"  #use timestamp input and t_1 when you want to use uncompleted current timestep values
        
    
        rsi = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='rsi', overwrite_file=True, plot=False, ohlc="close", 
                                        rsi={'period': 14})

        atr = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='atr', overwrite_file=True, plot=False, ohlc="close", 
                                        atr={'period': 7})

        
        ema_long = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ema', overwrite_file=True, plot=False, ohlc="close", 
                                        ema={'span': 50})

        # local_minima_price, local_maxima_price  = self.TA_creator.find_local_extrema(price_data["close"])
        local_minima_price, local_maxima_price = self.TA_creator.find_local_extrema_filtered(price_data["close"])

        # local_minima_rsi, local_maxima_rsi = self.TA_creator.find_local_extrema(rsi["rsi"])
        local_minima_rsi, local_maxima_rsi = self.TA_creator.find_local_extrema_filtered(rsi["rsi"])

        for i in range(-1, -5,-1):
            if local_minima_price[i][0] == local_minima_rsi[i][0]:
                support_t = local_minima_price[i][0]
                support = price_data.loc[support_t,"low"]
                current_local_minima_price = local_minima_price[i][0]
                current_local_minima_rsi = local_minima_rsi[i][0]
            if local_maxima_price[i][0] == local_maxima_rsi[i][0]:
                resistance_t = local_maxima_price[i][0]
                resistance = price_data.loc[resistance_t,"high"]
                current_local_maxima_price = local_maxima_price[i][0]
                current_local_maxima_rsi = local_maxima_rsi[i][0]

        atr.index = pd.to_datetime(atr.index)
        rsi.index = pd.to_datetime(rsi.index)
        ema_long.index = pd.to_datetime(ema_long.index)
       
        if timestamp_input_interval_adj not in rsi.index or timestamp_input_interval_adj not in rsi.index:
            # self.logger.info(f"Timestamp {timestamp_input} not in DataFrame.")
            current_trend = "neutral"
            return current_trend
        
        last_used_minute_data = minute_data.loc[minute_data['time'] <= timestamp_input_actual, 'time'].iloc[-1]
        minute_data = minute_data.set_index('time') 

        last_available_timestep = price_data.index[-2]
        prev_price = price_data.loc[last_available_timestep,'close']

        if verbose:
            # self.logger.info(f"{'#'*1} current trend is {current_trend} {'#'*1} with macd {macd.loc[utc_time, 'macd']:.2f} and signal {macd.loc[utc_time, 'signal']:.2f} at time {utc_time}")
            
            
            if simulate_live_data:
                try:
                    
                    self.logger.info(f"{'#'*1} Interval time:{timestamp_input_interval_adj}, interval price: {price_data.loc[timestamp_input_interval_adj, 'close']:.2f} # actual time: {timestamp_input_actual} minute price: {minute_data.loc[last_used_minute_data, 'close']:.2f} # Previous time: {last_available_timestep}, previous price: {prev_price} ")
                except:
                    pass
            else:
                
                self.logger.info(f"{'#'*1} Interval time:{timestamp_input_interval_adj}, interval price: {price_data.loc[timestamp_input_interval_adj, 'close']:.2f} # Previous time: {last_available_timestep}, previous price: {prev_price} #")
            
            self.logger.info(f"{'#'*1}  # rsi: {rsi.loc[timestamp_input_interval_adj,'rsi']} # atr: {atr.loc[timestamp_input_interval_adj,'atr']} # local minima price: {local_maxima_price} # local maxima price: {local_maxima_price} #  resistance: {resistance}  # support: {support} {'#'*1} ")
    
        price_cond_buy_1 = price_data.loc[timestamp_input_interval_adj, "close"] > resistance and rsi.loc[timestamp_input_interval_adj, 'rsi'] > 50
        price_cond_sell_1 = price_data.loc[timestamp_input_interval_adj, "close"] < support and rsi.loc[timestamp_input_interval_adj, 'rsi'] < 50
        
        price_buy_conditions = [price_cond_buy_1]
        price_sell_conditions = [price_cond_sell_1]

        # Determine the signal and the index of the condition that triggered it
        if any(price_buy_conditions):
            signal = "buy"
            condition_index = price_buy_conditions.index(True)
            if condition_index == 0:
                condition = "resistance_break"

            self.logger.info(f"{'#'*1} Buy signal triggered by condition {condition} {'#'*1}")
        elif any(price_sell_conditions):
            signal = "sell"
            condition_index = price_sell_conditions.index(True)
            if condition_index == 0:
                condition = "support_break"
            
            self.logger.info(f"{'#'*1} Sell signal triggered by condition {condition} {'#'*1}")
        else:
            signal = "neutral"
            condition_index = None

        strategy_data = pd.DataFrame(index=[timestamp_input_actual], columns=["price","signal","rsi","atr","ema_long","maxima_t","maxima_t1","maxima_rsi_t","maxima_rsi_t1","minima_t","minima_t1","minima_rsi_t","minima_rsi_t1","maxima_t_value","maxima_t1_value","maxima_rsi_t_value","maxima_rsi_t1_value","minima_t_value","minima_t1_value","minima_rsi_t_value","minima_rsi_t1_value","condition_1_buy","condition_1_sell"])
        strategy_data.loc[timestamp_input_actual, "price"] = price_data.loc[timestamp_input_interval_adj, "close"]
        strategy_data.loc[timestamp_input_actual, "signal"] = signal
        strategy_data.loc[timestamp_input_actual, "rsi"] = rsi.loc[timestamp_input_interval_adj, "rsi"]
        strategy_data.loc[timestamp_input_actual, "atr"] = atr.loc[timestamp_input_interval_adj, "atr"]
        strategy_data.loc[timestamp_input_actual, "ema_long"] = ema_long.loc[timestamp_input_interval_adj, "ema"]

        strategy_data.loc[timestamp_input_actual, "maxima_t"] = pd.Timestamp(local_maxima_price[-1][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "maxima_t1"] = pd.Timestamp(local_maxima_price[-2][0]).strftime('%Y-%m-%d %H:%M:%S')      
        strategy_data.loc[timestamp_input_actual, "maxima_rsi_t"] = pd.Timestamp(local_maxima_rsi[-1][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "maxima_rsi_t1"] = pd.Timestamp(local_maxima_rsi[-2][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "minima_t"] = pd.Timestamp(local_minima_price[-1][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "minima_t1"] = pd.Timestamp(local_minima_price[-2][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "minima_rsi_t"] = pd.Timestamp(local_minima_rsi[-1][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "minima_rsi_t1"] = pd.Timestamp(local_minima_rsi[-2][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "maxima_t_value"] = local_maxima_price[-1][1]
        strategy_data.loc[timestamp_input_actual, "maxima_t1_value"] = local_maxima_price[-2][1]
        strategy_data.loc[timestamp_input_actual, "maxima_rsi_t_value"] = local_maxima_rsi[-1][1]
        strategy_data.loc[timestamp_input_actual, "maxima_rsi_t1_value"] = local_maxima_rsi[-2][1]
        strategy_data.loc[timestamp_input_actual, "minima_t_value"] = local_minima_price[-1][1]
        strategy_data.loc[timestamp_input_actual, "minima_t1_value"] = local_minima_price[-2][1]
        strategy_data.loc[timestamp_input_actual, "minima_rsi_t_value"] = local_minima_rsi[-1][1]
        strategy_data.loc[timestamp_input_actual, "minima_rsi_t1_value"] = local_minima_rsi[-2][1]
        strategy_data.loc[timestamp_input_actual, "condition_1_buy"] = price_cond_buy_1
        strategy_data.loc[timestamp_input_actual, "condition_1_sell"] = price_cond_sell_1


        return signal, atr.loc[timestamp_input_interval_adj, 'atr'], condition_index, strategy_data
    
    def kama_boll(self, coin, fiat, interval, timestamp_input_interval_adj, timestamp_input_actual, logger=None, simulate_live_data=False, verbose=False):
        """
        EMA Cross strategy
        #uses tm1 timestamp to check if values crossed on hourly and todays timestamps for multi timeframe strategy

        if trend interval is not specified the function does not use multitimeframe approach
        """

        price_data, minute_data = self.kucoin_price_loader.download_data(coin, fiat, interval, end_date=timestamp_input_actual, 
                                                            use_local_timezone=True, drop_not_complete_timestamps=False, simulate_live_data=simulate_live_data, overwrite_file=False)    #start_date="2017-10-05 00:00:00"  #use timestamp input and t_1 when you want to use uncompleted current timestep values
        
        
        adx = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='adx', overwrite_file=True, plot=False, ohlc="close", 
                                        adx={'di_length': 14})
    
        atr = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='atr', overwrite_file=True, plot=False, ohlc="close", 
                                        atr={'period': 7})

        chop_index = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ma', overwrite_file=True, plot=False, ohlc="close", 
                                        chop={'length': 14})
        
        kama_short = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='kama_short', overwrite_file=True, plot=False, ohlc="close", 
                                        kama={'length': 14})
        
        kama_long = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='kama_long', overwrite_file=True, plot=False, ohlc="close", 
                                        kama={'length': 100})
        
        bollinger = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='bollinger', overwrite_file=True, plot=False, ohlc="close", 
                                        bollinger={'window': 20, 'num_std_dev': 2})
      
        bollinger_width_band = bollinger["bb_bandwith"]

    
        atr.index = pd.to_datetime(atr.index)
        adx.index = pd.to_datetime(adx.index)
        chop_index.index = pd.to_datetime(chop_index.index) 
        kama_short.index = pd.to_datetime(kama_short.index)
        kama_long.index = pd.to_datetime(kama_long.index)
        bollinger_width_band.index = pd.to_datetime(bollinger_width_band.index)
       
        if timestamp_input_interval_adj not in kama_short.index or timestamp_input_interval_adj not in kama_short.index:
            # self.logger.info(f"Timestamp {timestamp_input} not in DataFrame.")
            current_trend = "neutral"
            return current_trend
        
        last_used_minute_data = minute_data.loc[minute_data['time'] <= timestamp_input_actual, 'time'].iloc[-1]
        minute_data = minute_data.set_index('time') 

        last_available_timestep = price_data.index[-2]
        trend_window_index = price_data.index[-4] 

        prev_price = price_data.loc[last_available_timestep,'close']

        #check if last 3 prices were above low ema 
        buy_trend_values = [min(price_data.loc[step, ["open","close"]]) for step in price_data[trend_window_index:last_available_timestep].index]
        
        #check if last 3 prices were below high ema
        sell_trend_values = [max(price_data.loc[step, ["open","close"]]) for step in price_data[trend_window_index:last_available_timestep].index]
        
        
        if verbose:
            if simulate_live_data:
                try:
                    self.logger.info(f"{'#'*1}          Interval time:{timestamp_input_interval_adj},          interval price: {price_data.loc[timestamp_input_interval_adj, 'close']:.2f}          # actual time: {timestamp_input_actual}          minute price: {minute_data.loc[last_used_minute_data, 'close']:.2f}          # Previous time: {last_available_timestep}, previous price: {prev_price} ")
                except:
                    pass
            else:
                self.logger.info(f"{'#'*1} Interval time:{timestamp_input_interval_adj}, interval price: {price_data.loc[timestamp_input_interval_adj, 'close']:.2f} # Previous time: {last_available_timestep}, previous price: {prev_price} #")
            
            self.logger.info(f"{'#'*1} # adx: {adx.loc[timestamp_input_interval_adj,'adx']} # atr: {atr.loc[timestamp_input_interval_adj,'atr']} # kama short: {kama_short.loc[timestamp_input_interval_adj,'kama']} # kama long: {kama_long.loc[timestamp_input_interval_adj,'kama']} #  bb_width: {bollinger_width_band.loc[timestamp_input_interval_adj]}  # chop: {chop_index.loc[timestamp_input_interval_adj,'chop']}  {'#'*1} ")
    
        
        # #strong break through high and low ma
        # price_cond_buy_1 =  (((price_data.loc[timestamp_input_interval_adj, "close"] > kama_short.loc[timestamp_input_interval_adj, 'kama'] and price_data.loc[timestamp_input_interval_adj, "low"] < kama_short.loc[timestamp_input_interval_adj, 'kama'] and timestamp_input_actual.minute >= 57) or 
        #                      (price_data.loc[timestamp_input_interval_adj, "close"] > kama_long.loc[timestamp_input_interval_adj, 'kama'] and price_data.loc[timestamp_input_interval_adj, "low"] < kama_long.loc[timestamp_input_interval_adj, 'kama'] and timestamp_input_actual.minute >= 57)) and
        #                     adx.loc[timestamp_input_interval_adj, 'adx'] > 25 and
        #                     chop_index.loc[timestamp_input_interval_adj, 'chop'] < 50 and
        #                     bollinger_width_band.loc[timestamp_input_interval_adj] < 8 )     
    
        # #ma condition sell 
        # price_cond_sell_1 = (((price_data.loc[timestamp_input_interval_adj, "close"] < kama_short.loc[timestamp_input_interval_adj, 'kama'] and price_data.loc[timestamp_input_interval_adj, "high"] > kama_short.loc[timestamp_input_interval_adj, 'kama'] and timestamp_input_actual.minute >= 57) or
        #                      (price_data.loc[timestamp_input_interval_adj, "high"] > kama_long.loc[timestamp_input_interval_adj, 'kama'] and price_data.loc[timestamp_input_interval_adj, "close"] < kama_long.loc[timestamp_input_interval_adj, 'kama'] and timestamp_input_actual.minute >= 57)) and
        #                      adx.loc[timestamp_input_interval_adj, 'adx'] > 25 and
        #                      chop_index.loc[timestamp_input_interval_adj, 'chop'] < 50 and
        #                      bollinger_width_band.loc[timestamp_input_interval_adj] <8)
        

         #strong break through high and low ma
        price_cond_buy_1 =  (price_data.loc[timestamp_input_interval_adj, "close"] > kama_short.loc[timestamp_input_interval_adj, 'kama'] and 
                             price_data.loc[timestamp_input_interval_adj, "low"] < kama_short.loc[timestamp_input_interval_adj, 'kama'] and 
                             price_data.loc[timestamp_input_interval_adj, "close"] > kama_long.loc[timestamp_input_interval_adj, 'kama'] and
                             timestamp_input_actual.minute >= 57 and
                            adx.loc[timestamp_input_interval_adj, 'adx'] > 25 and
                            chop_index.loc[timestamp_input_interval_adj, 'chop'] < 50 and
                            bollinger_width_band.loc[timestamp_input_interval_adj] < 8 )     
    
        #ma condition sell 
        price_cond_sell_1 = (price_data.loc[timestamp_input_interval_adj, "close"] < kama_short.loc[timestamp_input_interval_adj, 'kama'] and 
                             price_data.loc[timestamp_input_interval_adj, "high"] > kama_short.loc[timestamp_input_interval_adj, 'kama'] and 
                             price_data.loc[timestamp_input_interval_adj, "close"] < kama_long.loc[timestamp_input_interval_adj, 'kama'] and
                             timestamp_input_actual.minute >= 57 and 
                             adx.loc[timestamp_input_interval_adj, 'adx'] > 25 and
                             chop_index.loc[timestamp_input_interval_adj, 'chop'] < 50 and
                             bollinger_width_band.loc[timestamp_input_interval_adj] <8)
        
    
      
        price_buy_conditions = [price_cond_buy_1]
        price_sell_conditions = [price_cond_sell_1]

        # Determine the signal and the index of the condition that triggered it
        if any(price_buy_conditions):
            signal = "buy"
            condition_index = price_buy_conditions.index(True)
            if condition_index == 0:
                condition = "cond_buy_1"
            self.logger.info(f"{'#'*1} Buy signal triggered by condition {condition} {'#'*1}")
        elif any(price_sell_conditions):
            signal = "sell"
            condition_index = price_sell_conditions.index(True)
            if condition_index == 0:
                condition = "cond_sell_1"
            self.logger.info(f"{'#'*1} Sell signal triggered by condition {condition} {'#'*1}")
        else:
            signal = "neutral"
            condition_index = None

        strategy_data = pd.DataFrame(index=[timestamp_input_actual], columns=["price","signal","condition"])
        strategy_data.loc[timestamp_input_actual, "price"] = price_data.loc[timestamp_input_interval_adj, "close"]
        strategy_data.loc[timestamp_input_actual, "signal"] = signal
        strategy_data.loc[timestamp_input_actual, "condition"] = condition_index +1 if condition_index is not None else None
        strategy_data.loc[timestamp_input_actual, "adx"] = adx.loc[timestamp_input_interval_adj, "adx"]
        strategy_data.loc[timestamp_input_actual, "atr"] = atr.loc[timestamp_input_interval_adj, "atr"]
        strategy_data.loc[timestamp_input_actual, "kama_short"] = kama_short.loc[timestamp_input_interval_adj, "kama"]
        strategy_data.loc[timestamp_input_actual, "kama_long"] = kama_long.loc[timestamp_input_interval_adj, "kama"]
        strategy_data.loc[timestamp_input_actual, "chop"] = chop_index.loc[timestamp_input_interval_adj, "chop"] 
        strategy_data.loc[timestamp_input_actual, "bollinger_width_band"] = bollinger_width_band.loc[timestamp_input_interval_adj]
        strategy_data.loc[timestamp_input_actual, "condition_1_buy"] = price_cond_buy_1
        strategy_data.loc[timestamp_input_actual, "condition_1_sell"] = price_cond_sell_1

        return signal, atr.loc[timestamp_input_interval_adj, 'atr'], condition_index, strategy_data


    def candle_pattern(self, coin, fiat, interval, timestamp_input_interval_adj, timestamp_input_actual, logger=None, simulate_live_data=False, verbose=False):
        """
        EMA Cross strategy
        #uses tm1 timestamp to check if values crossed on hourly and todays timestamps for multi timeframe strategy

        if trend interval is not specified the function does not use multitimeframe approach
        """

        price_data, minute_data = self.kucoin_price_loader.download_data(coin, fiat, interval, end_date=timestamp_input_actual, 
                                                            use_local_timezone=True, drop_not_complete_timestamps=False, simulate_live_data=simulate_live_data, overwrite_file=False)    #start_date="2017-10-05 00:00:00"  #use timestamp input and t_1 when you want to use uncompleted current timestep values
        
        rsi = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='rsi', overwrite_file=True, plot=False, ohlc="close", 
                                        rsi={'period': 10})

        atr = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='atr', overwrite_file=True, plot=False, ohlc="close", 
                                        atr={'period': 7})

        ema_short = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ema', overwrite_file=True, plot=False, ohlc="close", 
                                        ema={'span': 5})

        adx = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='adx', overwrite_file=True, plot=False, ohlc="close", 
                                        adx={'di_length': 14})

        volume_ma = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ma', overwrite_file=True, plot=False, ohlc="turnover", 
                                        ma={'span': 50})
        
        chop_index = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='chop', overwrite_file=True, plot=False, ohlc="close", 
                                        chop={'length': 14})
        

        relative_volume = np.round((price_data['turnover'] / volume_ma["ma"]),4)

        cdl_patterns = ["morningstardoji", "eveningstardoji", "3whitesoldiers","3blackcrows", "engulfing_bullish", "engulfing_bearish", "flag_bullish", "flag_bearish", "inside_candle", "retest_bullish", "retest_bearish"]
        
        cdl_patterns_results = self.TA_creator.cdl_pattern(price_data, cdl_patterns, offset=1)
                                
        if timestamp_input_interval_adj not in price_data.index:
            # self.logger.info(f"Timestamp {timestamp_input} not in DataFrame.")
            current_trend = "neutral"
            return current_trend
        
        last_used_minute_data = minute_data.loc[minute_data['time'] <= timestamp_input_actual, 'time'].iloc[-1]
        minute_data = minute_data.set_index('time') 

        tm1 = price_data.index[-2]
        prev_price = price_data.loc[tm1,'close']

        tm2 = price_data.index[-3]
        tm3 = price_data.index[-4]
        tm4 = price_data.index[-5]
        
        rsi_pct_change  = rsi.loc[:,"rsi"].pct_change()
        rsi_3candle_mean = rsi.loc[:,"rsi"].pct_change().rolling(2).mean()

        if verbose:
            if simulate_live_data:
                try:
                    self.logger.info(f"{'#'*1}          Interval time:{timestamp_input_interval_adj},          interval price: {price_data.loc[timestamp_input_interval_adj, 'close']:.2f}          # actual time: {timestamp_input_actual}          minute price: {minute_data.loc[last_used_minute_data, 'close']:.2f}          # Previous time: {tm1}, previous price: {prev_price} ")
                except:
                    pass
            else:
                self.logger.info(f"{'#'*1} Interval time:{timestamp_input_interval_adj}, interval price: {price_data.loc[timestamp_input_interval_adj, 'close']:.2f} # Previous time: {tm1}, previous price: {prev_price} #")
            
            self.logger.info(f"{'#'*1}  # rsi: {rsi.loc[timestamp_input_interval_adj,'rsi']} # atr: {atr.loc[timestamp_input_interval_adj,'atr']}# adx: {adx.loc[timestamp_input_interval_adj,'adx']} # volume_ma: {volume_ma.loc[timestamp_input_interval_adj,'ma']} # relative_volume: {relative_volume.loc[timestamp_input_interval_adj]} {'#'*1} # chop index: {chop_index.loc[timestamp_input_interval_adj,'chop']} ")    #ema long {ema_long.loc[timestamp_input_interval_adj,'ema']} 
    
        
        #buy signal conditions

        # price_cond_buy_1 = cdl_patterns_results["special_engulfing_bullish"] and rsi.loc[timestamp_input_interval_adj, 'rsi'] > 50

        # price_cond_buy_1 = (cdl_patterns_results["retest_bullish"] and
        #                     adx.loc[timestamp_input_interval_adj, 'adx'] > 25 and
        #                     chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and
        #                     80 > rsi.loc[timestamp_input_interval_adj, 'rsi'] > 50 and
        #                     ((rsi.loc[timestamp_input_interval_adj, 'rsi'] / rsi.loc[tm1, 'rsi'])-1) > 0 and
        #                     20 >= timestamp_input_actual.minute >= 5)

        price_cond_buy_1 = (cdl_patterns_results["morningstardoji"] and
                            (adx.loc[timestamp_input_interval_adj, 'adx'] > 25 or relative_volume.loc[tm1] > 1.5) and
                            rsi_pct_change.loc[timestamp_input_interval_adj] > 0 and 
                            20 >= timestamp_input_actual.minute)  # relative_volume.loc[tm1] > 1.5 <-- with rel. vol all trades were correct  #and rsi.loc[timestamp_input_interval_adj, 'rsi'] < 30   20 >= timestamp_input_actual.minute >= 5

        price_cond_buy_2 = (cdl_patterns_results["3whitesoldiers"] and
                            adx.loc[timestamp_input_interval_adj, 'adx'] > 25 and
                            chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and
                            rsi_pct_change.loc[timestamp_input_interval_adj] > 0 and
                            rsi.loc[timestamp_input_interval_adj,"rsi"] < 80 and
                            20 >= timestamp_input_actual.minute >= 10)   #and relative_volume.loc[tm1] > 1.5 

        price_cond_buy_3 = (cdl_patterns_results["flag_bullish"] and
                            adx.loc[timestamp_input_interval_adj, 'adx'] > 25 and
                            chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and
                            relative_volume.loc[tm4] > 1.5 and
                            rsi_pct_change.loc[timestamp_input_interval_adj] > 0 and 
                            20 >= timestamp_input_actual.minute >= 5)

        # price_cond_buy_4 = (cdl_patterns_results["inside_candle"] and 
        #                     adx.loc[timestamp_input_interval_adj, 'adx'] > 25 and
        #                     chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and
        #                     relative_volume.loc[tm2] > 1.5 and
        #                     75 > rsi.loc[timestamp_input_interval_adj,"rsi"] > 50 and
        #                     rsi_pct_change.loc[timestamp_input_interval_adj] > 0 and
        #                     25 >= timestamp_input_actual.minute >= 13)

        price_cond_buy_4 = (cdl_patterns_results["engulfing_bullish"] and
                            adx.loc[timestamp_input_interval_adj, 'adx'] > 25 and
                            chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and
                            relative_volume.loc[tm1] > 1.8 and
                            rsi.loc[timestamp_input_interval_adj,"rsi"] < 80 and
                            rsi_pct_change.loc[timestamp_input_interval_adj] > 0 and
                            25 >= timestamp_input_actual.minute ) #and rsi.loc[timestamp_input_interval_adj, 'rsi'] < 30 
        
        #volume indicator
        price_cond_buy_5 = (# rsi.loc[timestamp_input_interval_adj,"rsi"] > 50 and 
                            rsi.loc[timestamp_input_interval_adj,"rsi"] < 80 and 
                            rsi_3candle_mean.loc[timestamp_input_interval_adj] > 0 and
                            price_data.loc[timestamp_input_interval_adj, "close"] >  price_data.loc[timestamp_input_interval_adj, "open"] and
                            relative_volume.loc[timestamp_input_interval_adj] > 2.5 and
                            chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and 
                            price_data.loc[timestamp_input_interval_adj, "close"] > ema_short.loc[timestamp_input_interval_adj,"ema"]) # and timestamp_input_actual.minute >= 10)  adx.loc[timestamp_input_interval_adj, 'adx'] > 25 and
                            

        #sell signal conditions

        # price_cond_sell_1 = cdl_patterns_results["special_engulfing_bearish"] and rsi.loc[timestamp_input_interval_adj, 'rsi'] < 50
        
        # price_cond_sell_1 = (cdl_patterns_results["retest_bearish"] and 
        #                     adx.loc[timestamp_input_interval_adj, 'adx'] > 25  and 
        #                     chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and
        #                     50 > rsi.loc[timestamp_input_interval_adj, 'rsi'] > 20 and
        #                     ((rsi.loc[timestamp_input_interval_adj, 'rsi'] / rsi.loc[tm1, 'rsi'])-1) < 0 and
        #                     20 >= timestamp_input_actual.minute >= 5)

        price_cond_sell_1 = (cdl_patterns_results["eveningstardoji"] and 
                            (adx.loc[timestamp_input_interval_adj, 'adx'] > 25 or relative_volume.loc[tm1] > 1.5) and 
                            rsi_pct_change.loc[timestamp_input_interval_adj] < 0 and
                            20 >= timestamp_input_actual.minute ) #relative_volume.loc[tm1] > 1.5 #and rsi.loc[timestamp_input_interval_adj, 'rsi'] > 70   20 >= timestamp_input_actual.minute >= 5

        price_cond_sell_2 = (cdl_patterns_results["3blackcrows"] and 
                            adx.loc[timestamp_input_interval_adj, 'adx'] > 25 and
                            chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and
                            rsi_pct_change.loc[timestamp_input_interval_adj] < 0 and
                            rsi.loc[timestamp_input_interval_adj,"rsi"] > 20 and
                            20 >=  timestamp_input_actual.minute >= 10)   #relative_volume.loc[tm1] > 1.5 and 

        price_cond_sell_3 = (cdl_patterns_results["flag_bearish"] and
                            adx.loc[timestamp_input_interval_adj, 'adx'] > 25 and
                            chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and
                            relative_volume.loc[tm4] > 1.5 and
                            rsi_pct_change.loc[timestamp_input_interval_adj] < 0 and 
                            20 >= timestamp_input_actual.minute >= 5)
        
        # price_cond_sell_4 = (cdl_patterns_results["inside_candle"] and
        #                     adx.loc[timestamp_input_interval_adj, 'adx'] > 25 and
        #                     chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and
        #                     relative_volume.loc[tm2] > 1.5 and
        #                     50 > rsi.loc[timestamp_input_interval_adj,"rsi"] > 25 and
        #                     rsi_pct_change.loc[timestamp_input_interval_adj] < 0 and
        #                     25 >= timestamp_input_actual.minute >= 13)


        price_cond_sell_4 = (cdl_patterns_results["engulfing_bearish"] and
                            adx.loc[timestamp_input_interval_adj, 'adx'] > 25 and
                            chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and
                            relative_volume.loc[tm1] > 1.8  and
                            rsi.loc[timestamp_input_interval_adj,"rsi"]  > 20 and
                            rsi_pct_change.loc[timestamp_input_interval_adj] < 0 and
                            25 >= timestamp_input_actual.minute)   #and  rsi.loc[timestamp_input_interval_adj, 'rsi'] > 70 
        
        #volume indicator 
        price_cond_sell_5 = (#rsi.loc[timestamp_input_interval_adj,"rsi"] < 50 and 
                             rsi.loc[timestamp_input_interval_adj,"rsi"] > 20 and 
                             rsi_3candle_mean.loc[timestamp_input_interval_adj] < 0  and
                             price_data.loc[timestamp_input_interval_adj, "close"] < price_data.loc[timestamp_input_interval_adj, "open"] and
                             relative_volume.loc[timestamp_input_interval_adj] > 2.5 and
                             chop_index.loc[timestamp_input_interval_adj, "chop"] < 50 and 
                             price_data.loc[timestamp_input_interval_adj, "close"] < ema_short.loc[timestamp_input_interval_adj,"ema"])   # and timestamp_input_actual.minute >= 10)  adx.loc[timestamp_input_interval_adj, 'adx'] > 25 and
                            
        price_buy_conditions = [price_cond_buy_1, price_cond_buy_2, price_cond_buy_3, price_cond_buy_4, price_cond_buy_5] # , price_cond_buy_6
        price_sell_conditions = [price_cond_sell_1, price_cond_sell_2, price_cond_sell_3, price_cond_sell_4, price_cond_sell_5] # ,  price_cond_sell_6

        buy_condition_names = ["morningstardoji", "3whitesoldiers", "flag_bullish","engulfing_bullish","volume_buy"] # "inside_candle", ,"volume_buy"   # "special_engulfing_bullish", "inside_candle",  "retest_bullish", 
        sell_condition_names = ["eveningstardoji", "3blackcrows", "flag_bearish", "engulfing_bearish","volume_sell"]   # "inside_candle", ,"volume_sell"   # "special_engulfing_bearish" "inside_candle",  "retest_bearish", 

        if any(price_buy_conditions):
            signal = "buy"
            true_indices = [i for i, condition in enumerate(price_buy_conditions) if condition]
            for item in true_indices:
                condition_index = item 
                condition = buy_condition_names[condition_index]
            self.logger.info(f"{'#'*1} Buy signal triggered by condition {condition} {'#'*1}")
        elif any(price_sell_conditions):
            signal = "sell"
            true_indices = [i for i, condition in enumerate(price_sell_conditions) if condition]
            for item in true_indices:
                condition_index = item 
                condition = sell_condition_names[condition_index]
            self.logger.info(f"{'#'*1} Sell signal triggered by condition {condition} {'#'*1}")
        else:
            signal = "neutral"
            condition_index = None

        strategy_data = pd.DataFrame(index=[timestamp_input_actual], columns=["price","signal", "condition"])
        strategy_data.loc[timestamp_input_actual, "price"] = price_data.loc[timestamp_input_interval_adj, "close"]
        strategy_data.loc[timestamp_input_actual, "signal"] = signal
        strategy_data.loc[timestamp_input_actual, "condition"] = condition_index +1 if condition_index is not None else None
        strategy_data.loc[timestamp_input_actual,"open"] = price_data.loc[timestamp_input_interval_adj, "open"]
        strategy_data.loc[timestamp_input_actual,"high"] = price_data.loc[timestamp_input_interval_adj, "high"]
        strategy_data.loc[timestamp_input_actual,"low"] = price_data.loc[timestamp_input_interval_adj, "low"]
        strategy_data.loc[timestamp_input_actual,"close"] = price_data.loc[timestamp_input_interval_adj, "close"]
        
        strategy_data.loc[timestamp_input_actual, "rsi"] = rsi.loc[timestamp_input_interval_adj, "rsi"]
        strategy_data.loc[timestamp_input_actual, "atr"] = atr.loc[timestamp_input_interval_adj, "atr"]
        # strategy_data.loc[timestamp_input_actual, "ema_long"] = ema_long.loc[timestamp_input_interval_adj, "ema"]
        strategy_data.loc[timestamp_input_actual, "adx"] = adx.loc[timestamp_input_interval_adj, "adx"]
        strategy_data.loc[timestamp_input_actual, "chop"] = chop_index.loc[timestamp_input_interval_adj, "chop"]
        strategy_data.loc[timestamp_input_actual, "rel_vol"] = relative_volume.loc[timestamp_input_interval_adj]
        strategy_data.loc[timestamp_input_actual, "rsi_pct_change"] = rsi_pct_change.loc[timestamp_input_interval_adj]
        strategy_data.loc[timestamp_input_actual, "rsi_3candle_mean"] = rsi_3candle_mean.loc[timestamp_input_interval_adj]
        
        strategy_data.loc[timestamp_input_actual, "condition_1_buy"] = price_cond_buy_1
        strategy_data.loc[timestamp_input_actual, "condition_2_buy"] = price_cond_buy_2
        strategy_data.loc[timestamp_input_actual, "condition_3_buy"] = price_cond_buy_3
        strategy_data.loc[timestamp_input_actual, "condition_4_buy"] = price_cond_buy_4
        strategy_data.loc[timestamp_input_actual, "condition_5_buy"] = price_cond_buy_5
        # strategy_data.loc[timestamp_input_actual, "condition_6_buy"] = price_cond_buy_6
        # strategy_data.loc[timestamp_input_actual, "condition_7_buy"] = price_cond_buy_7

        strategy_data.loc[timestamp_input_actual, "condition_1_sell"] = price_cond_sell_1
        strategy_data.loc[timestamp_input_actual, "condition_2_sell"] = price_cond_sell_2
        strategy_data.loc[timestamp_input_actual, "condition_3_sell"] = price_cond_sell_3
        strategy_data.loc[timestamp_input_actual, "condition_4_sell"] = price_cond_sell_4
        strategy_data.loc[timestamp_input_actual, "condition_5_sell"] = price_cond_sell_5
        # strategy_data.loc[timestamp_input_actual, "condition_6_sell"] = price_cond_sell_6
        # strategy_data.loc[timestamp_input_actual, "condition_7_sell"] = price_cond_sell_7

        return signal, atr.loc[timestamp_input_interval_adj, 'atr'], condition_index, strategy_data


    def waverider_indicator(self, coin, fiat, interval, timestamp_input_interval_adj, timestamp_input_actual, logger=None, simulate_live_data=False, verbose=False):
        """
        WaveRider Indicator Strategy
        """

        # Download price data
        price_data, minute_data = self.kucoin_price_loader.download_data(
            coin, fiat, interval, end_date=timestamp_input_actual, 
            use_local_timezone=True, drop_not_complete_timestamps=False, simulate_live_data=simulate_live_data, overwrite_file=False
        )
        
        
        # Calculate indicators
        ema_short = self.TA_creator.calculate_indicator(
            price_data, coin=coin, fiat=fiat, interval=interval, 
            timestamp_input=timestamp_input_interval_adj, indicator_name='ema', overwrite_file=True, plot=False, ohlc="close",   
            ema={'span': 100}
        )
        
        ema_long = self.TA_creator.calculate_indicator(
            price_data, coin=coin, fiat=fiat, interval=interval, 
            timestamp_input=timestamp_input_interval_adj, indicator_name='ema', overwrite_file=True, plot=False, ohlc="close", 
            ema={'span': 200}
        )
        
        rsi = self.TA_creator.calculate_indicator(
            price_data, coin=coin, fiat=fiat, interval=interval, 
            timestamp_input=timestamp_input_interval_adj, indicator_name='rsi', overwrite_file=True, plot=False, ohlc="close", 
            rsi={'period': 3}
        )
        
        atr = self.TA_creator.calculate_indicator(
            price_data, coin=coin, fiat=fiat, interval=interval, 
            timestamp_input=timestamp_input_interval_adj, indicator_name='atr', overwrite_file=True, plot=False, ohlc="close", 
            atr={'period': 10}
        )
        
        # Calculate ATR bands
        atr_multiplier = 3
        upper_band = price_data['close'] + (atr['atr'] * atr_multiplier)
        lower_band = price_data['close'] - (atr['atr'] * atr_multiplier)
        
        # Ensure the index is datetime
        price_data.index = pd.to_datetime(price_data.index)
        minute_data.index = pd.to_datetime(minute_data['time'])

        # Ensure indicators have datetime index
        ema_short.index = pd.to_datetime(ema_short.index)
        ema_long.index = pd.to_datetime(ema_long.index)
        rsi.index = pd.to_datetime(rsi.index)
        atr.index = pd.to_datetime(atr.index)
        upper_band.index = pd.to_datetime(upper_band.index)
        lower_band.index = pd.to_datetime(lower_band.index)
        
        # Check if the adjusted timestamp exists in the data
        if timestamp_input_interval_adj not in price_data.index:
            if verbose:
                self.logger.info(f"Timestamp {timestamp_input_interval_adj} not in DataFrame.")
            return "neutral"
        
        # Get current values
        current_price = price_data.loc[timestamp_input_interval_adj, 'close']
        current_ema_short = ema_short.loc[timestamp_input_interval_adj, 'ema']
        current_ema_long = ema_long.loc[timestamp_input_interval_adj, 'ema']
        current_rsi = rsi.loc[timestamp_input_interval_adj, 'rsi']
        current_atr = atr.loc[timestamp_input_interval_adj, 'atr']
        current_upper_band = upper_band.loc[timestamp_input_interval_adj]
        current_lower_band = lower_band.loc[timestamp_input_interval_adj]
        
        # Determine crossover signals
        ema_crossover = (ema_short['ema'].shift(1) < ema_long['ema'].shift(1)) & (ema_short['ema'] > ema_long['ema'])
        ema_crossunder = (ema_short['ema'].shift(1) > ema_long['ema'].shift(1)) & (ema_short['ema'] < ema_long['ema'])
        
        # Get signal at the current timestamp
        buy_signal = ema_crossover.loc[timestamp_input_interval_adj] and current_rsi < 20
        sell_signal = ema_crossunder.loc[timestamp_input_interval_adj] and current_rsi > 80
        
        # Determine the signal
        if buy_signal:
            signal = "buy"
            if verbose:
                self.logger.info(f"Buy signal at {timestamp_input_interval_adj}")
        elif sell_signal:
            signal = "sell"
            if verbose:
                self.logger.info(f"Sell signal at {timestamp_input_interval_adj}")
        else:
            signal = "neutral"
            if verbose:
                self.logger.info(f"No signal at {timestamp_input_interval_adj}")
        
        # Prepare strategy data for logging or further analysis
        strategy_data = pd.DataFrame(index=[timestamp_input_actual], columns=[
            "price", "signal", "rsi", "atr", "ema_short", "ema_long", "upper_band", "lower_band"
        ])
        strategy_data.loc[timestamp_input_actual, "price"] = current_price
        strategy_data.loc[timestamp_input_actual, "signal"] = signal
        strategy_data.loc[timestamp_input_actual, "rsi"] = current_rsi
        strategy_data.loc[timestamp_input_actual, "atr"] = current_atr
        strategy_data.loc[timestamp_input_actual, "ema_short"] = current_ema_short
        strategy_data.loc[timestamp_input_actual, "ema_long"] = current_ema_long
        strategy_data.loc[timestamp_input_actual, "upper_band"] = current_upper_band
        strategy_data.loc[timestamp_input_actual, "lower_band"] = current_lower_band
        
        condition_index=1

        return signal, current_atr, condition_index, strategy_data


    def ma_high_low_rsi_5m_strategy(self, coin, fiat, interval, trend_interval, timestamp_input_interval_adj, timestamp_input_actual, logger=None, simulate_live_data=False, verbose=False):
        """
        EMA Cross strategy
        #uses tm1 timestamp to check if values crossed on hourly and todays timestamps for multi timeframe strategy

        if trend interval is not specified the function does not use multitimeframe approach
        """

        price_data, minute_data = self.kucoin_price_loader.download_data(coin, fiat, interval, end_date=timestamp_input_actual, 
                                                            use_local_timezone=True, drop_not_complete_timestamps=False, simulate_live_data=simulate_live_data, overwrite_file=False)    #start_date="2017-10-05 00:00:00"  #use timestamp input and t_1 when you want to use uncompleted current timestep values
        
        ma_high = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                            timestamp_input=timestamp_input_interval_adj, indicator_name='ma_high', overwrite_file=True, plot=False, ohlc="high",   
                                            high_ma={'span': 20})
        
        ma_low = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ma_low', overwrite_file=True, plot=False, ohlc="low", 
                                        low_ma={'span': 20})
        
        adx = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='adx', overwrite_file=True, plot=False, ohlc="close", 
                                        adx={'di_length': 14})
        
        rsi = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='rsi', overwrite_file=True, plot=False, ohlc="close", 
                                        rsi={'period': 14})

        atr = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='atr', overwrite_file=True, plot=False, ohlc="close", 
                                        atr={'period': 7})


        # local_minima_price, local_maxima_price  = self.TA_creator.find_local_extrema(price_data["close"])
        local_minima_price, local_maxima_price = self.TA_creator.find_local_extrema_filtered(price_data["close"])

        # local_minima_rsi, local_maxima_rsi = self.TA_creator.find_local_extrema(rsi["rsi"])
        local_minima_rsi, local_maxima_rsi = self.TA_creator.find_local_extrema_filtered(rsi["rsi"])
        minima_extrema_diff = abs(rsi.loc[timestamp_input_interval_adj, "rsi"] - local_minima_rsi[-1][1])
        maxima_extrema_diff = abs(local_maxima_rsi[-1][1] - rsi.loc[timestamp_input_interval_adj, "rsi"])



        ma_low.index = pd.to_datetime(ma_low.index)
        ma_high.index = pd.to_datetime(ma_high.index)
        atr.index = pd.to_datetime(atr.index)
        rsi.index = pd.to_datetime(rsi.index)
        adx.index = pd.to_datetime(adx.index)
       
        if timestamp_input_interval_adj not in ma_low.index or timestamp_input_interval_adj not in ma_high.index:
            # self.logger.info(f"Timestamp {timestamp_input} not in DataFrame.")
            current_trend = "neutral"
            return current_trend
        
        last_used_minute_data = minute_data.loc[minute_data['time'] <= timestamp_input_actual, 'time'].iloc[-1]
        minute_data = minute_data.set_index('time') 

        last_available_timestep = price_data.index[-2]
        trend_window_index = price_data.index[-4] 

        buy_tm1_price = min(price_data.loc[last_available_timestep,["open","close"]])
        sell_tm1_price = max(price_data.loc[last_available_timestep,["open","close"]])
        prev_price = price_data.loc[last_available_timestep,'close']

        #check if last 3 prices were above low ema 
        buy_trend_values = [min(price_data.loc[step, ["open","close"]]) for step in price_data[trend_window_index:last_available_timestep].index]
        
        #check if last 3 prices were below high ema
        sell_trend_values = [max(price_data.loc[step, ["open","close"]]) for step in price_data[trend_window_index:last_available_timestep].index]
        
        
        if verbose:
            # self.logger.info(f"{'#'*1} current trend is {current_trend} {'#'*1} with macd {macd.loc[utc_time, 'macd']:.2f} and signal {macd.loc[utc_time, 'signal']:.2f} at time {utc_time}")
            
            
            if simulate_live_data:
                try:
                    
                    self.logger.info(f"{'#'*1} Interval time:{timestamp_input_interval_adj}, interval price: {price_data.loc[timestamp_input_interval_adj, 'close']:.2f} # actual time: {timestamp_input_actual} minute price: {minute_data.loc[last_used_minute_data, 'close']:.2f} # Previous time: {last_available_timestep}, previous price: {prev_price} ")
                except:
                    pass
            else:
                
                self.logger.info(f"{'#'*1} Interval time:{timestamp_input_interval_adj}, interval price: {price_data.loc[timestamp_input_interval_adj, 'close']:.2f} # Previous time: {last_available_timestep}, previous price: {prev_price} #")
            
            self.logger.info(f"{'#'*1} high_ma: {ma_high.loc[timestamp_input_interval_adj, 'high_ma']:.2f} # low_ma: {ma_low.loc[timestamp_input_interval_adj, 'low_ma']:.2f} # adx: {adx.loc[timestamp_input_interval_adj,'adx']} # rsi: {rsi.loc[timestamp_input_interval_adj,'rsi']} # atr: {atr.loc[timestamp_input_interval_adj,'atr']} {'#'*1} ")
            
        price_buy_zone = ma_high.loc[timestamp_input_interval_adj, 'high_ma'] * 0.995 
        price_sell_zone = ma_low.loc[timestamp_input_interval_adj, 'low_ma'] * 1.005
        
        # ma condition buy
        price_cond_buy_1 =   (price_data.loc[timestamp_input_interval_adj, "close"] < ma_high.loc[timestamp_input_interval_adj, 'high_ma'] and 
                              buy_tm1_price >= ma_high.loc[last_available_timestep, 'high_ma'] and 
                              rsi.loc[timestamp_input_interval_adj, 'rsi'] > 50 and 
                              price_data.loc[timestamp_input_interval_adj, "close"] > price_buy_zone and 
                              adx.loc[timestamp_input_interval_adj, 'adx'] > 25)   #   and  #price under high ma & prev  low above ma
        

        #ma condition sell
        price_cond_sell_1 = (price_data.loc[timestamp_input_interval_adj, "close"] > ma_low.loc[timestamp_input_interval_adj, 'low_ma'] and 
                             sell_tm1_price <= ma_low.loc[last_available_timestep, 'low_ma'] and
                             rsi.loc[timestamp_input_interval_adj, 'rsi'] < 50 and 
                             price_data.loc[timestamp_input_interval_adj, "close"] < price_sell_zone and
                             adx.loc[timestamp_input_interval_adj, 'adx'] > 25)   #adx.loc[timestamp_input_interval_adj, 'adx'] > 25 and
       
        
        price_buy_conditions = [price_cond_buy_1]
        price_sell_conditions = [price_cond_sell_1]

        # Determine the signal and the index of the condition that triggered it
        if any(price_buy_conditions):
            signal = "buy"
            condition_index = price_buy_conditions.index(True)
            if condition_index == 0:
                condition = "cond_buy_1_ma"
        
            self.logger.info(f"{'#'*1} Buy signal triggered by condition {condition} {'#'*1}")
        elif any(price_sell_conditions):
            signal = "sell"
            condition_index = price_sell_conditions.index(True)
            if condition_index == 0:
                condition = "cond_sell_1_ma"
        
            self.logger.info(f"{'#'*1} Sell signal triggered by condition {condition} {'#'*1}")
        else:
            signal = "neutral"
            condition_index = None

        strategy_data = pd.DataFrame(index=[timestamp_input_actual], columns=["price","signal","rsi","rsi_7","adx","atr","rel_vol","ema_long","ma_high","ma_low","buy_tm1","sell_tm1"])
        strategy_data.loc[timestamp_input_actual, "price"] = price_data.loc[timestamp_input_interval_adj, "close"]
        strategy_data.loc[timestamp_input_actual, "signal"] = signal
        strategy_data.loc[timestamp_input_actual, "condition"] = condition_index +1 if condition_index is not None else None
        strategy_data.loc[timestamp_input_actual, "rsi"] = rsi.loc[timestamp_input_interval_adj, "rsi"]
        strategy_data.loc[timestamp_input_actual, "adx"] = adx.loc[timestamp_input_interval_adj, "adx"]
        strategy_data.loc[timestamp_input_actual, "atr"] = atr.loc[timestamp_input_interval_adj, "atr"]
        strategy_data.loc[timestamp_input_actual, "ma_high"] = ma_high.loc[timestamp_input_interval_adj, "high_ma"]
        strategy_data.loc[timestamp_input_actual, "ma_low"] = ma_low.loc[timestamp_input_interval_adj, "low_ma"]
        strategy_data.loc[timestamp_input_actual, "buy_tm1"] = buy_tm1_price
        strategy_data.loc[timestamp_input_actual, "sell_tm1"] = sell_tm1_price
        strategy_data.loc[timestamp_input_actual, "maxima_t"] = pd.Timestamp(local_maxima_price[-1][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "maxima_t1"] = pd.Timestamp(local_maxima_price[-2][0]).strftime('%Y-%m-%d %H:%M:%S')      
        strategy_data.loc[timestamp_input_actual, "maxima_rsi_t"] = pd.Timestamp(local_maxima_rsi[-1][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "maxima_rsi_t1"] = pd.Timestamp(local_maxima_rsi[-2][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "minima_t"] = pd.Timestamp(local_minima_price[-1][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "minima_t1"] = pd.Timestamp(local_minima_price[-2][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "minima_rsi_t"] = pd.Timestamp(local_minima_rsi[-1][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "minima_rsi_t1"] = pd.Timestamp(local_minima_rsi[-2][0]).strftime('%Y-%m-%d %H:%M:%S')
        strategy_data.loc[timestamp_input_actual, "maxima_t_value"] = local_maxima_price[-1][1]
        strategy_data.loc[timestamp_input_actual, "maxima_t1_value"] = local_maxima_price[-2][1]
        strategy_data.loc[timestamp_input_actual, "maxima_rsi_t_value"] = local_maxima_rsi[-1][1]
        strategy_data.loc[timestamp_input_actual, "maxima_rsi_t1_value"] = local_maxima_rsi[-2][1]
        strategy_data.loc[timestamp_input_actual, "minima_t_value"] = local_minima_price[-1][1]
        strategy_data.loc[timestamp_input_actual, "minima_t1_value"] = local_minima_price[-2][1]
        strategy_data.loc[timestamp_input_actual, "minima_rsi_t_value"] = local_minima_rsi[-1][1]
        strategy_data.loc[timestamp_input_actual, "minima_rsi_t1_value"] = local_minima_rsi[-2][1]
        strategy_data.loc[timestamp_input_actual, "condition_1_buy"] = price_cond_buy_1
        strategy_data.loc[timestamp_input_actual, "condition_1_sell"] = price_cond_sell_1

        return signal, atr.loc[timestamp_input_interval_adj, 'atr'], condition_index, strategy_data

    def fibonacci_retracement_strategy(self, coin, fiat, interval, trend_interval, timestamp_input_interval_adj, timestamp_input_actual, logger=None, simulate_live_data=False, verbose=False):
        """
        EMA Cross strategy
        #uses tm1 timestamp to check if values crossed on hourly and todays timestamps for multi timeframe strategy

        if trend interval is not specified the function does not use multitimeframe approach
        """

        price_data, minute_data = self.kucoin_price_loader.download_data(coin, fiat, interval, end_date=timestamp_input_actual, 
                                                            use_local_timezone=True, drop_not_complete_timestamps=False, simulate_live_data=simulate_live_data, overwrite_file=False)    #start_date="2017-10-05 00:00:00"  #use timestamp input and t_1 when you want to use uncompleted current timestep values
        
        adx = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='adx', overwrite_file=True, plot=False, ohlc="close", 
                                        adx={'di_length': 14})
        
        rsi = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='rsi', overwrite_file=True, plot=False, ohlc="close", 
                                        rsi={'period': 14})

        atr = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='atr', overwrite_file=True, plot=False, ohlc="close", 
                                        atr={'period': 14})
        
        ema_short = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ema', overwrite_file=True, plot=False, ohlc="close", 
                                        ema={'span': 9})
        
        ema_long = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ema', overwrite_file=True, plot=False, ohlc="close", 
                                        ema={'span': 21})
        
        fibonacci = self.TA_creator.calculate_fibonacci_retracement_levels(price_data, window=(24*21))    #fibonacci dataframe containing each retracement as colum on 72 hour window (3days)

        volume_ma = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ma', overwrite_file=True, plot=False, ohlc="turnover", 
                                        ma={'span': 50})
        
        relative_volume = (price_data['turnover'] / volume_ma["ma"])

        # volume_profile = self.TA_creator.calculate_volume_profile(price_data, window= 100, num_bins= 20)
        # volume_zones =  self.TA_creator.identify_hvn_lvn(volume_profile)

        atr.index = pd.to_datetime(atr.index)
        rsi.index = pd.to_datetime(rsi.index)
        adx.index = pd.to_datetime(adx.index)
        fibonacci.index = pd.to_datetime(fibonacci.index)
        # volume_profile.index = pd.to_datetime(volume_profile.index)
        # volume_zones.index = pd.to_datetime(volume_zones.index)
        ema_short.index = pd.to_datetime(ema_short.index)
        ema_long.index = pd.to_datetime(ema_long.index)

    
        if isinstance(timestamp_input_interval_adj, dt.datetime):
            utc_time = pd.Timestamp(timestamp_input_interval_adj)
        
        if not np.all(price_data.localized):
            utc_time = utc_time.tz_localize('Europe/Berlin', ambiguous=True).tz_convert('UTC')
            utc_time = utc_time.replace(tzinfo=None)
        
        if trend_interval == '1d':
            utc_time = utc_time.tz_localize('Europe/Berlin', ambiguous=True).tz_convert('UTC')
            utc_time = utc_time.replace(tzinfo=None)
            utc_time = utc_time.replace(hour=0, minute=0, second=0, microsecond=0)
        elif trend_interval == '1h':
            utc_time = utc_time.replace(tzinfo=None)
            utc_time = utc_time.replace(minute=0, second=0, microsecond=0) #- timedelta(hours=1) #Using tm1 as hour is not completed yet       uncomment to get data
        elif trend_interval == '15m':
            utc_time = utc_time.replace(tzinfo=None)      
            utc_time = utc_time.replace(second=0, microsecond=0)
            utc_time = utc_time.replace(minute=15 * (utc_time.minute // 15)) # - timedelta(minutes=15) #Using tm1 as 15m is not completed yet       uncomment to get data
        elif trend_interval == '5m':
            utc_time = utc_time.replace(tzinfo=None)      
            utc_time = utc_time.replace(second=0, microsecond=0)
            utc_time = utc_time.replace(minute=5 * (utc_time.minute // 5))
    
        if utc_time not in price_data.index or utc_time not in price_data.index:
            # self.logger.info(f"Timestamp {timestamp_input} not in DataFrame.")
            current_trend = "neutral"
            return current_trend
        
        last_used_minute_data = minute_data.loc[minute_data['time'] <= timestamp_input_actual, 'time'].iloc[-1]
        minute_data = minute_data.set_index('time') 

        last_available_timestep = price_data.index[-2]
        tm2_timestamp = price_data.index[-3]
        prev_price = price_data.loc[last_available_timestep,'close']
        
        # Calculate key levels from Fibonacci and Volume Profile zones
        fib_levels = fibonacci.loc[utc_time]
        trend = fibonacci.loc[utc_time, 'Trend']
        # hvn_zones = volume_zones.loc[utc_time, 'HVN_Zones']
        # lvn_zones = volume_zones.loc[utc_time, 'LVN_Zones']
        
        # Get current values
        current_price = price_data.loc[utc_time, 'close']
        current_high = price_data.loc[utc_time, 'high']
        current_low = price_data.loc[utc_time, 'low']
        rsi_value = rsi.loc[utc_time, 'rsi']
        last_rsi = rsi.loc[last_available_timestep, 'rsi']
        tm2_rsi = rsi.loc[tm2_timestamp, 'rsi']
        rsi_slope = ((rsi_value / last_rsi -1) + (last_rsi / tm2_rsi -1)) / 2  
        atr_value = atr.loc[utc_time, 'atr']
        adx_value = adx.loc[utc_time, 'adx']
        ema_short_value = ema_short.loc[utc_time, 'ema']
        ema_long_value = ema_long.loc[utc_time, 'ema']
        relative_volume_value = relative_volume.loc[utc_time]

        if verbose:
            # self.logger.info(f"{'#'*1} current trend is {current_trend} {'#'*1} with macd {macd.loc[utc_time, 'macd']:.2f} and signal {macd.loc[utc_time, 'signal']:.2f} at time {utc_time}")
            self.logger.info(f"{'#'*1} Current interval time: {timestamp_input_interval_adj} price: {price_data.loc[timestamp_input_interval_adj, 'close']:.2f}. Previous time: {last_available_timestep} price: {prev_price}.  {'#'*1} ")
        
            if simulate_live_data:
                try:
                    self.logger.info(f"{'#'*1} Current  actual time: {timestamp_input_actual} price: {minute_data.loc[last_used_minute_data, 'close']:.2f}. {'#'*1} ")
                except:
                    pass
                
            self.logger.info(f"{'#'*1} atr: {atr_value}, rsi: {rsi_value}, adx: {adx_value}, ema_short: {ema_short_value}, ema_long: {ema_long_value}, relative volume: {relative_volume_value}, rsi slope: {rsi_slope}  {'#'*1} ") 
            self.logger.info(f"{'#'*1} fib trend: {trend}, fib 23.6: {fib_levels['Fib_23.6']}, fib 38.2: {fib_levels['Fib_38.2']}, fib 50.0: {fib_levels['Fib_50.0']}, fib 61.8: {fib_levels['Fib_61.8']}, fib 78.6: {fib_levels['Fib_78.6']}, fib 100.0: {fib_levels['Fib_100.0']} {'#'*1}")

        # Determine trend direction based on EMA cross
        in_uptrend = ema_short_value > ema_long_value
        in_downtrend = ema_short_value < ema_long_value

        # near_hvn_zone = any(start <= current_price <= end for start, end in hvn_zones)
        # near_lvn_zone = any(start <= current_price <= end for start, end in lvn_zones)              

        # Conditions for Buy/Sell signals with adjusted criteria
        # Buy conditions
        # Conditions for Buy/Sell signals with adjusted criteria
        
        if trend == "uptrend":
            if adx_value < 25:

                if rsi_value >= 50 and rsi_slope>=0:
                    if current_price > fib_levels['Fib_78.6'] and current_low < fib_levels['Fib_78.6']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_78.6']  - (min(atr_value,750) * 2)
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price > fib_levels['Fib_61.8'] and current_low < fib_levels['Fib_61.8']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_78.6']
                        take_profit = fib_levels['Fib_50.0']
                    elif current_price > fib_levels['Fib_50.0'] and current_low < fib_levels['Fib_50.0']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_61.8']
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price > fib_levels['Fib_38.2'] and current_low < fib_levels['Fib_38.2']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_50.0']
                        take_profit = fib_levels['Fib_23.6']
                    else:
                        signal = "neutral"
                        stop_loss = None
                        take_profit = None
                    
                    
                elif rsi_value < 50 and rsi_slope<=0:
                    if current_price < fib_levels['Fib_23.6'] and current_high > fib_levels['Fib_23.6']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_23.6'] + (min(atr_value,750) * 2)
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price < fib_levels['Fib_38.2'] and current_high > fib_levels['Fib_38.2']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_23.6']
                        take_profit = fib_levels['Fib_50.0']
                    elif current_price < fib_levels['Fib_50.0'] and current_high > fib_levels['Fib_50.0']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_38.2']
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price < fib_levels['Fib_61.8'] and current_high > fib_levels['Fib_61.8']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_50.0']
                        take_profit = fib_levels['Fib_78.6']
                    else:
                        signal = "neutral"
                        stop_loss = None
                        take_profit = None
                else:
                    signal = "neutral"
                    stop_loss = None
                    take_profit = None  

            elif adx_value > 25:
                if rsi_value >50 and relative_volume_value > 1.3:
                    if current_price > fib_levels['Fib_78.6'] and current_low < fib_levels['Fib_78.6']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_78.6']  - (min(atr_value,750) * 2)
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price > fib_levels['Fib_61.8'] and current_low < fib_levels['Fib_61.8']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_78.6']
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price > fib_levels['Fib_50.0'] and current_low < fib_levels['Fib_50.0']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_61.8']
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price > fib_levels['Fib_38.2'] and current_low < fib_levels['Fib_38.2']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_50.0']
                        take_profit = fib_levels['Fib_23.6']
                    elif current_price > fib_levels['Fib_23.6'] and current_low < fib_levels['Fib_23.6']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_38.2']
                        take_profit = fib_levels['Fib_23.6']  + (min(atr_value,750) * 2)
                    else:
                        signal = "neutral"
                        stop_loss = None
                        take_profit = None
                
                elif rsi_value < 50 and relative_volume_value > 1.3:
                    if current_price < fib_levels['Fib_23.6'] and current_high > fib_levels['Fib_23.6']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_23.6']  + (min(atr_value,750) * 2)
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price < fib_levels['Fib_38.2'] and current_high > fib_levels['Fib_38.2']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_23.6']
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price < fib_levels['Fib_50.0'] and current_high > fib_levels['Fib_50.0']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_38.2']
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price < fib_levels['Fib_61.8'] and current_high > fib_levels['Fib_61.8']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_50.0']
                        take_profit = fib_levels['Fib_78.6']
                    elif current_price < fib_levels['Fib_78.6'] and current_high > fib_levels['Fib_78.6']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_61.8']
                        take_profit =  fib_levels['Fib_78.6'] - (min(atr_value,750) * 2)
                    else:
                        signal = "neutral"
                        stop_loss = None
                        take_profit = None
                else:
                    signal = "neutral"
                    stop_loss = None
                    take_profit = None
            
        elif trend == "downtrend":
            if adx_value <= 25:
                if rsi_value > 50 and rsi_slope>=0:
                    
                    if current_price > fib_levels['Fib_23.6'] and current_high < fib_levels['Fib_23.6']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_23.6'] - (min(atr_value,750) * 2)
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price > fib_levels['Fib_38.2'] and current_high < fib_levels['Fib_38.2']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_23.6']
                        take_profit = fib_levels['Fib_50.0']
                    elif current_price > fib_levels['Fib_50.0'] and current_high < fib_levels['Fib_50.0']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_38.2']
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price > fib_levels['Fib_61.8'] and current_high < fib_levels['Fib_61.8']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_50.0']
                        take_profit = fib_levels['Fib_78.6']
                    else:
                        signal = "neutral"
                        stop_loss = None
                        take_profit = None
                
                elif rsi_value < 50 and rsi_slope<=0:
                    if current_price < fib_levels['Fib_78.6'] and current_low > fib_levels['Fib_78.6']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_78.6'] + (min(atr_value,750) * 2)
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price < fib_levels['Fib_61.8'] and current_low > fib_levels['Fib_61.8']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_78.6']
                        take_profit = fib_levels['Fib_50.0']
                    elif current_price < fib_levels['Fib_50.0'] and current_low > fib_levels['Fib_50.0']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_61.8']
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price < fib_levels['Fib_38.2'] and current_low > fib_levels['Fib_38.2']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_50.0']
                        take_profit = fib_levels['Fib_23.6']
                    else:
                        signal = "neutral"
                        stop_loss = None
                        take_profit = None
                else:
                    signal = "neutral"
                    stop_loss = None
                    take_profit = None

            elif adx_value > 25:
                if rsi_value > 50 and relative_volume_value > 1.3:
                    if current_price > fib_levels['Fib_23.6'] and current_low < fib_levels['Fib_23.6']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_23.6'] - (min(atr_value,750) * 2)
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price > fib_levels['Fib_38.2'] and current_low < fib_levels['Fib_38.2']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_23.6']
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price > fib_levels['Fib_50.0'] and current_low < fib_levels['Fib_50.0']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_38.2']
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price > fib_levels['Fib_61.8'] and current_low < fib_levels['Fib_61.8']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_50.0']
                        take_profit = fib_levels['Fib_78.6']
                    elif current_price > fib_levels['Fib_78.6'] and current_low < fib_levels['Fib_78.6']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_61.8']
                        take_profit =  fib_levels['Fib_78.6'] + (min(atr_value,750) * 2)
                    else:
                        signal = "neutral"
                        stop_loss = None
                        take_profit = None

                elif rsi_value < 50 and relative_volume_value > 1.3:
                    if current_price < fib_levels['Fib_78.6'] and current_high > fib_levels['Fib_78.6']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_78.6'] + (min(atr_value,750) * 2)
                        take_profit = fib_levels['Fib_61.8']
                    
                    elif current_price < fib_levels['Fib_61.8'] and current_high > fib_levels['Fib_61.8']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_78.6']
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price < fib_levels['Fib_50.0'] and current_high > fib_levels['Fib_50.0']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_61.8']
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price < fib_levels['Fib_38.2'] and current_high > fib_levels['Fib_38.2']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_50.0']
                        take_profit = fib_levels['Fib_23.6']
                    elif current_price < fib_levels['Fib_23.6'] and current_high > fib_levels['Fib_23.6']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_38.2']
                        take_profit = fib_levels['Fib_23.6']  + (min(atr_value,750) * 2)
                    else:
                        signal = "neutral"
                        stop_loss = None
                        take_profit = None
                else:
                    signal = "neutral"
                    stop_loss = None
                    take_profit = None
        
        return signal, stop_loss, take_profit, atr_value

    def fibonacci_ma_high_low_strategy(self, coin, fiat, interval, trend_interval, timestamp_input_interval_adj, timestamp_input_actual, logger=None, simulate_live_data=False, verbose=False):
        """
        EMA Cross strategy
        #uses tm1 timestamp to check if values crossed on hourly and todays timestamps for multi timeframe strategy

        if trend interval is not specified the function does not use multitimeframe approach
        """

        price_data, minute_data = self.kucoin_price_loader.download_data(coin, fiat, interval, end_date=timestamp_input_actual, 
                                                            use_local_timezone=True, drop_not_complete_timestamps=False, simulate_live_data=simulate_live_data, overwrite_file=False)    #start_date="2017-10-05 00:00:00"  #use timestamp input and t_1 when you want to use uncompleted current timestep values
        
        adx = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='adx', overwrite_file=True, plot=False, ohlc="close", 
                                        adx={'di_length': 14})
        
        rsi = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='rsi', overwrite_file=True, plot=False, ohlc="close", 
                                        rsi={'period': 14})

        atr = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='atr', overwrite_file=True, plot=False, ohlc="close", 
                                        atr={'period': 14})
        
        ema_short = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ema', overwrite_file=True, plot=False, ohlc="close", 
                                        ema={'span': 9})
        
        ema_long = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ema', overwrite_file=True, plot=False, ohlc="close", 
                                        ema={'span': 21})
        
        ma_high = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                            timestamp_input=timestamp_input_interval_adj, indicator_name='ma_high', overwrite_file=True, plot=False, ohlc="high",   
                                            high_ma={'span': 20})
        
        ma_low = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ma_low', overwrite_file=True, plot=False, ohlc="low", 
                                        low_ma={'span': 20})

        fibonacci = self.TA_creator.calculate_fibonacci_retracement_levels(price_data, window=(24*21))    #fibonacci dataframe containing each retracement as colum on 72 hour window (3days)

        volume_ma = self.TA_creator.calculate_indicator(price_data, coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='ma', overwrite_file=True, plot=False, ohlc="volume", 
                                        ma={'span': 50})
        
        relative_volume = price_data['volume'] / volume_ma

        volume_profile = self.TA_creator.calculate_volume_profile(price_data, window= 100, num_bins= 20)
        volume_zones =  self.TA_creator.identify_hvn_lvn(volume_profile)

        atr.index = pd.to_datetime(atr.index)
        rsi.index = pd.to_datetime(rsi.index)
        adx.index = pd.to_datetime(adx.index)
        fibonacci.index = pd.to_datetime(fibonacci.index)
        volume_profile.index = pd.to_datetime(volume_profile.index)
        volume_zones.index = pd.to_datetime(volume_zones.index)
        ema_short.index = pd.to_datetime(ema_short.index)
        ema_long.index = pd.to_datetime(ema_long.index)
        ma_low.index = pd.to_datetime(ma_low.index)
        ma_high.index = pd.to_datetime(ma_high.index)

    
        if isinstance(timestamp_input_interval_adj, dt.datetime):
            utc_time = pd.Timestamp(timestamp_input_interval_adj)
        
        if not np.all(price_data.localized):
            utc_time = utc_time.tz_localize('Europe/Berlin', ambiguous=True).tz_convert('UTC')
            utc_time = utc_time.replace(tzinfo=None)
        
        if trend_interval == '1d':
            utc_time = utc_time.tz_localize('Europe/Berlin', ambiguous=True).tz_convert('UTC')
            utc_time = utc_time.replace(tzinfo=None)
            utc_time = utc_time.replace(hour=0, minute=0, second=0, microsecond=0)
        elif trend_interval == '1h':
            utc_time = utc_time.replace(tzinfo=None)
            utc_time = utc_time.replace(minute=0, second=0, microsecond=0) #- timedelta(hours=1) #Using tm1 as hour is not completed yet       uncomment to get data
        elif trend_interval == '15m':
            utc_time = utc_time.replace(tzinfo=None)      
            utc_time = utc_time.replace(second=0, microsecond=0)
            utc_time = utc_time.replace(minute=15 * (utc_time.minute // 15)) # - timedelta(minutes=15) #Using tm1 as 15m is not completed yet       uncomment to get data
        elif trend_interval == '5m':
            utc_time = utc_time.replace(tzinfo=None)      
            utc_time = utc_time.replace(second=0, microsecond=0)
            utc_time = utc_time.replace(minute=5 * (utc_time.minute // 5))
    
        if utc_time not in price_data.index or utc_time not in price_data.index:
            # self.logger.info(f"Timestamp {timestamp_input} not in DataFrame.")
            current_trend = "neutral"
            return current_trend
        
        last_used_minute_data = minute_data.loc[minute_data['time'] <= timestamp_input_actual, 'time'].iloc[-1]
        minute_data = minute_data.set_index('time') 

        last_available_timestep = price_data.index[-2]
        tm2_timestamp = price_data.index[-3]
        prev_price = price_data.loc[last_available_timestep,'close']
        
        # Calculate key levels from Fibonacci and Volume Profile zones
        fib_levels = fibonacci.loc[utc_time]
        trend = fibonacci.loc[utc_time, 'Trend']
        hvn_zones = volume_zones.loc[utc_time, 'HVN_Zones']
        lvn_zones = volume_zones.loc[utc_time, 'LVN_Zones']
        
        # Get current values
        current_price = price_data.loc[utc_time, 'close']
        current_high = price_data.loc[utc_time, 'high']
        current_low = price_data.loc[utc_time, 'low']
        rsi_value = rsi.loc[utc_time, 'rsi']
        last_rsi = rsi.loc[last_available_timestep, 'rsi']
        tm2_rsi = rsi.loc[tm2_timestamp, 'rsi']
        rsi_slope = ((rsi_value / last_rsi -1) + (last_rsi / tm2_rsi -1)) / 2  
        atr_value = atr.loc[utc_time, 'atr']
        adx_value = adx.loc[utc_time, 'adx']
        ema_short_value = ema_short.loc[utc_time, 'ema']
        ema_long_value = ema_long.loc[utc_time, 'ema']
        relative_volume_value = relative_volume.loc[utc_time]

        if verbose:
            # self.logger.info(f"{'#'*1} current trend is {current_trend} {'#'*1} with macd {macd.loc[utc_time, 'macd']:.2f} and signal {macd.loc[utc_time, 'signal']:.2f} at time {utc_time}")
            self.logger.info(f"{'#'*1} Current interval time: {timestamp_input_interval_adj} price: {price_data.loc[timestamp_input_interval_adj, 'close']:.2f}. Previous time: {last_available_timestep} price: {prev_price}.  {'#'*1} ")
        
            if simulate_live_data:
                try:
                    self.logger.info(f"{'#'*1} Current  actual time: {timestamp_input_actual} price: {minute_data.loc[last_used_minute_data, 'close']:.2f}. {'#'*1} ")
                except:
                    pass
                
            self.logger.info(f"{'#'*1} atr: {atr_value}, rsi: {rsi_value}, adx: {adx_value}, ema_short: {ema_short_value}, ema_long: {ema_long_value}, relative volume: {relative_volume_value}, rsi slope: {rsi_slope}  {'#'*1} ") 
            self.logger.info(f"{'#'*1} fib trend: {trend}, fib 23.6: {fib_levels['Fib_23.6']}, fib 38.2: {fib_levels['Fib_38.2']}, fib 50.0: {fib_levels['Fib_50.0']}, fib 61.8: {fib_levels['Fib_61.8']}, fib 78.6: {fib_levels['Fib_78.6']}, fib 100.0: {fib_levels['Fib_100.0']} {'#'*1}")

        # Determine trend direction based on EMA cross
        in_uptrend = ema_short_value > ema_long_value
        in_downtrend = ema_short_value < ema_long_value

        near_hvn_zone = any(start <= current_price <= end for start, end in hvn_zones)
        near_lvn_zone = any(start <= current_price <= end for start, end in lvn_zones)              

        # Conditions for Buy/Sell signals with adjusted criteria
        # Buy conditions
        # Conditions for Buy/Sell signals with adjusted criteria
        
        if trend == "uptrend":
            if adx_value < 25:

                if rsi_value >= 50 and rsi_slope>=0:
                    if current_price > fib_levels['Fib_78.6'] and current_low < fib_levels['Fib_78.6']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_78.6']  - (min(atr_value,750) * 2)
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price > fib_levels['Fib_61.8'] and current_low < fib_levels['Fib_61.8']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_78.6']
                        take_profit = fib_levels['Fib_50.0']
                    elif current_price > fib_levels['Fib_50.0'] and current_low < fib_levels['Fib_50.0']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_61.8']
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price > fib_levels['Fib_38.2'] and current_low < fib_levels['Fib_38.2']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_50.0']
                        take_profit = fib_levels['Fib_23.6']
                    else:
                        signal = "neutral"
                        stop_loss = None
                        take_profit = None
                    
                    
                elif rsi_value < 50 and rsi_slope<=0:
                    if current_price < fib_levels['Fib_23.6'] and current_high > fib_levels['Fib_23.6']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_23.6'] + (min(atr_value,750) * 2)
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price < fib_levels['Fib_38.2'] and current_high > fib_levels['Fib_38.2']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_23.6']
                        take_profit = fib_levels['Fib_50.0']
                    elif current_price < fib_levels['Fib_50.0'] and current_high > fib_levels['Fib_50.0']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_38.2']
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price < fib_levels['Fib_61.8'] and current_high > fib_levels['Fib_61.8']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_50.0']
                        take_profit = fib_levels['Fib_78.6']
                    else:
                        signal = "neutral"
                        stop_loss = None
                        take_profit = None
                else:
                    signal = "neutral"
                    stop_loss = None
                    take_profit = None  

            elif adx_value > 25:
                if rsi_value >50 and relative_volume_value > 1.3:
                    if current_price > fib_levels['Fib_78.6'] and current_low < fib_levels['Fib_78.6']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_78.6']  - (min(atr_value,750) * 2)
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price > fib_levels['Fib_61.8'] and current_low < fib_levels['Fib_61.8']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_78.6']
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price > fib_levels['Fib_50.0'] and current_low < fib_levels['Fib_50.0']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_61.8']
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price > fib_levels['Fib_38.2'] and current_low < fib_levels['Fib_38.2']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_50.0']
                        take_profit = fib_levels['Fib_23.6']
                    elif current_price > fib_levels['Fib_23.6'] and current_low < fib_levels['Fib_23.6']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_38.2']
                        take_profit = fib_levels['Fib_23.6']  + (min(atr_value,750) * 2)
                    else:
                        signal = "neutral"
                        stop_loss = None
                        take_profit = None
                
                elif rsi_value < 50 and relative_volume_value > 1.3:
                    if current_price < fib_levels['Fib_23.6'] and current_high > fib_levels['Fib_23.6']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_23.6']  + (min(atr_value,750) * 2)
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price < fib_levels['Fib_38.2'] and current_high > fib_levels['Fib_38.2']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_23.6']
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price < fib_levels['Fib_50.0'] and current_high > fib_levels['Fib_50.0']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_38.2']
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price < fib_levels['Fib_61.8'] and current_high > fib_levels['Fib_61.8']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_50.0']
                        take_profit = fib_levels['Fib_78.6']
                    elif current_price < fib_levels['Fib_78.6'] and current_high > fib_levels['Fib_78.6']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_61.8']
                        take_profit =  fib_levels['Fib_78.6'] - (min(atr_value,750) * 2)
                    else:
                        signal = "neutral"
                        stop_loss = None
                        take_profit = None
                else:
                    signal = "neutral"
                    stop_loss = None
                    take_profit = None
            
        elif trend == "downtrend":
            if adx_value <= 25:
                if rsi_value > 50 and rsi_slope>=0:
                    
                    if current_price > fib_levels['Fib_23.6'] and current_high < fib_levels['Fib_23.6']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_23.6'] - (min(atr_value,750) * 2)
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price > fib_levels['Fib_38.2'] and current_high < fib_levels['Fib_38.2']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_23.6']
                        take_profit = fib_levels['Fib_50.0']
                    elif current_price > fib_levels['Fib_50.0'] and current_high < fib_levels['Fib_50.0']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_38.2']
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price > fib_levels['Fib_61.8'] and current_high < fib_levels['Fib_61.8']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_50.0']
                        take_profit = fib_levels['Fib_78.6']
                    else:
                        signal = "neutral"
                        stop_loss = None
                        take_profit = None
                
                elif rsi_value < 50 and rsi_slope<=0:
                    if current_price < fib_levels['Fib_78.6'] and current_low > fib_levels['Fib_78.6']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_78.6'] + (min(atr_value,750) * 2)
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price < fib_levels['Fib_61.8'] and current_low > fib_levels['Fib_61.8']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_78.6']
                        take_profit = fib_levels['Fib_50.0']
                    elif current_price < fib_levels['Fib_50.0'] and current_low > fib_levels['Fib_50.0']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_61.8']
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price < fib_levels['Fib_38.2'] and current_low > fib_levels['Fib_38.2']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_50.0']
                        take_profit = fib_levels['Fib_23.6']
                    else:
                        signal = "neutral"
                        stop_loss = None
                        take_profit = None
                else:
                    signal = "neutral"
                    stop_loss = None
                    take_profit = None

            elif adx_value > 25:
                if rsi_value > 50 and relative_volume_value > 1.3:
                    if current_price > fib_levels['Fib_23.6'] and current_low < fib_levels['Fib_23.6']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_23.6'] - (min(atr_value,750) * 2)
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price > fib_levels['Fib_38.2'] and current_low < fib_levels['Fib_38.2']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_23.6']
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price > fib_levels['Fib_50.0'] and current_low < fib_levels['Fib_50.0']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_38.2']
                        take_profit = fib_levels['Fib_61.8']
                    elif current_price > fib_levels['Fib_61.8'] and current_low < fib_levels['Fib_61.8']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_50.0']
                        take_profit = fib_levels['Fib_78.6']
                    elif current_price > fib_levels['Fib_78.6'] and current_low < fib_levels['Fib_78.6']:
                        signal = "buy"
                        stop_loss = fib_levels['Fib_61.8']
                        take_profit =  fib_levels['Fib_78.6'] + (min(atr_value,750) * 2)
                    else:
                        signal = "neutral"
                        stop_loss = None
                        take_profit = None

                elif rsi_value < 50 and relative_volume_value > 1.3:
                    if current_price < fib_levels['Fib_78.6'] and current_high > fib_levels['Fib_78.6']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_78.6'] + (min(atr_value,750) * 2)
                        take_profit = fib_levels['Fib_61.8']
                    
                    elif current_price < fib_levels['Fib_61.8'] and current_high > fib_levels['Fib_61.8']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_78.6']
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price < fib_levels['Fib_50.0'] and current_high > fib_levels['Fib_50.0']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_61.8']
                        take_profit = fib_levels['Fib_38.2']
                    elif current_price < fib_levels['Fib_38.2'] and current_high > fib_levels['Fib_38.2']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_50.0']
                        take_profit = fib_levels['Fib_23.6']
                    elif current_price < fib_levels['Fib_23.6'] and current_high > fib_levels['Fib_23.6']:
                        signal = "sell"
                        stop_loss = fib_levels['Fib_38.2']
                        take_profit = fib_levels['Fib_23.6']  + (min(atr_value,750) * 2)
                    else:
                        signal = "neutral"
                        stop_loss = None
                        take_profit = None
                else:
                    signal = "neutral"
                    stop_loss = None
                    take_profit = None

    	


        # # Buy conditions
        # buy_signal_1 = (
        #     (current_price < fib_levels['Fib_78.6']) & 
        #     (rsi_value > 50) & 
        #     near_hvn_zone &
        #     (adx_value > 15) & 
        #     in_uptrend     
        # )
        # buy_signal_2 = (
        #     (current_price < fib_levels['Fib_61.8']) & 
        #     (rsi_value > 50) & 
        #     near_hvn_zone &
        #     (adx_value > 15) & 
        #     in_uptrend
        # )

        # # Sell conditions
        # sell_signal_2 = (
        #     (current_price > fib_levels['Fib_78.6']) & 
        #     (rsi_value < 50) & 
        #     near_lvn_zone &
        #     (adx_value > 15) & 
        #     in_downtrend 
        
        # )
        
        # sell_signal_1 = (
        #     (current_price > fib_levels['Fib_61.8']) & 
        #     (rsi_value < 50) & 
        #     near_lvn_zone &
        #     (adx_value > 15) & 
        #     in_downtrend 
            
        # )
        
        # # Breakout conditions for trend-following trades
        # breakout_long = (
        #     (current_price > fib_levels['Fib_61.8']) & 
        #     (rsi_value > 50) & 
        #     (relative_volume > 1.3) & 
        #     (adx_value > 15) & 
        #     trend == "uptrend"
        # )
        # breakout_short = (
        #     (current_price < fib_levels['Fib_61.8']) & 
        #     (rsi_value < 50) & 
        #     (relative_volume > 1.3) & 
        #     (adx_value > 15) &
        #     trend == "downtrend"
        # )

        # # Determine final signal
        # if signal == "buy":
        #     stop_loss = stop_loss
        #     take_profit = take_profit

        # elif buy_signal_1 or buy_signal_2 or breakout_long:
        #     signal = "buy"
        #     stop_loss = fib_levels["Fib_100.0"] if current_price < fib_levels['Fib_78.6'] else fib_levels['Fib_78.6'] 
        #     take_profit = fib_levels['Fib_61.8'] if current_price < fib_levels['Fib_78.6'] else fib_levels['Fib_38.2']
        # elif sell_signal_1 or sell_signal_2 or breakout_short:
        #     signal = "sell"
        #     stop_loss = fib_levels["Fib_100.0"] if current_price > fib_levels['Fib_78.6'] else fib_levels['Fib_78.6']
        #     take_profit = fib_levels['Fib_61.8'] if current_price > fib_levels['Fib_78.6'] else fib_levels['Fib_38.2']
        # else:
        #     signal = "neutral"
        #     stop_loss = None
        #     take_profit = None

        return signal, atr_value, stop_loss, take_profit

    def volume_strategy_support_resistance(self, coin, fiat, interval, trend_interval, timestamp_input_interval_adj, timestamp_input_actual, logger=None, simulate_live_data=False, verbose=False):
        """
        EMA Cross strategy
        #uses tm1 timestamp to check if values crossed on hourly and todays timestamps for multi timeframe strategy

        if trend interval is not specified the function does not use multitimeframe approach
        """

        price_data, minute_data = self.kucoin_price_loader.download_data(coin, fiat, interval, start_date="2017-10-05 00:00:00", end_date=timestamp_input_actual, 
                                                            use_local_timezone=True, drop_not_complete_timestamps=False, simulate_live_data=simulate_live_data, overwrite_file=False)   #use timestamp input and t_1 when you want to use uncompleted current timestep values
        

        atr = self.TA_creator.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval, 
                                        timestamp_input=timestamp_input_interval_adj, indicator_name='atr', overwrite_file=True, plot=False, ohlc="close", 
                                        atr={'period': 14})

        # volume_ema = self.TA_creator.calculate_indicator("usdt_kucoin", coin=coin, fiat=fiat, interval=interval, 
        #                                 timestamp_input=timestamp_input_interval_adj, indicator_name='volume_ema', overwrite_file=True, plot=False, ohlc="volume", 
        #                                 volume_ema={'span': 10})

        resistance = price_data['high'].rolling(window=20).max()
        support = price_data['low'].rolling(window=20).min()

        rel_volume = price_data['volume'] / price_data['volume'].rolling(window=10).mean()


        atr.index = pd.to_datetime(atr.index)
        resistance.index = pd.to_datetime(resistance.index)
        support.index = pd.to_datetime(support.index)
        rel_volume.index = pd.to_datetime(rel_volume.index)

    
        if isinstance(timestamp_input_interval_adj, dt.datetime):
            utc_time = pd.Timestamp(timestamp_input_interval_adj)
        
        if not np.all(price_data.localized):
            utc_time = utc_time.tz_localize('Europe/Berlin', ambiguous=True).tz_convert('UTC')
            utc_time = utc_time.replace(tzinfo=None)
        
        if trend_interval == '1d':
            utc_time = utc_time.tz_localize('Europe/Berlin', ambiguous=True).tz_convert('UTC')
            utc_time = utc_time.replace(tzinfo=None)
            utc_time = utc_time.replace(hour=0, minute=0, second=0, microsecond=0)
        elif trend_interval == '1h':
            utc_time = utc_time.replace(tzinfo=None)
            utc_time = utc_time.replace(minute=0, second=0, microsecond=0) #- timedelta(hours=1) #Using tm1 as hour is not completed yet       uncomment to get data
        elif trend_interval == '15m':
            utc_time = utc_time.replace(tzinfo=None)      
            utc_time = utc_time.replace(second=0, microsecond=0)
            utc_time = utc_time.replace(minute=15 * (utc_time.minute // 15)) # - timedelta(minutes=15) #Using tm1 as 15m is not completed yet       uncomment to get data
        elif trend_interval == '5m':
            utc_time = utc_time.replace(tzinfo=None)      
            utc_time = utc_time.replace(second=0, microsecond=0)
            utc_time = utc_time.replace(minute=5 * (utc_time.minute // 5))
    
        if utc_time not in ma_low.index or utc_time not in ma_high.index:
            # self.logger.info(f"Timestamp {timestamp_input} not in DataFrame.")
            current_trend = "neutral"
            return current_trend
        
            
        last_used_minute_data = minute_data.loc[minute_data['time'] <= timestamp_input_actual, 'time'].iloc[-1]
        minute_data = minute_data.set_index('time') 

        last_available_timestep = price_data.index[-2]
        buy_tm1_price = max(price_data.loc[last_available_timestep,"low"],price_data.loc[last_available_timestep,"open"]) if price_data.loc[last_available_timestep, "close"] > price_data.loc[last_available_timestep, "open"] else max(price_data.loc[last_available_timestep,"low"],price_data.loc[last_available_timestep,"close"])
        sell_tm1_price = min(price_data.loc[last_available_timestep,"high"],price_data.loc[last_available_timestep,"close"]) if price_data.loc[last_available_timestep, "close"] > price_data.loc[last_available_timestep, "open"] else min(price_data.loc[last_available_timestep,"high"],price_data.loc[last_available_timestep,"open"])
        prev_price = price_data.loc[last_available_timestep,'close']


        if verbose:
            # self.logger.info(f"{'#'*1} current trend is {current_trend} {'#'*1} with macd {macd.loc[utc_time, 'macd']:.2f} and signal {macd.loc[utc_time, 'signal']:.2f} at time {utc_time}")
            self.logger.info(f"{'#'*1} Current interval time is {timestamp_input_interval_adj} with current interval price at {price_data.loc[timestamp_input_interval_adj, 'close']:.2f}. {'#'*1} ")
            
            if simulate_live_data:
                try:
                    self.logger.info(f"{'#'*1} Current  actual time is {timestamp_input_actual} with current price at {minute_data.loc[last_used_minute_data, 'close']:.2f}. {'#'*1} ")
                except:
                    pass
            self.logger.info(f"{'#'*1} Previous time is {last_available_timestep} with previous price of {prev_price}. {'#'*1} ")
            self.logger.info(f"{'#'*1} Current high_ma is {ma_high.loc[timestamp_input_interval_adj, 'high_ma']:.2f} and current low_ma is {ma_low.loc[timestamp_input_interval_adj, 'low_ma']:.2f}    {'#'*1} ")
                

        price_buy_zone = ma_high.loc[timestamp_input_interval_adj, 'high_ma'] * 0.995 
        price_sell_zone = ma_low.loc[timestamp_input_interval_adj, 'low_ma'] * 1.005
        
        price_cond_buy_1 =   price_data.loc[timestamp_input_interval_adj, "close"] < ma_high.loc[timestamp_input_interval_adj, 'high_ma'] and buy_tm1_price >= ma_high.loc[timestamp_input_interval_adj, 'high_ma']    #price under high ma & prev  low above ma
        # price_cond_buy_2 = price_data.loc[timestamp_input_interval_adj, "close"] > ema_long.loc[timestamp_input_interval_adj, 'long_ema'] and price_data.loc[timestamp_input_interval_adj, "close"] >= vwma.loc[timestamp_input_interval_adj, 'vwma']  #price over long ema & prev high and low any below long ema
        price_cond_buy_2 = rel_volume > 1.5
        price_cond_buy_zone = price_data.loc[timestamp_input_interval_adj, "close"] > price_buy_zone
        
        price_cond_sell_1 = price_data.loc[timestamp_input_interval_adj, "close"] > ma_low.loc[timestamp_input_interval_adj, 'low_ma'] and sell_tm1_price <= ma_low.loc[timestamp_input_interval_adj, 'low_ma'] #price above low ma & prev high below ma
        # price_cond_sell_2 = price_data.loc[timestamp_input_interval_adj, "close"] < ema_long.loc[timestamp_input_interval_adj, 'long_ema'] and price_data.loc[timestamp_input_interval_adj, "close"] <= vwma.loc[timestamp_input_interval_adj, 'vwma'] #price below long ema & prev high and low any above long ema
        price_cond_sell_2 = rel_volume > 1.5
        price_cond_sell_zone = price_data.loc[timestamp_input_interval_adj, "close"] < price_sell_zone
        
        signal_buy = [price_cond_buy_1, price_cond_buy_2, price_cond_buy_zone]

        signal_sell = [price_cond_sell_1, price_cond_sell_2, price_cond_sell_zone] 


        #get signal 
        if np.all(signal_buy):  #  and current_trend == "bullish":
            signal = "buy"
        elif np.all(signal_sell):    #   and current_trend == "bearish"  
            signal  = "sell"
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
        elif interval == '15m':
            t_1 = timestamp_input - pd.Timedelta(minutes=15)
        elif interval == '5m':
            t_1 = timestamp_input - pd.Timedelta(minutes=5)
        
        metric_name = "ema"
        path = os.path.join(data_path_crypto, metric_name, interval)
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
        path = os.path.join(data_path_crypto, metric_name, interval)
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




    