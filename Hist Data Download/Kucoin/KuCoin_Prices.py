import os
import json
import pandas as pd
import numpy as np 
import kucoin
import sys
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import websocket
import threading
import json
from pathlib import Path
from io import StringIO
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

# base_path = os.path.dirname(os.path.realpath(__file__))
base_path = os.path.dirname(os.path.abspath(__file__))

if env == 'wsl':
    # crypto_bot_path = "/mnt/c/Users/mauri/Documents/Trading Bot/Python/AlgoTrader" 
    crypto_bot_path = os.path.join(os.path.dirname(base_path))
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
                AWS_path, GOOGLE_path,  histo_data_path, 
                data_loader, hist_data_download_kucoin, strategy_path]

# Add paths to sys.path and verify
for path in paths_to_add:
    if path not in sys.path:
        sys.path.append(path)

import mo_utils as utils
from utilsAWS import S3Utility
from utilsGoogleDrive import GoogleDriveUtility

#################################################################################################################################################################
#
#
#                                                                  KUCOIN Historical Data CLASS
#
#################################################################################################################################################################


class KuCoinDataDownloader:
    def __init__(self, created_logger=None):
        #config
        config_path = utils.find_config_path() 
        config = utils.read_config_file(os.path.join(config_path,"AlgoTrader_config.ini"))
        
        # self.key = config["AWS"]["access_key"]
        # self.sec_key = config["AWS"]["secret_access_key"]
        # self.bucket = config["AWS"]["bucket_name"]
        # self.arn_role = config["AWS"]["arn_role"]
        # self.region_name = config["AWS"]["region_name"]
        
        self.logger = created_logger if created_logger is not None else logger
        self.configure_logger()
        
        self.logger.info("KuCoinDataDownloader initialized")
        
    def configure_logger(self):
        
        # logger_path = utils.find_logging_path()

        #logger
        current_datetime = dt.datetime.now()
        timestamp = current_datetime.strftime('%d-%m-%Y_%H-%M')
        log_directory = "Kucoin Price Loader" 
        log_file_name = f"kucoin_price_loader_log_{timestamp}.txt"
        log_file_path = os.path.join(logging_path, log_directory, log_file_name)
        
        self.logger.add(log_file_path, rotation="500 MB", level="INFO")
        
    
        
    
    def get_interval_seconds(self, interval):
        interval_mapping = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '2h': 7200,
            '4h': 14400,
            '6h': 21600,
            '8h': 28800,
            '12h': 43200,
            '1d': 86400,
            '1w': 604800
        }
        return interval_mapping[interval]
    
    def get_interval_mapping(self, interval):
        interval_mapping = {
            "1m":'1min',
            "5m":'5min',
            "15m":'15min',
            '30m':'30min',
            "1h":"1hour",
            '2h':'2hour',
            '4h':'4hour',
            '6h':'6hour',
            '8h':'8hour',
            '12h':'12hour',
            '1d':'1day',
            '1w':'1week'
        }
        return interval_mapping[interval]


    def download_data(self, coin, fiat, interval, start_date=None, end_date=None, use_local_timezone=True,use_exchange_timezone=False, drop_not_complete_timestamps=True, simulate_live_data=False, overwrite_file=True, verbose=False):
        """Download historical price data for a specific coin/fiat pair from KuCoin.
        Args:
            coin (str): crypto to trade e.g. BTC
            fiat (str): fiat to exchange against e.g. USDT
            interval (str): interval of the data e.g. 1h
            start_date (datetime): start date of the data to download. Defaults to None.
            end_date (datetime): end date of the data to download. Defaults to None.
            use_local_timezone (bool): True if you want to convert times to your time. Defaults to True.
            use_exchange_timezone (bool): True if you want to convert times to your time. Defaults to False.
            drop_not_complete_timestamps (bool): True if you want to delete timestamps that are not completed yet. E.g. if now is 13:05 timestamp 13 is not completed. Defaults to True.
            simulate_live_data (bool, optional): True if you are downloading incomplete timestamps and want to replace the current interval value with the current minute close. Defaults to False.

        Returns:
            _type_: _description_
        """
        
        filename = f'price_usdt_kucoin_{coin}_{fiat}_{interval}.csv'
        path = os.path.join(histo_data_path, "price_usdt_kucoin",interval, filename)
        
        symbol = coin +"-"+ fiat
        interval_seconds = self.get_interval_seconds(interval)
        interval_mapping = self.get_interval_mapping(interval)
        
        if start_date is not None:      #adjust start date to timestamp
            start_date_unix = int(self.convert_datetime_to_timestamp(start_date) / 1000)
        elif start_date is None:
            start_date = "2017-10-07 05:00:00"     # if no start date is given, start from the first available date
            start_date_unix = int(self.convert_datetime_to_timestamp(start_date) / 1000)
        
        if end_date is not None:    #adjust end date to timestamp
            end_date_unix = int(self.convert_datetime_to_timestamp(end_date) / 1000)
        else:
            end_date_unix = int(time.time())              # if no end date is given, end at the current date
            end_date_request = self.convert_timestamp_to_datetime(end_date_unix*1000)
            end_date_request = datetime.strptime(end_date_request, '%Y-%m-%d %H:%M:%S')
        
        if not os.path.exists(os.path.join(histo_data_path, "price_usdt_kucoin",interval)):   
            os.makedirs(os.path.join(histo_data_path, "price_usdt_kucoin",interval))
             
        if os.path.exists(path) and not overwrite_file:       
            df_existing = pd.read_csv(path)
              # Convert 'time' column to datetime
            time_df = pd.to_datetime(df_existing.time)
            
            start_date_str = start_date.strftime(format='%Y-%m-%d %H:%M:%S') if isinstance(start_date, pd.Timestamp) else start_date

            if df_existing['time'][0] > start_date_str:     #if start date is before the first existing date download from start date, intentionally set default start date to 5th of October 2017 since this is first available point and we do not want to always download 
                
                startAt = start_date_unix
            else:
                if interval == '1h':
                    tm1 = time_df.iloc[-1] - timedelta(hours=2)  #redownloading fot the past two hours to update non complete value
                elif interval == '1d':
                    tm1 = time_df.iloc[-1] - timedelta(days=2)   #redownloading fot the past two days to update non complete value
                elif interval == '5m':
                    tm1 = time_df.iloc[-1] - timedelta(minutes=10)
                elif interval == '15m':
                    tm1 = time_df.iloc[-1] - timedelta(minutes=30)
                elif interval == "1m":
                    tm1 = time_df.iloc[-1] - timedelta(minutes=2)

                startAt = int(tm1.timestamp())
            
                if df_existing["localized"].iloc[-1] == True:
                    startAt = int(tm1.tz_localize('Europe/Berlin', ambiguous=True).tz_convert('UTC').timestamp())

        else:
            df_existing = pd.DataFrame()         #if overwrite is True or file does not exist, create empty dataframe
            startAt = start_date_unix


        data = []
        endAt = end_date_unix # Current timestamp
        endAt +=1
        
        if interval == '1h':
            time_now = dt.datetime.now().replace(minute=0,second=0, microsecond=0) if end_date is None else end_date.replace(minute=0,second=0, microsecond=0)
            time_tm1 = time_now - timedelta(hours=1)
        elif interval == "1d":
            time_now = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) if end_date is None else end_date.replace(hour=0, minute=0, second=0, microsecond=0)
            time_tm1 = time_now - timedelta(days=1)
        elif interval == "5m":
            time_now = dt.datetime.now().replace(minute=(dt.datetime.now().minute//5)*5, second=0, microsecond=0) if end_date is None else end_date.replace(minute=(end_date.minute//5)*5, second=0, microsecond=0)
            time_tm1 = time_now - timedelta(minutes=5)
        elif interval == "15m":
            time_now = dt.datetime.now().replace(minute=(dt.datetime.now().minute//15)*15, second=0, microsecond=0)    if end_date is None else end_date.replace(minute=(end_date.minute//15)*15, second=0, microsecond=0)
            time_tm1 = time_now - timedelta(minutes=15)
        elif interval == "1m":
            time_now = dt.datetime.now().replace(second=0, microsecond=0) if end_date is None else end_date.replace(second=0, microsecond=0)
            time_tm1 = time_now - timedelta(minutes=1)

        data_in_df_existing = False

        if not df_existing.empty:
            time_now_str = time_now.strftime(format='%Y-%m-%d %H:%M:%S') if isinstance(time_now, pd.Timestamp) else time_now
            tm1_date_str = time_tm1.strftime(format='%Y-%m-%d %H:%M:%S') if isinstance(time_tm1, pd.Timestamp) else time_tm1

            
            if time_now_str in df_existing['time'].values:  #check if df_existing already has end date to avoid redownloading
                df_updated = df_existing.copy(deep=True)
                df_updated = df_updated[df_updated['time'] <= time_now_str] if time_now is not None else df_updated
                data_in_df_existing = True
            # elif end_date_str not in df_existing['time'].values:
            #     last_aval_date = pd.Timestamp(df_existing[df_existing['time']<=end_date_str]["time"].iloc[-1])
            #     if pd.Timedelta(pd.Timestamp(end_date_str) - last_aval_date).days <= 5:   #if end date is within 5 days of last available date, download only the missing data
            #         df_updated = df_existing.copy(deep=True)
            #         df_updated = df_updated[df_updated['time'] <= end_date_str] if end_date is not None else df_updated
            #         data_in_df_existing = True
        
        if df_existing.empty or not data_in_df_existing:
            
            if startAt < endAt:
                
                if verbose:
                    self.logger.info(f"Downloading data from {self.convert_timestamp_to_datetime(startAt*1000)} until {self.convert_timestamp_to_datetime(endAt*1000)}")
                
                batch_size = 1500  # Number of datapoints per batch
                batches = list(range(startAt, endAt, batch_size * interval_seconds))

                # Ensure the last batch is included
                if batches[-1] + batch_size * interval_seconds < endAt:
                    batches.append(batches[-1] + batch_size * interval_seconds)
                
                # n_retries = 3
                
                while len(data)==0:
                    try:
                        with ThreadPoolExecutor() as executor:
                            futures = []
                            for batch in batches:
                                batchEnd = min(batch+batch_size*interval_seconds,endAt)
                                future = executor.submit(self.download_batch, batch, batchEnd, symbol, interval_mapping)
                                futures.append(future)
                                time.sleep(0.1)  # Respect the rate limit

                            #for future in tqdm(as_completed(futures), total=len(futures), unit='batch'):
                            for future in as_completed(futures):
                                batch_data = future.result()
                                data.extend(batch_data)
            
                        break
                    except Exception as e:
                        self.logger.error(f"Error downloading data: {e}. Retrying...")
                        n_retries += 1
                        
                # Assuming 'data' is populated as per the previous logic
                df_new = pd.DataFrame(data, columns=['time', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                df_new['time'] = pd.to_datetime(df_new['time'].astype(int), unit='s')
                df_new['localized'] = False
                df_new.sort_values('time', inplace=True)
                
                
                if interval not in ["1d", "1w"]:
                    if use_local_timezone:  
                        if not df_new.empty:
                            if np.logical_and(df_new["localized"].iloc[-1]== True, df_new["time"].tolist()[-1] == time_tm1): #use minus two because the last closing value is always tm1 and current value is just actual snapshot
                                pass
                            else:
                                df_new["time"] = df_new["time"].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin').dt.strftime('%Y-%m-%d %H:%M:%S')
                                df_new["time"] = pd.to_datetime(df_new["time"])
                                df_new["localized"] = True
                        else:
                            pass
                else:
                    df_new["localized"] = False
                
                if not df_existing.empty:
                    cols_to_use = df_existing.columns.union(df_new.columns, sort=False)
                    df_existing = df_existing.dropna(axis=1, how='all').reindex(columns=cols_to_use)
                    df_new = df_new.dropna(axis=1, how='all').reindex(columns=cols_to_use)

                df_updated = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset='time', keep='last')
                df_updated['time'] = pd.to_datetime(df_updated['time'])  # Convert 'time' column to datetime
                df_updated.sort_values('time', inplace=True)
                end_date_str = end_date.strftime(format='%Y-%m-%d %H:%M:%S') if isinstance(end_date, pd.Timestamp) else end_date
                df_updated = df_updated[df_updated['time'] <= end_date_str] if end_date is not None else df_updated
            else:
                df_updated = df_existing.copy(deep=True)
                end_date_str = end_date.strftime(format='%Y-%m-%d %H:%M:%S') if isinstance(end_date, pd.Timestamp) else end_date
                df_updated = df_updated[df_updated['time'] <= end_date_str] if end_date is not None else df_updated
        


        minute_df = None
        minute_data = None
        
        # Simulate live data: replace last hourly data with minute data if needed
        if simulate_live_data:
            filename_minute_data = f'price_usdt_kucoin_{coin}_{fiat}_1m.csv'
            path_minute_data = os.path.join(histo_data_path, "price_usdt_kucoin","1m", filename_minute_data)

            if os.path.exists(path_minute_data):
                minute_df = pd.read_csv(path_minute_data)
                minute_df['time'] = pd.to_datetime(minute_df['time'])
                existing_minute_df = minute_df.copy(deep=True)

            current_minute = int(self.convert_datetime_to_timestamp(end_date+pd.Timedelta(minutes=2)) / 1000) 
            last_hour_time = end_date.replace(minute=0, second=0, microsecond=0) 
            last_hour_time = pd.Timestamp(last_hour_time).floor('h')
            last_hour_time = pd.to_datetime(last_hour_time)
            volume_hour = last_hour_time

            if use_local_timezone:
                last_hour_time = last_hour_time.tz_localize('Europe/Berlin', ambiguous=True).tz_convert('UTC').tz_localize(None)
                
                # Ensure end_date is timezone-naive
                if end_date.tzinfo is not None:
                    end_date = end_date.tz_localize(None)
                
            last_hour_time_m5 = last_hour_time - pd.Timedelta(hours=1)
            last_hour_time_unix = int(self.convert_datetime_to_timestamp(last_hour_time_m5) / 1000)

            if interval == '1h':
                last_timestamp = end_date.replace(minute=0, second=0, microsecond=0)
            elif interval == '1d':
                last_timestamp = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
            elif interval == '5m':
                last_timestamp = end_date.replace(minute=(end_date.minute//5)*5, second=0, microsecond=0)
            elif interval == '15m':
                last_timestamp = end_date.replace(minute=(end_date.minute//15)*15, second=0, microsecond=0)
            
            # end_date_str = end_date.strftime(format='%Y-%m-%d %H:%M:%S') if isinstance(end_date, pd.Timestamp) else end_date
            
            if minute_df is not None and end_date in minute_df['time'].values:
                minute_data = minute_df[minute_df['time'] <= end_date]
                minute_data_existing = True
            else:
                minute_interval_mapping = self.get_interval_mapping('1m')
                interval_seconds_sim = self.get_interval_seconds("1m")
                
                batch_size = 1500  # Number of datapoints per batch
                batch_starts = list(range(last_hour_time_unix, current_minute, batch_size * interval_seconds_sim))
                minute_data = []
                
                # Ensure the last batch is included
                if batch_starts[-1] + batch_size * interval_seconds < current_minute:
                    batch_starts.append(batch_starts[-1] + batch_size * interval_seconds)
                
                while len(minute_data)==0:    
                    try:
                        with ThreadPoolExecutor() as executor:
                            futures = []
                            for batch_start in batch_starts:
                                batch_end = min(batch_start + batch_size * interval_seconds_sim, current_minute)
                                future = executor.submit(self.download_batch, batch_start, batch_end, symbol, minute_interval_mapping)
                                futures.append(future)
                                time.sleep(0.1)  # Respect the rate limit

                            #for future in tqdm(as_completed(futures), total=len(futures), unit='batch'):
                            for future in as_completed(futures):
                                batch_data = future.result()
                                minute_data.extend(batch_data)
                        break
                    except Exception as e:
                        self.logger.error(f"Error downloading data: {e}. Retrying...")
                        n_retries += 1
                
            if len(minute_data) > 0 :
                if not minute_data_existing: 
                    minute_df = pd.DataFrame(minute_data, columns=['time', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                    minute_df['time'] = pd.to_datetime(minute_df['time'].astype(int), unit='s')
                    minute_df['localized'] = False
                    # df_new['localized'] = False   
                    minute_df.sort_values('time', inplace=True)
                    
                    if use_local_timezone:
                        # Convert 'time' column to the specified timezone for rows where 'localized' is False
                        mask = minute_df["localized"] == False
                        minute_df.loc[mask, "time"] = minute_df.loc[mask, "time"].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin').dt.strftime('%Y-%m-%d %H:%M:%S')
                        minute_df.loc[mask, "time"] = pd.to_datetime(minute_df.loc[mask, "time"])
                        minute_df.loc[mask, "localized"] = True

                if interval == '1h':
                    df_updated['time'] = pd.to_datetime(df_updated['time']).dt.floor('h')  # Floor to the nearest hour
                elif interval == '1d':
                    df_updated['time'] = pd.to_datetime(df_updated['time']).dt.floor('d')
                elif interval == '5m':
                    df_updated['time'] = pd.to_datetime(df_updated['time']).dt.floor('5min')
                elif interval == '15m':
                    df_updated['time'] = pd.to_datetime(df_updated['time']).dt.floor('15min')
                    
                # Replace the last hourly close with the last minute close
                if not minute_df[minute_df["time"] == end_date].empty:
    
                    minute_close = float(minute_df[minute_df["time"] == end_date]['close'].iloc[0]) if 'close' in minute_df.columns else 0
                    minute_high = minute_df[(minute_df.time >= volume_hour) & (minute_df.time <= end_date)]["high"].astype(float).max()
                    minute_low = minute_df[(minute_df.time >= volume_hour) & (minute_df.time <= end_date)]["low"].astype(float).min()
                    minute_turnover = minute_df[(minute_df.time >= volume_hour) & (minute_df.time <= end_date)]["turnover"].astype(float).sum()
                    minute_volume = minute_df[(minute_df.time >= volume_hour) & (minute_df.time <= end_date)]["volume"].astype(float).sum()

                    if last_timestamp in df_updated['time'].values:
                        df_updated.loc[df_updated['time'] == last_timestamp, 'close'] = minute_close 
                        df_updated.loc[df_updated['time'] == last_timestamp, 'high'] = minute_high 
                        df_updated.loc[df_updated['time'] == last_timestamp, 'low'] = minute_low
                        df_updated.loc[df_updated['time'] == last_timestamp, 'turnover'] = minute_turnover 
                        df_updated.loc[df_updated['time'] == last_timestamp, 'volume'] = minute_volume 
                    else:
                        new_row = { 
                            'time': last_timestamp,
                            'open': df_updated.loc[df_updated['time'] == last_timestamp, 'open'].iloc[0],
                            'close': minute_close,
                            'high': minute_high,
                            'low': minute_low,    #exclude volume as we need aggregate vol
                            'turnover': minute_turnover
                        }
                        df_updated = pd.concat([df_updated, pd.DataFrame([new_row])], ignore_index=True).drop_duplicates(subset='time', keep='last')
                        df_updated.sort_values('time', inplace=True)
                    
                    if verbose:
                        self.logger.info(f"Replaced close at {last_timestamp} with minute close {minute_df[minute_df['time'] == end_date]['close'].iloc[0]} from minute {minute_df[minute_df['time'] == end_date]['time'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')}")
                
    
                elif minute_df[minute_df["time"] == end_date].empty:
                    
                    # Adjust timestamp to last available minute timestamp
                    try:
                        last_available_timestamp = minute_df.loc[minute_df['time'] <= end_date, 'time'].iloc[-1]
                    except IndexError:
                        last_available_timestamp = minute_df.loc[minute_df['time'] > end_date, 'time'].iloc[1]

                    minute_close = float(minute_df[minute_df["time"] == last_available_timestamp]['close'].iloc[0]) if 'close' in minute_df.columns else 0
                    minute_high = minute_df[(minute_df.time >= volume_hour) & (minute_df.time <= end_date)]["high"].astype(float).max()
                    minute_low = minute_df[(minute_df.time >= volume_hour) & (minute_df.time <= end_date)]["low"].astype(float).min()
                    minute_turnover = minute_df[(minute_df.time >= volume_hour) & (minute_df.time <= end_date)]["turnover"].astype(float).sum()
                    minute_volume = minute_df[(minute_df.time >= volume_hour) & (minute_df.time <= end_date)]["volume"].astype(float).sum()

                    if last_timestamp in df_updated['time'].values:
                        df_updated.loc[df_updated['time'] == last_timestamp, 'close'] = minute_close 
                        df_updated.loc[df_updated['time'] == last_timestamp, 'high'] = minute_high 
                        df_updated.loc[df_updated['time'] == last_timestamp, 'low'] = minute_low
                        df_updated.loc[df_updated['time'] == last_timestamp, 'turnover'] = minute_turnover 
                        df_updated.loc[df_updated['time'] == last_timestamp, 'volume'] = minute_volume 
                    else:
                        new_row = {
                            'time': last_timestamp,
                            'open': df_updated.loc[df_updated['time'] == last_timestamp, 'open'].iloc[0],
                            'close': minute_close,
                            'high': minute_high,
                            'low': minute_low, #exclude volume as we need aggregate vol
                            'turnover': minute_turnover
                        }
                        df_updated = pd.concat([df_updated, pd.DataFrame([new_row])], ignore_index=True).drop_duplicates(subset='time', keep='last')
                        df_updated.sort_values('time', inplace=True)
                    
                    if verbose:
                        self.logger.info(f"Replaced close at {last_timestamp} with minute close {minute_df[minute_df['time'] == last_available_timestamp]['close'].iloc[0]} from minute {minute_df[minute_df['time'] == last_available_timestamp]['time'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')}")
                
                df_updated = df_updated[df_updated['time'] <= last_timestamp]   
                
                if not minute_data_existing:    #if 
                    minute_df = pd.concat([existing_minute_df, minute_df], ignore_index=True).drop_duplicates(subset='time', keep='last')
                    minute_df.to_csv(path_minute_data, index=False)
                    self.logger.info(f"Minute data updated and saved as {filename_minute_data}")

        if interval not in ["1d", "1w"]: 
            # Check if all values in the 'localized' column are True
            if use_local_timezone:
                if not df_updated["localized"].all():
                    # Convert 'time' column to the specified timezone for rows where 'localized' is False
                    mask = df_updated["localized"] == False
                    df_updated.loc[mask, "time"] = df_updated.loc[mask, "time"].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin').dt.strftime('%Y-%m-%d %H:%M:%S')
                    df_updated.loc[mask, "time"] = pd.to_datetime(df_updated.loc[mask, "time"])
                    df_updated.loc[mask, "localized"] = True
        
        if use_exchange_timezone:
            if interval not in ["1d", "1w"]: 
                # Convert 'time' column back to the original timezone (UTC) for all rows
                df_updated["time"] = df_updated["time"].dt.tz_convert('UTC').dt.strftime('%Y-%m-%d %H:%M:%S')
                df_updated["time"] = pd.to_datetime(df_updated["time"])
                df_updated["localized"] = False
            
        numerical_cols = ['open', 'close', 'high', 'low', 'volume']
        turnover_col = ['turnover']
        decimals = 4  # Number of decimal places

        for col in numerical_cols + turnover_col:
            try:                
                df_updated.loc[:,col] = df_updated[col].astype(float).copy(deep=True) #df_updated[col].apply(clean_numeric_value)
                
                if simulate_live_data:
                    minute_df.loc[:,col] = minute_df[col].astype(float).copy(deep=True) #minute_df[col].apply(clean_numeric_value)
            
            except ValueError:
                df_updated[col] = pd.np.nan
                
        # Round the values and convert to integers for the numerical columns
        df_updated.loc[:,numerical_cols] = (df_updated.loc[:,numerical_cols] * 10**decimals).round().astype(int)
        df_updated.loc[:,numerical_cols] = df_updated.loc[:,numerical_cols] / (10**decimals)
        
        if simulate_live_data:
            minute_df.loc[:,numerical_cols] = (minute_df.loc[:,numerical_cols] * 10**decimals).round().astype(int)
            minute_df.loc[:,numerical_cols] = minute_df.loc[:,numerical_cols] / (10**decimals)
        
        # Optional: Remove rows with missing values
        df_updated = df_updated.copy(deep=True).dropna(subset=numerical_cols + turnover_col)

        if drop_not_complete_timestamps:
            df_updated = df_updated[df_updated["time"] != time_now]

        # if use_local_timezone and not data_in_df_existing:
        #     berlin_timezone = pytz.timezone('Europe/Berlin')
        #     time_now = time_now.tz_localize('UTC').astimezone(berlin_timezone).strftime('%Y-%m-%d %H:%M:%S')
        #     time_now = pd.to_datetime(time_now)

        df_updated.loc[:,"time"] = pd.to_datetime(df_updated["time"])
        df_updated = df_updated.set_index('time')
        
        if df_updated.index.duplicated().any():
            df_updated = df_updated[~df_updated.index.duplicated(keep='last')]

        df_updated = df_updated.loc[:time_now,:]
        df_updated = df_updated.reset_index()
        
        if simulate_live_data:
            if minute_df.index.duplicated().any():
                minute_df = minute_df[~minute_df.index.duplicated(keep='last')]

        # self.logger.info(f"{'#'*1} Kucoin price usdt downloaded until {df_updated['time'].iloc[-1]} for interval {interval} {'#'*1}")
        
        if not data_in_df_existing:
            df_updated.to_csv(path, index=False, date_format='%Y-%m-%d %H:%M:%S')  # Save datetime in a readable format

        # self.logger.info(f"Data updated and saved as {filename}")
        
        df_updated = df_updated.set_index('time')
        minute_df = minute_df[minute_df['time'] <= end_date] if simulate_live_data else None

        return df_updated, minute_df

    def download_batch(self, batch_start, batch_end, symbol, interval_mapping):
        url = 'https://api.kucoin.com/api/v1/market/candles'
        session = requests.Session()

        # Setup retries with exponential backoff
        retries = Retry(
            total=5,  # Total number of retries
            backoff_factor=0.5,  # Backoff factor for retry delays
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)

        try:
            response = session.get(
                url,
                params={
                    'symbol': symbol,
                    'type': interval_mapping,
                    'startAt': batch_start,
                    'endAt': batch_end
                },
                timeout=10  # Set a timeout to prevent hanging requests
            )
            response.raise_for_status()  # Raise an HTTPError for bad responses
            return response.json().get('data', [])

        except requests.exceptions.SSLError as ssl_err:
            self.logger.error(f'SSL error occurred: {ssl_err}. Retrying...')
            return []

        except requests.exceptions.RequestException as req_err:
            self.logger.error(f'Failed to retrieve data: {req_err}')
            return []

        finally:
            session.close()

    def convert_timestamp_to_datetime(self, timestamp):
        """
        REQ:        timestamp       timestamp in unix ms format

        Returns:    datetime:       datetime object of format "Y-m-d H:M" 
        """
        if isinstance(timestamp, pd.Series):
            # Apply the conversion to each element in the series
            return timestamp.apply(self.convert_timestamp_to_datetime)
        else:
            timestamp_sec = timestamp / 1000.0
            datetime_object = dt.datetime.fromtimestamp(timestamp_sec)

            # Format datetime object as "%Y-%m-%d %H:%M:%S"
            formatted_datetime = datetime_object.strftime("%Y-%m-%d %H:%M:%S")
        return formatted_datetime
        
    def convert_datetime_to_timestamp(self, datetime_input):
        """
        REQ:        datetime       datetime object of format "Y-m-d H:M" 

        Returns:    timestamp:       timestamp in unix ms format
        """
        if isinstance(datetime_input, pd.Series):
            # Apply the conversion to each element in the series
            return datetime_input.apply(self.convert_datetime_to_timestamp)
        elif isinstance(datetime_input, str):
            # Convert date string to datetime object
            datetime_object = dt.datetime.strptime(datetime_input, "%Y-%m-%d %H:%M:%S")
        elif isinstance(datetime_input, (dt.datetime, pd.Timestamp)):
            # Use datetime object directly
            datetime_object = datetime_input
        else:
            raise ValueError("Input must be a string, datetime object, or pandas Series")

        # Convert datetime object to Unix timestamp in milliseconds
        unix_timestamp_ms = int(datetime_object.timestamp() * 1000)
        return unix_timestamp_ms

#################################################################################################################################################################
#
#
#                                                                  BACKTEST
#
#################################################################################################################################################################


# # # Example usage
# coin = "BTC"
# fiat = "USDT"
# frequency = "1h"
# downloader = KuCoinDataDownloader()
# downloader.download_data(coin, fiat, frequency, '2018-01-01 00:00:00', '2024-07-16 19:00:00')



#################################################################################################################################################################
#
#
#                                                                  KUCOIN Get price
#
#################################################################################################################################################################


def get_current_price(symbol):
    response = requests.get('https://api.kucoin.com/api/v1/market/orderbook/level1', params={'symbol': symbol})
    if response.status_code == 200:
        data = response.json()['data']
        timestamp = datetime.fromtimestamp(int(data['time']) / 1000)
        iso_timestamp = timestamp.isoformat().replace('T', ' ').rsplit('.')[0]
        return iso_timestamp, float(data['price'])
    else:
        print('Failed to retrieve current price')
        return None

# print(get_current_price('BTC-USDT')[1])


#################################################################################################################################################################
#
#
#                                                                  KUCOIN WebSocket API CLASS
#
#################################################################################################################################################################


# class KucoinWebSocketClient:
#     def __init__(self, config_file='JansConfig.ini'):
#         """
#         initilize the client
#         """
#         self.config = configparser.ConfigParser()
#         base_dir = os.path.dirname(os.path.realpath(__file__))
#         config_path = os.path.join(base_dir,"..", "Config", config_file)
#         self.config.read(config_path)
#         self.api_key = self.config["kucoin_api_keys"]["api_key"]
#         self.api_secret = self.config["kucoin_api_keys"]["api_secret"]
#         self.api_passphrase = self.config["kucoin_api_keys"]["api_passphrase"]
#         self.spot_and_margin_url = "https://api.kucoin.com"
#         self.ws = None
#         self.token = None
#         self.instance_servers = None 
#         self.ping_interval = None  # recommended Interval for sending 'ping' to server to maintain the connection 
#         self.ping_timeout = None  # After such a long time(seconds), if you do not receive pong, it will be considered as disconnected.
#         self.last_ping_time = 0
#         self.shutting_down = False
#         self.reconnecting = False
#         self.latest_price_data = {}
#         self.data_lock = threading.Lock()
#         self.topics = {
#             "ticker": "/market/ticker:{}",  # get the specified [symbol] push of BBO changes; Push frequency: once every 100ms
#             "index": "/indicator/index:{}",  # get the mark price for margin trading.
#             "markPrice": "/indicator/markPrice:{}",  # get the index price for the margin trading.
#             # "positions": "/margin/position"
#             # add here endpoints of websockt URL's if needed   
#         }     
    
#     def get_server_timestamp(self):
#         response = requests.get(f"{self.spot_and_margin_url}/api/v1/timestamp")
#         if response.status_code == 200:
#             timestamp = response.json()["data"] / 1000
#             logger.info("Fetching server timestamp...")
#             return datetime.utcfromtimestamp(timestamp)
#         else:
#             raise Exception(f"Failed to fetch server time, status code: {response.status_code}")

#     def request_token(self, private=False):
#         url = f"{self.spot_and_margin_url}/api/v1/bullet-private" if private else f"{self.spot_and_margin_url}/api/v1/bullet-public"
#         if private:
#             now = int(time.time() * 1000)
#             str_to_sign = str(now) + 'POST' + '/api/v1/bullet-private'
#             signature = base64.b64encode(hmac.new(self.api_secret.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest())
#             api_passphrase_encoded = base64.b64encode(hmac.new(self.api_secret.encode('utf-8'), self.api_passphrase.encode('utf-8'), hashlib.sha256).digest())
#             headers = {
#                 "KC-API-SIGN": signature,
#                 "KC-API-TIMESTAMP": str(now),
#                 "KC-API-KEY": self.api_key,
#                 "KC-API-PASSPHRASE": api_passphrase_encoded,
#                 "KC-API-KEY-VERSION": "2"
#             }
#             response = requests.post(url, headers=headers)
#         else:
#             response = requests.post(url)

#         if response.status_code == 200:
#             self.token = response.json()['data']['token']
#             self.instance_servers = response.json()['data']['instanceServers'][0]
#             self.ping_interval = self.instance_servers['pingInterval'] / 1000  # Convert milliseconds to seconds
#             self.ping_timeout = self.instance_servers['pingTimeout'] / 1000  # Convert milliseconds to seconds
#             self.init_websocket(self.instance_servers['endpoint'])
#         else:
#             raise Exception(f"Token request failed: {response.status_code}")

#     def init_websocket(self, endpoint):
#         self.ws = websocket.WebSocketApp(f"{endpoint}?token={self.token}",
#                                          on_open=self.on_open,
#                                          on_message=self.on_message,
#                                          on_error=self.on_error,
#                                          on_close=self.on_close)
#         self.last_ping_time = time.time()
#         websocket_thread = threading.Thread(target=self.run_forever)
#         websocket_thread.start()
#         if time.time() - self.last_ping_time >= self.ping_interval:
#                 self.send_ping()
    
#     def run_forever(self):
#         self.ws.run_forever()
    
#     def on_open(self, ws):
#         self.last_ping_time = time.time()
#         threading.Thread(target=self.ping_thread).start()

#     def on_message(self, ws, message):
#         print(message)
#         logger.debug(f"{datetime.now()}: {message}")
#         message_data = json.loads(message)
        
#         if message_data.get('type') == "pong":
#             self.last_pong_time = time.time() * 1000

#         elif message_data.get('type') == 'message' and message_data.get('topic').startswith("/market/ticker:"):
#             self.ws_priceData(message_data)
       
#         current_time = time.time()
#         if current_time - self.last_ping_time >= self.ping_interval:
#             self.send_ping(ws)

#     def on_error(self, ws, error):  
#         logger.error(f"WebSocket error: {error}")
#         self.reconnect()
    
#     def on_close(self, ws, close_status_code, close_msg):
#         logger.info("WebSocket connection closed.")
#         if not self.shutting_down and not self.reconnecting:
#             self.reconnect()

#     def subscribe_to_topic(self, symbol, topic_type):
#         if not self.ws or not self.ws.sock or not self.ws.sock.connected:
#             logger.error("WebSocket is not connected.")
#             return
#         logger.info(f"Subscribing to topic: {topic_type} for symbol: {symbol}")
#         if not self.ws:
#             logger.warning(f"Trying to Subscribe to topic: {topic_type} for symbol: {symbol}, but but WebSocket is not initialized")
#             return

#         topic = self.topics.get(topic_type)
#         if not topic:
#             return
#         subscribe_message = json.dumps({
#             "id": str(int(time.time() * 1000)),
#             "type": "subscribe",
#             "topic": topic.format(symbol),
#             "response": True
#         })
#         self.ws.send(subscribe_message)

#     def unsubscribe_from_topic(self, symbol, topic_type):
#         logger.info(f"Unsubscribing to topic: {topic_type} for symbol: {symbol}")
#         if not self.ws:
#             logger.warning(f"Trying to unsubscribe from topic {topic_type} for symbol {symbol} but WebSocket is not initialized.")
#             return

#         topic = self.topics.get(topic_type)
#         if not topic:
#             print("Unknown topic type:", topic_type)
#             return

#         unsubscribe_message = json.dumps({
#             "id": str(int(time.time() * 1000)),
#             "type": "unsubscribe",
#             "topic": topic.format(symbol),
#             "response": True
#         })
#         self.ws.send(unsubscribe_message)

#     def ws_priceData(self, message_data):
#         symbol = message_data.get('topic').split(":")[-1]
#         data = message_data.get('data')
#         if symbol and data:
#             price_data = {
#                 "timestamp": data.get("time"), 
#                 "price": data.get("price")
#             }
#             self.update_latest_price_data(symbol, price_data)

#             timestamp_datetime = datetime.fromtimestamp(price_data["timestamp"] / 1000.0, tz=timezone.utc)
#             formatted_timestamp = timestamp_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

#     def update_latest_price_data(self, symbol, price_data):
#         with self.data_lock:
#             self.latest_price_data[symbol] = price_data

#     def get_latest_price_data(self, symbol):
#         with self.data_lock:
#             return self.latest_price_data.get(symbol, {"timestamp": None, "price": None})
    
#     def send_ping(self):
#         logging.DEBUG("Senden eines Ping")
#         if self.ws_connected():
#             self.last_ping_time = time.time()
#             ping_message = json.dumps({"id": str(self.last_ping_time), "type": "ping"})
#             self.ws.send(ping_message)
#         else:
#             logging.WARNING("try's to ping, but ws is not connected")

#     def check_pong_received(self):
#         if self.last_ping_time > self.last_pong_time:
#             logging.WARNING("Pong not recieved, starts reconnecting function...")
#             self.reconnect()

#     def ping_thread(self):
#         while not self.shutting_down:
#             time.sleep(self.ping_interval)
#             self.send_ping()

#     def reconnect(self):
#         if self.reconnecting:
#             return
#         self.reconnecting = True
#         try:
#             if self.ws:
#                 self.ws.close()
#             self.request_token(private=False)
#             self.reconnecting = False
#         except Exception as e:
#             logger.error(f"Reconnection failed: {e}")
#             self.reconnecting = False 

#     def ws_connected(self):
#         return self.ws is not None and self.ws.sock and self.ws.sock.connected

#     def close_connection(self):
#         if self.ws:
#             self.ws.close()

#     def shutdown_handler(self):
#         self.shutting_down = True
#         self.close_connection()


# class ColorFormatter(logging.Formatter):
#     '''
#     class for logging, includes colour's for better log vizualisation and two functions for converting datetime in ms format
#     '''
#     format = "%(asctime)s [%(levelname)s] %(message)s"
#     FORMATS = {
#         logging.DEBUG: "\033[94m" + format + "\033[0m",  # Blue
#         logging.INFO: "\033[92m" + format + "\033[0m",  # Green
#         logging.WARNING: "\033[93m" + format + "\033[0m",  # Gelb
#         logging.ERROR: "\033[91m" + format + "\033[0m",  # Red
#         logging.CRITICAL: "\033[91m\033[1m" + format + "\033[0m",  # Red + Bold
#     }

#     def formatTime(self, record, datefmt=None):
#         # Erzeugt ein datetime-Objekt aus dem timestamp
#         record_datetime = datetime.fromtimestamp(record.created)
#         # Formatierung des datetime-Objekts mit Millisekunden
#         if datefmt:
#             formatted_time = record_datetime.strftime(datefmt)
#         else:
#             formatted_time = record_datetime.strftime("%Y-%m-%d %H:%M:%S")
#         # Füge die Millisekunden hinzu
#         formatted_time_with_ms = f"{formatted_time}.{int(record.msecs):03d}"
#         return formatted_time_with_ms

#     def format(self, record):
#         log_fmt = self.FORMATS.get(record.levelno, self.format)
#         self._style._fmt = log_fmt
#         # Nutze die Basis-Klasse, um das Format festzulegen
#         formatter = logging.Formatter(self._style._fmt, datefmt='%Y-%m-%d %H:%M:%S')
#         # Setze das korrekte Datum und die Uhrzeit mit Millisekunden
#         record.asctime = self.formatTime(record)
#         return super().format(record)


# if __name__ == "__main__":
#     client = KucoinWebSocketClient("JansConfig.ini")  # Initialisierung des Clients
#     client.request_token(private=False)  # Anfrage eines Tokens
#     time.sleep(3)  # Kurze Pause, um die Initialisierung abzuschließen

#     symbol = "BTC-USDT"  # Symbol, für das Ticker-Daten abonniert werden sollen
#     client.subscribe_to_topic(symbol, "ticker")  # Abonnieren von Ticker-Daten

#     # Starte die kontinuierliche Aktualisierung
#     time.sleep(3)  # Warte einen Moment, um Daten zu erhalten


#     client.unsubscribe_from_topic(symbol, "ticker")  # Abonnement beenden
#     client.shutdown_handler()  # Schließen der Verbindung und Beenden





