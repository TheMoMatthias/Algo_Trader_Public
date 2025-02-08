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
from io import StringIO
from functools import wraps
# from data_download_entire_history  import *

def retry(max_retries=5, delay=2, backoff=2):
    """
    Decorator for retrying a function call with exponential backoff.
    Retries the function up to max_retries times. If it still fails,
    the exception is raised and passed through.

    :param max_retries: Maximum number of retries (default: 5).
    :param delay: Initial delay between retries (default: 2 seconds).
    :param backoff: Backoff multiplier for delay between retries (default: 2).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    print(f"Error occurred: {e}. Retrying {retries}/{max_retries} in {current_delay} seconds...")
                    if retries >= max_retries:
                        print("Max retries reached. Raising exception.")
                        raise  # Pass the exception after the retries are exhausted
                    time.sleep(current_delay)
                    current_delay *= backoff  # Exponential backoff
        return wrapper
    return decorator

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
    crypto_bot_path = os.path.expanduser("~/Documents/Trading Bot/Python/AlgoTrader")
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

# Add paths to sys.path and verify
import mo_utils as utils
from utilsGoogleDrive import GoogleDriveUtility
from DataLoader import DataLoader
# from TradeBacktester import Backtester
from StrategyEvaluator import StrategyEvaluator
from TAIndicatorCreator import TAIndicatorCreator
from KuCoin_Prices import KuCoinDataDownloader
from main_api_trader import MainApiTrader



class AlgoTrader(MainApiTrader, DataLoader):

    def __init__(self):
     
        #config
        config_path = utils.find_config_path() 
        config = utils.read_config_file(os.path.join(config_path,"AlgoTrader_config.ini"))
        
        if utils.get_config_value(config, "general", "delete_trade_histo"):
            if os.path.exists(os.path.join(Trading_path, "trading_ledger.json")):
                os.remove(os.path.join(Trading_path, "trading_ledger.json"))
            if os.path.exists(os.path.join(Trading_path, "trading_metrics_AlgoTrader.json")):
                os.remove(os.path.join(Trading_path, "trading_metrics_AlgoTrader.json"))

        #init aws
        self.key = config["AWS"]["access_key"]
        self.sec_key = config["AWS"]["secret_access_key"]
        self.bucket = config["AWS"]["bucket_name"]
        self.arn_role = config["AWS"]["arn_role"]
        self.region_name = config["AWS"]["region_name"]
        
        self.gutils = GoogleDriveUtility(parent_folder="AlgoTrader Drive")   
        
        #logger
        self.logger = logger
        self.configure_logger()
        
        #trading pair
        self.coin = utils.get_config_value(config, "general", "coin")
        self.fiat = utils.get_config_value(config, "general", "currency")
        self.symbol = f"{self.coin}-{self.fiat}"
        self.slippage =  utils.get_config_value(config, "general", "slippage")
        self.leverage_factor = utils.get_config_value(config, "trading", "leverage_factor")
        self.atr_factor_trending = utils.get_config_value(config, "trading", "atr_factor_trending")
        self.atr_factor_ranging = utils.get_config_value(config, "trading", "atr_factor_ranging")
        self.exchange_name = utils.get_config_value(config, "general", "exchange")
        self.interval =  utils.get_config_value(config, "general", "interval")
        self.start_date = "2017-01-05 00:00:00"
        
        if self.interval == "1h":
            self.current_datetime = dt.datetime.now().replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
            self.current_datetime_tm1 = (dt.datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        if self.interval == "1d":
            self.current_datetime = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
            self.current_datetime_tm1 = (dt.datetime.now().replace(hour=0,minute=0, second=0, microsecond=0) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        if self.interval == "5m":
            current_minute = dt.datetime.now().minute
            current_minute = current_minute - (current_minute % 5)
            self.current_datetime = dt.datetime.now().replace(minute=current_minute,second=0,microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
            self.current_datetime_tm1 = (dt.datetime.now().replace(minute=current_minute,second=0,microsecond=0)- timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        if self.interval == "15m":
            current_minute = dt.datetime.now().minute
            current_minute = current_minute - (current_minute % 15)
            self.current_datetime = dt.datetime.now().replace(minute=current_minute,second=0,microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
            self.current_datetime_tm1 = (dt.datetime.now().replace(minute=current_minute,second=0,microsecond=0) - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
        
        # init kucoin trader   (also init self.ledger_data)
        MainApiTrader.__init__(self, exchange=self.exchange_name, fiat=self.fiat, slippage=self.slippage, leverage=self.leverage_factor,logger_input=self.logger)
        
        #init TA creator
        self.TA_creator = TAIndicatorCreator(logger=self.logger)
        
        #init kucoin price loader
        self.kucoin_price_loader = KuCoinDataDownloader(created_logger=self.logger)
        self.kucoin_price_loader.download_data(self.coin, self.fiat, self.interval, start_date=self.start_date, use_local_timezone=True, drop_not_complete_timestamps=False)
        
        
        #init Strategy Evaluator
        self.strategy_evaluator = StrategyEvaluator(logger=self.logger)
        
        filename = utils.get_config_value(config, "general", "dataset_filename")
        self.use_data_loader = utils.get_config_value(config, "general", "use_data_loader")
        self.use_multi_timeframe = utils.get_config_value(config, "general", "use_multi_timeframe")
        self.multi_timeframe = utils.get_config_value(config, "general", "multi_timeframe") if self.use_multi_timeframe else None
        self.use_atr = utils.get_config_value(config, "general", "use_atr")
        self.can_switch_sides = utils.get_config_value(config, "general", "can_switch_sides")
        self.performance_filename = utils.get_config_value(config, "general", "performance_filename")
        self.cooldown_condtions = utils.get_config_value(config, "general", "cooldown_condtions")
        self.trade_restrictions_conditions = utils.get_config_value(config, "general", "trade_restrictions_conditions")
        self.trade_cooldown_counter = utils.get_config_value(config, "general", "trade_cooldown_counter")

        if self.use_data_loader:
            dataloader = DataLoader(logger_input=self.logger, filename=filename)
            csv_name = os.path.splitext(filename)[0] + "_processed" + ".csv"   
            hdf5_name = os.path.splitext(filename)[0] + "_processed" + ".h5"
            
            if not os.path.exists(os.path.join(csv_dataset_path, (csv_name))):
                self.data = dataloader.main_loader(save_dataset=True)
            else:
                self.data = pd.read_csv(os.path.join(csv_dataset_path, (csv_name)), index_col=0)
        else:
            metric = "price_usdt_kucoin" #filename.split("_")[0]
            filename = f"{metric}_{self.coin}_{self.fiat}_{self.interval}.csv" 
            
            if os.path.exists(os.path.join(histo_data_path,metric,self.interval,filename)):
                self.data = pd.read_csv(os.path.join(histo_data_path,metric,self.interval,filename), index_col=0)
                
                if self.data.empty:
                    self.data, minute_data = self.kucoin_price_loader.download_data(self.coin, self.fiat, self.interval, start_date=self.start_date, use_local_timezone=True)
                    
                    for interval in ["1d"]:
                        self.kucoin_price_loader.download_data(self.coin, self.fiat, interval, start_date=self.start_date, use_local_timezone=True)
                    
                    self.data.index = pd.to_datetime(self.data.index)
                
                self.data.index = pd.to_datetime(self.data.index)
                
                if self.data.index[-1] <= self.start_datetime:
                    self.data, minute_data = self.kucoin_price_loader.download_data(self.coin, self.fiat, self.interval, start_date=self.start_date, use_local_timezone=True)
                    for interval in ["1d"]:
                        self.kucoin_price_loader.download_data(self.coin, self.fiat, interval, start_date=self.start_date, use_local_timezone=True)
                    
                    self.data.index = pd.to_datetime(self.data.index)
                
                self.data[f"price_{self.fiat.lower()}_{self.ticker_to_slug_mapping[self.coin].lower()}"] = self.data["close"]
                # self.data = self.data.rename(columns={"close":f"price_{self.fiat.lower()}_{self.ticker_to_slug_mapping[self.coin].lower()}"})
                
            else:
                self.data, minute_data = self.kucoin_price_loader.download_data(self.coin, self.fiat, self.interval, start_date=self.start_date, use_local_timezone=True)
                
                for interval in ["1d"]:
                        self.kucoin_price_loader.download_data(self.coin, self.fiat, interval, start_date=self.start_date, use_local_timezone=True)
                
                self.data[f"price_{self.fiat.lower()}_{self.ticker_to_slug_mapping[self.coin].lower()}"] = self.data["close"]
                # self.data = self.data.rename(columns={"close":f"price_{self.fiat.lower()}_{self.ticker_to_slug_mapping[self.coin].lower()}"})

                # timestamp input is self.end_date in real calc use timestamp_input=timestam_input
            
        self.data.index = pd.to_datetime(self.data.index, format="%Y-%m-%d %H:%M:%S")
        
        #balance coin
        self.current_account_balance_coin = self.get_margin_account_details(curr=self.coin)["total"].values[0]
        self.current_available_account_balance_coin = self.get_margin_account_details(curr=self.coin)["available"].values[0]
        
        #balance curr
        self.current_account_balance_curr = self.get_margin_account_details(curr=self.fiat)["total"].values[0]
        self.current_available_account_balance_curr = self.get_margin_account_details(curr=self.fiat)["available"].values[0]
        
        #value coin in currency
        self.current_coin_value_in_fiat = self.calculate_balance_in_fiat("EUR", self.coin, getSum=True)
        self.current_coin_value_in_currency = self.calculate_balance_in_fiat(self.fiat, self.coin, getSum=True)
        
        #trade settings
        self.taker_trade_fee, self.taker_maker_fee = self.get_actual_trading_fees(self.symbol)
        self.max_open_positions = utils.get_config_value(config, "trading", "max_open_positions")
          
        #ratios:
        self.investment_per_trade = utils.get_config_value(config, "trading", "investment_per_trade")
        self.stop_loss_percentage = utils.get_config_value(config, "trading", "stop_loss_percentage")
        self.take_profit_percentage = utils.get_config_value(config, "trading", "take_profit_percentage")
        
        # Account variables
        self.positions = self.ledger_data["positions"]
        self.ledger_data["balances"]["trailing_balances"]["total_balance"] = self.current_account_balance_curr + self.current_coin_value_in_currency
        self.balances = self.ledger_data["balances"]
        
        if len(self.ledger_data["positions"]) ==0:
            initial_balance_value = self.current_account_balance_curr
            self.ledger_data["balances"]["initial_balance"][self.fiat] = initial_balance_value
            self.ledger_data["balances"]["initial_balance"][self.coin] = {"fiat_value":(self.current_account_balance_coin *self.get_price(self.symbol)),"coin_amt":self.current_account_balance_coin}
            self.initial_balance_curr = initial_balance_value
            self.initial_balance_coin_in_curr = self.current_account_balance_coin*self.get_price(self.symbol)
            self.initial_balance_coin = self.current_account_balance_coin
            self.ledger_data["balances"]["trailing_balances"]["total_balance"] = initial_balance_value
        else:
            self.initial_balance_curr = self.ledger_data["balances"]["initial_balance"][self.fiat]
            self.initial_balance_coin_in_curr =self.ledger_data["balances"]["initial_balance"][self.coin]["fiat_value"]
            self.initial_balance_coin = self.ledger_data["balances"]["initial_balance"][self.coin]["coin_amt"]
            self.ledger_data["balances"]["trailing_balances"]["total_balance"] = self.initial_balance_curr + self.initial_balance_coin_in_curr

        self.pnl_account_abs = (self.current_account_balance_curr + self.get_price(self.symbol)*self.current_account_balance_coin) - (self.initial_balance_curr + self.initial_balance_coin_in_curr)
        self.pnl_account_pct = ((self.current_account_balance_curr + self.get_price(self.symbol)*self.current_account_balance_coin) - (self.initial_balance_curr + self.initial_balance_coin_in_curr)) / (self.initial_balance_curr + self.initial_balance_coin_in_curr)
        self.total_profit_balance = max(((self.current_account_balance_curr + self.get_price(self.symbol)*self.current_account_balance_coin)- (self.initial_balance_curr + self.initial_balance_coin_in_curr)),0)

        if not os.path.exists(os.path.join(Trading_path,f"trade_performance_{self.coin}_{self.fiat}_{self.performance_filename}.csv")):
            self.trade_performance = self.configure_performance_file(data="balances", datetime_input=self.current_datetime, filename=self.performance_filename)
        else:
            self.trade_performance = pd.read_csv(os.path.join(Trading_path,f"trade_performance_{self.coin}_{self.fiat}_{self.performance_filename}.csv"), index_col=0)
        
        if not os.path.exists(os.path.join(Trading_path,f"trade_positions_{self.coin}_{self.fiat}_{self.performance_filename}.csv")):
            self.trade_positions = self.configure_performance_file(data="positions", datetime_input=self.current_datetime, filename=self.performance_filename)
        else:
            self.trade_positions = pd.read_csv(os.path.join(Trading_path,f"trade_positions_{self.coin}_{self.fiat}_{self.performance_filename}.csv"), index_col=0)
        
        self.strategy_data = pd.DataFrame()

        self.last_signal_datetime = None
        self.last_cooldown_time  = None
        self.trade_cooldown = 0
        self.position_close = None
        
        self.logger.info(f"{'#'*1}   Initial balance in {self.fiat} is: {self.initial_balance_curr}   {'#'*1}")
        self.logger.info(f"{'#'*1}   Initial balance in {self.coin} is: {self.initial_balance_coin_in_curr}   {'#'*1}")

        self.logger.info("AlgoTrader initialized")
#################################################################################################################################################################
#
#
#                                                                  Algo trader functions
#
#################################################################################################################################################################


    # def load_trading_metrics(self, metrics_file_path=None):
    #     if not self.aws.check_if_path_exists(metrics_file_path):
    #         trading_metrics_data = {"losing_trades": 0 , "winning_trades": 0, "largest_loss": 0, "largest_gain": 0}
            
    #         # Convert the trading metrics data to a JSON string
    #         metrics_json = json.dumps(trading_metrics_data, indent=4)
            
    #         # Save the JSON string directly to S3
    #         metrics_buffer = BytesIO(metrics_json.encode('utf-8'))
    #         self.aws.save_file(metrics_buffer.getvalue(), metrics_file_path)
            
    #         return trading_metrics_data
    #     else:
    #         return self.aws.load_file(metrics_file_path)
        
    @logger.catch
    @retry(max_retries=5, delay=2, backoff=2)
    def trade(self, datetime_input=None):
        try:
            actual_datetime = datetime_input
            stop_loss_price = None
            take_profit_price = None
            coin_to_trade = None
            
            if self.interval == "1h":
                datetime_input= datetime_input.replace(minute=0, second=0, microsecond=0)
                tm1 = (datetime_input - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                tm2 = (datetime_input - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
            elif self.interval == "1d":
                datetime_input= datetime_input.replace(hours=0, minute=0, second=0, microsecond=0)
                tm1 = (datetime_input - timedelta(days=1)).replace(hours=0, minute=0, second=0, microsecond=0)
                tm2 = (datetime_input - timedelta(days=2)).replace(hours=0, minute=0, second=0, microsecond=0)
            elif self.interval == "5m":
                datetime_input= datetime_input.replace(second=0, microsecond=0).replace(minute=5 * (datetime_input.minute // 5))
                tm1 = (datetime_input - timedelta(minutes=5)).replace(second=0, microsecond=0)
                tm2 = (datetime_input - timedelta(minutes=10)).replace(second=0, microsecond=0)
            elif self.interval == "15m":
                datetime_input= datetime_input.replace(second=0, microsecond=0).replace(minute=15 * (datetime_input.minute // 15))
                tm1 = (datetime_input - timedelta(minutes=15)).replace(second=0, microsecond=0)
                tm2 = (datetime_input - timedelta(minutes=30)).replace(second=0, microsecond=0)
                        
            
            #Get signal
            # signal_macd = self.strategy_evaluator.macd_cross_and_ema_trend_checker(coin=self.coin,fiat=self.fiat, interval=self.interval, trend_interval=self.multi_timeframe, 
            #                                                                 timestamp_input_interval_adj=datetime_input, timestamp_input_actual=actual_datetime, simulate_live_data=False)
            
            signal, atr, condition_index, strategy_data = self.strategy_evaluator.ema_high_low_boundary_strategy(coin=self.coin,fiat=self.fiat, interval=self.interval, trend_interval=self.interval, 
                                                                            timestamp_input_interval_adj=datetime_input, timestamp_input_actual=actual_datetime, simulate_live_data=False)
            
            signal_strategy = "signal_ema"
            self.strategy_data = pd.concat([self.strategy_data, strategy_data], axis=0)

            # Check if signal is already received
            if self.last_signal_datetime is not None and self.last_signal_datetime == datetime_input and condition_index in self.trade_restrictions_conditions:
                self.logger.info(f"{'#'*1}   Signal already received for this hour. Waiting for next update   {'#'*1}")
                return
            elif self.last_signal_datetime is not None and datetime_input == self.position_close and condition_index in self.trade_restrictions_conditions:
                self.logger.info(f"{'#'*1}   Position already closed this hour. Waiting for next update   {'#'*1}")
                return

            # Check for trade cooldown
            if self.trade_cooldown > 0 and self.last_cooldown_time != datetime_input:
                self.trade_cooldown -= 1
                self.last_cooldown_time = datetime_input
                self.logger.info(f"{'#'*1}   Trade cooldown reduced. Current cooldown at {self.trade_cooldown}  {'#'*1}")
                
            # Check for trade cooldown
            if condition_index == self.cooldown_condtions and self.trade_cooldown > 0:
                self.logger.info(f"{'#'*1}   Trade cooldown active. Waiting for next update. Current cooldown at {self.trade_cooldown}  {'#'*1}")
                return

            
            if signal == "neutral":
                self.logger.info(f"{'#'*30}   Received neutral signal. Waiting for next update   {'#'*30}")
                return

            else:
                coin_to_trade = self.coin if coin_to_trade is None else coin_to_trade
                symbol_to_trade = f"{coin_to_trade}-{self.fiat}"
                
                if self.has_position:
                    current_order_id = self.ledger_data["current_trades"][self.current_traded_symbol]["order_id"]
                    position = self.get_position_by_order_id(current_order_id)

                    if position["side"] == signal:
                        self.logger.info(f"Received signal for {signal} but already invested. Waiting for next tick")
                        return

                self.logger.info(f"     ")
                self.logger.info(f"{'#'*30}   Received {signal} signal. Executing trade   {'#'*30}")
                self.logger.info(f"     ")
                
                #check for open positions or orders
                has_open_order = False
                has_open_position = False
                trading_fees = self.get_trading_fees("0")[0]
                
                # Check if the balance is a DataFrame/Series and not a scalar
                balance_curr = self.balances["trailing_balances"].get(self.fiat)

                if isinstance(balance_curr, pd.Series) or isinstance(balance_curr, pd.DataFrame):
                    self.current_account_balance_curr = balance_curr.iloc[0] if not balance_curr.empty else 0
                else:
                    # Handle the case where the balance is a scalar or other type
                    self.current_account_balance_curr = balance_curr if balance_curr is not None else 0
                
                # Check if the balance for the coin is a DataFrame/Series and not a scalar
                balance_coin = self.balances["trailing_balances"].get(coin_to_trade)

                if isinstance(balance_coin, pd.Series) or isinstance(balance_coin, pd.DataFrame):
                    self.current_account_balance_coin = balance_coin.iloc[0] if not balance_coin.empty else 0
                else:
                    # Handle the case where the balance is a scalar or other type
                    self.current_account_balance_coin = balance_coin if balance_coin is not None else 0
                
                # for symbol in self.ledger_data["current_trades"]:
                #     coin = symbol.split("-")[0]
                #     if self.ledger_data["current_trades"][symbol] is not None:
                #         self.current_traded_coin = coin
                #         self.current_traded_symbol = f"{coin}-{self.fiat}"
                #     else:
                #         self.current_traded_coin = self.coin
                #         self.current_traded_symbol = f"{self.coin}-{self.fiat}"
                
                
                if self.ledger_data["current_trades"][self.current_traded_symbol] is not None:
                    has_open_position = True
                    current_order_id = self.ledger_data["current_trades"][self.current_traded_symbol]["order_id"]
                    current_position = self.get_position_by_order_id(current_order_id)
                    current_stop_loss_order_id = self.ledger_data["current_trades"][self.current_traded_symbol]["stop_loss_order_id"]
                    current_take_profit_order_id = self.ledger_data["current_trades"][self.current_traded_symbol]["take_profit_order_id"]

                    check_if_order_still_active = self.get_order_details_by_id(orderId=current_order_id)
                    check_if_order_done = self.get_order_details_by_id(orderId=current_order_id)
                    
                    if check_if_order_still_active["isActive"][0]:
                        has_open_order = True
                    elif not check_if_order_done["isActive"][0]:
                        self.has_position = True
                else:
                    #track last signal and signal 
                    self.trade_cooldown = self.trade_cooldown_counter+1 if (condition_index in self.cooldown_condtions) else 1    #enforce cooldown of 5 timesteps
                    self.last_signal_datetime = datetime_input 
                    self.last_signal_side = signal
                    self.last_cooldown_time = datetime_input
                    self.position_close = None
                
                price = self.get_price(symbol_to_trade)


                #implement check that when atr is less than order cost no trade is executed
                if self.interval in ["1m","5m","15m","30m"] and self.use_atr:
                    atr_factor = self.atr_factor_trending if condition_index in [0,1,2,3,4,5,6,7] else self.atr_factor_ranging
                    potential_payoff = (((price - (atr_factor*min(atr, 750))) / price) -1) * self.ledger_data["balances"]["trailing_balances"][self.fiat]
                    potential_fees = self.get_trading_fees(coin=coin_to_trade, side=signal)*self.ledger_data["balances"]["trailing_balances"][self.fiat]
                    if potential_payoff > potential_fees:
                        self.logger.info(f"{'#'*1}   Potential payoff is less than fees. No trade executed.  {'#'*1}")
                        return
                    
                # #creating sl and tp prices, changing percentage based of strategy
                # if signal == "buy":
                #     if strategy_sl_tp_adjustment == "signal_macd":
                #         stop_loss_price = (1-(self.stop_loss_percentage/100)) * price     #sl percentage is in absolute values and not in percentage (decimal)
                #         take_profit_price = (1+(self.take_profit_percentage/100)) * price #tp percentage is in absolute values and not in percentage (decimal)
                #     else:
                #         stop_loss_price = (1-(self.stop_loss_percentage/100/4)) * price     #sl percentage is in absolute values and not in percentage (decimal)
                #         take_profit_price = (1+(self.take_profit_percentage/100/2)) * price #tp percentage is in absolute values and not in percentage (decimal)
                # elif signal == "sell":
                #     if strategy_sl_tp_adjustment == "signal_macd":
                #         stop_loss_price = (1+(self.stop_loss_percentage/100)) * price       #sl percentage is in absolute values and not in percentage (decimal)
                #         take_profit_price = (1-(self.take_profit_percentage/100)) * price   #tp percentage is in absolute values and not in percentage (decimal)
                #     else:
                #         stop_loss_price = (1+(self.stop_loss_percentage/100/4)) * price       #sl percentage is in absolute values and not in percentage (decimal)
                #         take_profit_price = (1-(self.take_profit_percentage/100/2)) * price 


                if self.use_atr:
                    if signal == "buy":
                        if condition_index in [1,2,3]:    #for ma hl div vol use  [0,2,3]
                            stop_loss_price = price - 1*min(atr,750)    #self.atr_factor_trending
                            take_profit_price = price + self.atr_factor_trending*min(atr,750)
                        elif condition_index in []:
                            stop_loss_price = (1-(self.stop_loss_percentage/100)) * price     #sl percentage is in absolute values and not in percentage (decimal)
                            take_profit_price = (1+(self.take_profit_percentage/100)) * price
                        elif condition_index in [0]:  #last candle low (doji)
                            stop_loss_price = self.data["low"].loc[tm1]
                            take_profit_price = price + self.atr_factor_trending*min(atr,750)
                        elif condition_index in [4]:  #second engulfed candle high (engulfing)
                            stop_loss_price = self.data["high"].loc[tm2]
                            take_profit_price = price + self.atr_factor_ranging*min(atr,750)
                        elif condition_index in [5]: #only 1 atr for both
                            stop_loss_price = price - min(atr,750)   #self.atr_factor_trending
                            take_profit_price = price + min(atr,750)
                        else:
                            stop_loss_price = price - min(atr,750)   #self.atr_factor_ranging
                            take_profit_price = price + self.atr_factor_ranging*min(atr,750)
                    elif signal == "sell":
                        if condition_index in [1,2,3]: #for ma hl div vol use  [0,2,3]
                            stop_loss_price = price + 1*min(atr,750)                    #self.atr_factor_trending
                            take_profit_price = price - self.atr_factor_trending*min(atr,750)
                        elif condition_index in []:
                            stop_loss_price = (1+(self.stop_loss_percentage/100)) * price       #sl percentage is in absolute values and not in percentage (decimal)
                            take_profit_price = (1-(self.take_profit_percentage/100)) * price 
                        elif condition_index in [0]:  #last candle high (doji)
                            stop_loss_price = self.data["high"].loc[tm1]
                            take_profit_price = price - self.atr_factor_ranging*min(atr,750)
                        elif condition_index in [4]:  #second engulfed candle low (engulfing)
                            stop_loss_price = self.data["low"].loc[tm2]
                            take_profit_price = price - self.atr_factor_ranging*min(atr,750)
                        elif condition_index in [5]: #only 1 atr for both
                            stop_loss_price = price + min(atr,750)   #self.atr_factor_trending
                            take_profit_price = price - min(atr,750)
                        else:
                            stop_loss_price = price + min(atr,750)  #self.atr_factor_ranging
                            take_profit_price = price - self.atr_factor_ranging*min(atr,750)
                elif stop_loss_price is not None and take_profit_price is not None:
                    pass
                else:
                    if signal == "buy":
                        stop_loss_price = (1-(self.stop_loss_percentage/100)) * price     #sl percentage is in absolute values and not in percentage (decimal)
                        take_profit_price = (1+(self.take_profit_percentage/100)) * price #tp percentage is in absolute values and not in percentage (decimal)
                    elif signal == "sell":
                        stop_loss_price = (1+(self.stop_loss_percentage/100)) * price       #sl percentage is in absolute values and not in percentage (decimal)
                        take_profit_price = (1-(self.take_profit_percentage/100)) * price    #tp percentage is in absolute values and not in percentage (decimal)

                #executing trades
                if signal == "buy" and has_open_position ==False and has_open_order == False:
                    
                    # Check if the balance is a DataFrame/Series and not a scalar
                    balance_curr = self.balances["trailing_balances"].get(self.fiat)

                    if isinstance(balance_curr, pd.Series) or isinstance(balance_curr, pd.DataFrame):
                        self.current_account_balance_curr = balance_curr.iloc[0] if not balance_curr.empty else 0
                    else:
                        # Handle the case where the balance is a scalar or other type
                        self.current_account_balance_curr = balance_curr if balance_curr is not None else 0
                    
                    # Check if the balance for the coin is a DataFrame/Series and not a scalar
                    balance_coin = self.balances["trailing_balances"].get(coin_to_trade)

                    if isinstance(balance_coin, pd.Series) or isinstance(balance_coin, pd.DataFrame):
                        self.current_account_balance_coin = balance_coin.iloc[0] if not balance_coin.empty else 0
                    else:
                        # Handle the case where the balance is a scalar or other type
                        self.current_account_balance_coin = balance_coin if balance_coin is not None else 0
                    
                    unlevered_funds = self.current_account_balance_curr *self.investment_per_trade
                    funds = unlevered_funds * self.leverage_factor
                    size=None

                    self.enter_margin_trade(coin=coin_to_trade, fiat=self.fiat,
                                            size=size, funds=funds, balance_fiat = self.current_available_account_balance_curr, is_long=True, order_type="market",
                                            stop_price=stop_loss_price, take_profit_price=take_profit_price)
                    
                    self.ledger_data["current_trades"][symbol_to_trade]["current strategy"]  = signal_strategy
                    self.strategy_data.loc[strategy_data.index[0],"triggered"] = True 
                
                
                if signal == "buy" and has_open_position==True and current_position["side"]=="sell": #and  signal_strategy == self.ledger_data["current_trades"][self.symbol]["current strategy"]:   #only considers changing sides when previous trade was sell, currently two consecutive buy orders are not permitted
                    if self.can_switch_sides and condition_index in [0,3,4]:
                        self.close_margin_position(coin=coin_to_trade, fiat=self.fiat, order_id=current_order_id, current_position=current_position)
                        self.position_close 

                        # Check if the balance is a DataFrame/Series and not a scalar
                        balance_curr = self.balances["trailing_balances"].get(self.fiat)

                        if isinstance(balance_curr, pd.Series) or isinstance(balance_curr, pd.DataFrame):
                            self.current_account_balance_curr = balance_curr.iloc[0] if not balance_curr.empty else 0
                        else:
                            # Handle the case where the balance is a scalar or other type
                            self.current_account_balance_curr = balance_curr if balance_curr is not None else 0
                        
                        # Check if the balance for the coin is a DataFrame/Series and not a scalar
                        balance_coin = self.balances["trailing_balances"].get(coin_to_trade)

                        if isinstance(balance_coin, pd.Series) or isinstance(balance_coin, pd.DataFrame):
                            self.current_account_balance_coin = balance_coin.iloc[0] if not balance_coin.empty else 0
                        else:
                            # Handle the case where the balance is a scalar or other type
                            self.current_account_balance_coin = balance_coin if balance_coin is not None else 0
                        
                        unlevered_funds = self.current_account_balance_curr *self.investment_per_trade
                        funds = unlevered_funds * self.leverage_factor
                        size=None

                        self.enter_margin_trade(coin=coin_to_trade, fiat=self.fiat,
                                                size=size, funds=funds, balance_fiat = self.current_available_account_balance_curr, is_long=True, order_type="market",
                                                stop_price=stop_loss_price, take_profit_price=take_profit_price)

                        self.ledger_data["current_trades"][symbol_to_trade]["current strategy"]  = signal_strategy
                        self.strategy_data.loc[strategy_data.index[0],"triggered"] = True
                    else:
                        self.logger.info("{'#'*1}   Received buy signal although already short. Not changing sides. Condition index not met. Waiting for next update   {'#'*1}")

                elif signal == "buy" and has_open_position==True and current_position["side"]=="buy":
                    
                    # Check if the balance is a DataFrame/Series and not a scalar
                    balance_curr = self.balances["trailing_balances"].get(self.fiat)

                    if isinstance(balance_curr, pd.Series) or isinstance(balance_curr, pd.DataFrame):
                        self.current_account_balance_curr = balance_curr.iloc[0] if not balance_curr.empty else 0
                    else:
                        # Handle the case where the balance is a scalar or other type
                        self.current_account_balance_curr = balance_curr if balance_curr is not None else 0
                    
                    # Check if the balance for the coin is a DataFrame/Series and not a scalar
                    balance_coin = self.balances["trailing_balances"].get(coin_to_trade)

                    if isinstance(balance_coin, pd.Series) or isinstance(balance_coin, pd.DataFrame):
                        self.current_account_balance_coin = balance_coin.iloc[0] if not balance_coin.empty else 0
                    else:
                        # Handle the case where the balance is a scalar or other type
                        self.current_account_balance_coin = balance_coin if balance_coin is not None else 0
                    
                    self.logger.info(f"{'#'*30}   Received buy signal although already long. Waiting for next update   {'#'*30}")
                    pass
                
                elif signal == "buy" and has_open_position==True and current_position["side"]=="sell" and self.can_switch_sides==False:
                    self.logger.info(f"{'#'*1}   Received buy signal although already short. Not changing sides. Waiting for next update   {'#'*1}")

                elif signal == "sell" and has_open_position ==False and has_open_order == False:
                    
                    # Check if the balance is a DataFrame/Series and not a scalar
                    balance_curr = self.balances["trailing_balances"].get(self.fiat)

                    if isinstance(balance_curr, pd.Series) or isinstance(balance_curr, pd.DataFrame):
                        self.current_account_balance_curr = balance_curr.iloc[0] if not balance_curr.empty else 0
                    else:
                        # Handle the case where the balance is a scalar or other type
                        self.current_account_balance_curr = balance_curr if balance_curr is not None else 0
                    
                    # Check if the balance for the coin is a DataFrame/Series and not a scalar
                    balance_coin = self.balances["trailing_balances"].get(coin_to_trade)

                    if isinstance(balance_coin, pd.Series) or isinstance(balance_coin, pd.DataFrame):
                        self.current_account_balance_coin = balance_coin.iloc[0] if not balance_coin.empty else 0
                    else:
                        # Handle the case where the balance is a scalar or other type
                        self.current_account_balance_coin = balance_coin if balance_coin is not None else 0
                    
                    size = ((self.current_account_balance_curr *self.investment_per_trade)*self.leverage_factor)  / price
                    size = self.round_down(size * (1-trading_fees),8)*(1-(self.stop_loss_percentage/100))
                    funds=None

                    self.enter_margin_trade(coin=coin_to_trade, fiat=self.fiat,
                                            size=size, funds=funds, balance_fiat = self.current_available_account_balance_curr, is_long=False, order_type="market",
                                            stop_price=stop_loss_price, take_profit_price=take_profit_price)

                    self.ledger_data["current_trades"][symbol_to_trade]["current strategy"]  = signal_strategy
                    self.strategy_data.loc[strategy_data.index[0],"triggered"] = True 

                if signal == "sell" and has_open_position==True and current_position["side"]=="buy": # and  signal_strategy == self.ledger_data["current_trades"][self.symbol]["current strategy"]:
                    if self.can_switch_sides and condition_index in [0,3,4]:
                        self.close_margin_position(coin=coin_to_trade, fiat=self.fiat, order_id=current_order_id, current_position=current_position)
                        self.position_close 
                        # Check if the balance is a DataFrame/Series and not a scalar
                        balance_curr = self.balances["trailing_balances"].get(self.fiat)

                        if isinstance(balance_curr, pd.Series) or isinstance(balance_curr, pd.DataFrame):
                            self.current_account_balance_curr = balance_curr.iloc[0] if not balance_curr.empty else 0
                        else:
                            # Handle the case where the balance is a scalar or other type
                            self.current_account_balance_curr = balance_curr if balance_curr is not None else 0
                        
                        # Check if the balance for the coin is a DataFrame/Series and not a scalar
                        balance_coin = self.balances["trailing_balances"].get(coin_to_trade)

                        if isinstance(balance_coin, pd.Series) or isinstance(balance_coin, pd.DataFrame):
                            self.current_account_balance_coin = balance_coin.iloc[0] if not balance_coin.empty else 0
                        else:
                            # Handle the case where the balance is a scalar or other type
                            self.current_account_balance_coin = balance_coin if balance_coin is not None else 0
                        
                        size = ((self.current_account_balance_curr *self.investment_per_trade)*self.leverage_factor)  / price
                        size = self.round_down(size * (1-trading_fees),8)*(1-(self.stop_loss_percentage/100))
                        funds=None

                        self.enter_margin_trade(coin=coin_to_trade, fiat=self.fiat,
                                                size=size, funds=funds, balance_fiat = self.current_available_account_balance_curr, is_long=False, order_type="market",
                                                stop_price=stop_loss_price, take_profit_price=take_profit_price)

                        self.ledger_data["current_trades"][symbol_to_trade]["current strategy"]  = signal_strategy
                        self.strategy_data.loc[strategy_data.index[0],"triggered"] = True 
                    else:
                        self.logger.info("{'#'*1}   Received sell signal although already long. Not changing sides. Condition index not met. Waiting for next update   {'#'*1}")
                elif signal == "sell" and has_open_position==True and current_position["side"]=="sell":
                    
                    # Check if the balance is a DataFrame/Series and not a scalar
                    balance_curr = self.balances["trailing_balances"].get(self.fiat)

                    if isinstance(balance_curr, pd.Series) or isinstance(balance_curr, pd.DataFrame):
                        self.current_account_balance_curr = balance_curr.iloc[0] if not balance_curr.empty else 0
                    else:
                        # Handle the case where the balance is a scalar or other type
                        self.current_account_balance_curr = balance_curr if balance_curr is not None else 0
                    
                    # Check if the balance for the coin is a DataFrame/Series and not a scalar
                    balance_coin = self.balances["trailing_balances"].get(coin_to_trade)

                    if isinstance(balance_coin, pd.Series) or isinstance(balance_coin, pd.DataFrame):
                        self.current_account_balance_coin = balance_coin.iloc[0] if not balance_coin.empty else 0
                    else:
                        # Handle the case where the balance is a scalar or other type
                        self.current_account_balance_coin = balance_coin if balance_coin is not None else 0
                    
                    self.logger.info(f"{'#'*30}   Received sell signal although already short. Waiting for next update   {'#'*30}")
                    pass
                
                elif signal == "sell" and has_open_position==True and current_position["side"]=="buy" and self.can_switch_sides==False:
                    self.logger.info(f"{'#'*1}   Received sell signal although already long. Not changing sides. Waiting for next update   {'#'*1}")
                    pass

        except Exception as e:
            self.logger.error(f"An error occured during the execution of the trade function. Error:  {e}")
            raise Exception(f"An error occured during the execution of the trade function. Error:  {e}")
    
    # @staticmethod
    @logger.catch
    @retry(max_retries=5, delay=2, backoff=2)
    def status_scanner(self, datetime_input=None):
        try:
            actual_datetime = datetime_input 
            
            #check for open positions or orders
            has_position = False
            self.has_position = False
            
            #check which coin is currently traded
            for symbol in self.ledger_data["current_trades"]:
                coin = symbol.split("-")[0]
                if self.ledger_data["current_trades"][symbol] is not None:
                    self.current_traded_coin = coin
                    self.current_traded_symbol = f"{coin}-{self.fiat}"
                    
                    if coin != self.coin or coin != self.current_traded_coin:
                        self.data, minute_data = self.kucoin_price_loader.download_data(self.current_traded_coin, self.fiat, self.interval, start_date=self.start_datetime, end_date=actual_datetime, 
                                                                    use_local_timezone=True, drop_not_complete_timestamps=False, simulate_live_data=self.simulate_live_trading, overwrite_file=False)  
                
                        self.data[f"price_{self.fiat.lower()}_{self.ticker_to_slug_mapping[self.current_traded_coin].lower()}"] = self.data["close"]
                        # self.data = self.data.rename(columns={"close":f"price_{self.fiat.lower()}_{self.ticker_to_slug_mapping[self.coin].lower()}"})
                        self.data.index = pd.to_datetime(self.data.index)
                    
                else:
                    self.current_traded_coin = self.coin
                    self.current_traded_symbol = f"{self.coin}-{self.fiat}"
            
            #get last datetime
            if self.interval == "1h":
                datetime_input_adj = datetime_input.replace(minute=0, second=0, microsecond=0)
            if self.interval == "1d":
                datetime_input_adj = datetime_input.replace(hour=0,minute=0, second=0, microsecond=0)
            if self.interval == "5m":
                current_minute = datetime_input.minute
                current_minute = current_minute - (current_minute % 5)
                datetime_input_adj = datetime_input.replace(minute=current_minute,second=0,microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
            if self.interval == "15m":
                current_minute = datetime_input.minute
                current_minute = current_minute - (current_minute % 15)
                datetime_input_adj = datetime_input.replace(minute=current_minute,second=0,microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
            
            self.logger.info(f"{'#'*1}   Current price is: {self.get_price(self.current_traded_symbol)} at time {datetime_input}    {'#'*1}")
            
            #check if position is still open
            if self.ledger_data["current_trades"][self.current_traded_symbol] is not None:
                current_order_id = self.ledger_data["current_trades"][self.current_traded_symbol]["order_id"]
                position = self.get_position_by_order_id(current_order_id)
                self.has_position = True
            else:
                position = None
            
            if self.has_position:
                risk_prevention_status = self.check_sl_or_tp_triggered(coin=self.current_traded_coin, fiat=self.fiat, position=position)
                
                if risk_prevention_status != "not triggered":
                    self.has_position = False
                    self.position_close = datetime_input
            else:
                # check if any debt and if so pay back
                self.pay_off_all_debts(curr=self.current_traded_coin)
                
            last_date = self.data.index[-1]
        
            if last_date > datetime_input:   #adjust data to only go until current datetime
                self.data = self.data.loc[:datetime_input]
                last_date = self.data.index[-1]
            
            if last_date < actual_datetime:
                self.data, minute_data = self.kucoin_price_loader.download_data(self.current_traded_coin, self.fiat, self.interval, start_date=last_date, end_date=actual_datetime, 
                                                                    use_local_timezone=True, drop_not_complete_timestamps=False, simulate_live_data=False, overwrite_file=True)  
                
                self.data[f"price_{self.fiat.lower()}_{self.ticker_to_slug_mapping[self.current_traded_coin].lower()}"] = self.data["close"]
                # self.data = self.data.rename(columns={"close":f"price_{self.fiat.lower()}_{self.ticker_to_slug_mapping[self.coin].lower()}"})
                self.data.index = pd.to_datetime(self.data.index)

            #balance coin
            self.current_account_balance_coin = self.get_margin_account_details(curr=self.current_traded_coin)["total"].values[0]
            self.current_available_account_balance_coin = self.get_margin_account_details(curr=self.current_traded_coin)["available"].values[0]
            
            #balance curr
            self.current_account_balance_curr = self.get_margin_account_details(curr=self.current_traded_coin)["total"].values[0]
            self.current_available_account_balance_curr = self.get_margin_account_details(curr=self.current_traded_coin)["available"].values[0]

            self.balances = self.ledger_data["balances"]

            if self.balances["trailing_balances"]["total_balance"] <= 1:
                self.logger.info(f"{'#'*1}   Account balance is 0. Exiting backtest   {'#'*1}")
                self.save_performance_file(data="performance", file=self.trade_performance, filename=self.performance_filename)
                self.save_performance_file(data="positions", file=self.trade_positions, filename=self.performance_filename)
                self.strategy_data.to_csv(os.path.join(Trading_path, f"strategy_data_{self.performance_filename}.csv"))
                sys.exit()
            else:
                
                self.balances["trailing_balances"]["total_balance"] = self.calculate_balance_in_fiat(fiat=self.fiat, coin=self.current_traded_coin, getSum=True, get_only_trade_pair_bal=True, account_type="margin")
            
            self.logger.info(f"# STATUS SCANNER:          Trailing total balance is: {np.round(trailing_position,3)}         TOTAL PNL: {np.round((trailing_position/self.balances['initial_balance'][self.fiat]-1)*100,3)}%         WINNING TRADES: {self.trading_metrics_data['winning_trades']}         LOSING TRADES: {self.trading_metrics_data['losing_trades']}   {'#'*1}")

            self.trade_performance = pd.concat([self.trade_performance,pd.DataFrame([self.balances["trailing_balances"]], index=[datetime_input_adj])], axis=0)
            self.trade_performance.loc[datetime_input_adj, "price"] = self.get_price(self.current_traded_symbol)
            self.trade_performance.loc[datetime_input_adj, "trailing_balance"] = self.balances["trailing_balances"]["total_balance"]
            self.trade_performance.loc[datetime_input_adj, "winning_trades"] = self.trading_metrics_data["winning_trades"]
            self.trade_performance.loc[datetime_input_adj, "losing_trades"] = self.trading_metrics_data["losing_trades"]
            self.trade_performance.loc[datetime_input_adj, "largest_loss"] = self.trading_metrics_data["largest_loss"]
            self.trade_performance.loc[datetime_input_adj, "largest_gain"] = self.trading_metrics_data["largest_gain"]
            self.trade_performance.loc[datetime_input_adj, "total_pnl"] = (self.balances["trailing_balances"]["total_balance"]/self.balances['initial_balance'][self.fiat]-1)*100
                        
            if self.has_position:
                self.trade_positions = self.trade_positions[~self.trade_positions.index.duplicated(keep='last') | (self.trade_positions.index.duplicated(keep="last") & ~self.trade_positions.duplicated(subset=['close_order_id'], keep=False) & self.trade_positions['close_order_id'].notna())]
                position_df = pd.DataFrame([position], index=[datetime_input])
                if pd.Timestamp(position_df.time_opened.values[0]) == datetime_input and position_df["close_order_id"].values[0] not in self.trade_positions["close_order_id"].values:
                    self.trade_positions = pd.concat([self.trade_positions, position_df], axis=0) 
                else:
                    self.trade_positions = self.trade_positions.combine_first(position_df)
            else:
                if self.position_close == datetime_input:
                    if position is not None:
                        position_df = pd.DataFrame([position], index=[datetime_input])
                        self.trade_positions = self.trade_positions[~self.trade_positions.index.duplicated(keep='last') | (self.trade_positions.index.duplicated(keep="last") & ~self.trade_positions.duplicated(subset=['close_order_id'], keep=False) & self.trade_positions['close_order_id'].notna())]
                        if pd.Timestamp(position_df.time_opened.values[0]) == datetime_input and position_df["close_order_id"].values[0] not in self.trade_positions["close_order_id"].values:
                            self.trade_positions = pd.concat([self.trade_positions, position_df], axis=0) 
                        else:
                            self.trade_positions = self.trade_positions.combine_first(position_df)
                else:
                    self.trade_positions = self.trade_positions.combine_first(pd.DataFrame([None], index=[datetime_input]))
            
            self.ledger_data["balances"]["trailing_balances"][self.fiat] = self.current_available_account_balance_curr
            self.ledger_data["balances"]["trailing_balances"][self.current_traded_coin] = self.current_available_account_balance_coin
            self.ledger_data["balances"]["trailing_balances"]["total_balance"] = self.calculate_balance_in_fiat(fiat=self.fiat, coin=self.current_traded_coin, getSum=True, get_only_trade_pair_bal=True, account_type="margin")

            trailing_position = self.balances["trailing_balances"]["total_balance"]

            self.logger.info(f"# STATUS SCANNER: Trailing total balance is: {trailing_position}, TOTAL PNL: {(trailing_position/self.balances['initial_balance'][self.fiat]-1)*100}, WINNING TRADES: {self.trading_metrics_data['winning_trades']}, LOSING TRADES: {self.trading_metrics_data['largest_loss']}   {'#'*1}")
            
            if actual_datetime.hour == 0 and actual_datetime.minute == 0:
                if "timestamp_opened" in self.trade_positions.columns:
                    self.trade_positions["condition_index"] = self.trade_positions["timestamp_opened"].apply(self.get_conditions_for_hour)
                
                self.save_performance_file(data="performance", file=self.trade_performance, filename=self.performance_filename)
                self.save_performance_file(data="positions", file=self.trade_positions, filename=self.performance_filename)
                self.strategy_data.to_csv(os.path.join(Trading_path, f"strategy_data_{self.performance_filename}.csv"))

        except Exception as e:
            self.logger.error(f"An error occured during the execution of the status scanner function. Error:  {e}")
            raise Exception(f"An error occured during the execution of the status scanner function. Error:  {e}")

    def control_signal(self, datetime_input=None):
        if self.interval == "1h":
            if self.last_signal_datetime is not None and datetime_input - timedelta(hours=1) == self.last_signal_datetime:
                return True
            else:
                return False
        elif self.interval == "1d":
            if self.last_signal_datetime is not None and datetime_input - timedelta(days=1) == self.last_signal_datetime:
                return True
            else:
                return False
        elif self.interval == "5m":
            if self.last_signal_datetime is not None and datetime_input - timedelta(minutes=5) == self.last_signal_datetime:
                return True
            else:
                return False
            
    def get_conditions_for_hour(self, timestamp):
        timestamp = pd.Timestamp(timestamp) if isinstance(timestamp, str) else timestamp
        if pd.notna(timestamp) and isinstance(timestamp, pd.Timestamp):
            conditions = self.strategy_data.loc[
                (self.strategy_data.index.date == timestamp.date()) & 
                (self.strategy_data.index.hour == timestamp.hour) & self.strategy_data.triggered==True, "condition"
            ].tolist()
            conditions = [item for item in conditions if item is not None]
            return conditions[0] if len(conditions) > 0 else None
        return None

#################################################################################################################################################################
#
#                                                            performance functions
#
#################################################################################################################################################################


    def configure_performance_file(self, data=None, datetime_input=None, balances=True, positions=True, trades=False, filename=None):
        trade_performance = None
        trade_positions = None
        actual_trades = None

        if self.interval == "1h":
            datetime_input_tm1 = datetime.strptime(datetime_input, "%Y-%m-%d %H:%M:%S") - timedelta(hours=1)
        if self.interval == "1d":
            datetime_input_tm1 = datetime.strptime(datetime_input, "%Y-%m-%d %H:%M:%S") - timedelta(days=1)
        if self.interval == "5m":
            datetime_input_tm1 = datetime.strptime(datetime_input, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=5)
        if self.interval == "15m":
            datetime_input_tm1 = datetime.strptime(datetime_input, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=15)

        if balances:
            trade_performance = pd.DataFrame([self.ledger_data["balances"]["trailing_balances"]], index=[datetime_input])
            if "close" in self.data.columns:
                column_name = "close" 
            elif "Close" in self.data.columns:
                column_name = "Close"
            elif any("price_" in col for col in self.data.columns):
                column_name = next(col for col in self.data.columns if "price_" in col)
            
            if datetime_input_tm1 in self.data.index:
                trade_performance.loc[datetime_input, "price_tm1"] = self.data[column_name][datetime_input_tm1]
            else:
                trade_performance.loc[datetime_input, "price_tm1"] = None

        if positions:
            trade_positions = pd.DataFrame(self.ledger_data["positions"], index=[datetime_input])

        if trades:
            actual_trades = self.evaluate_trades()

        # Save the trade performance
        if trade_performance is not None:
            path = os.path.join(Trading_path, f"trade_performance_{self.coin}_{self.fiat}_{filename}.csv")            
            trade_performance.to_csv(path)

        # Save the trade positions
        if trade_positions is not None:
            path = os.path.join(Trading_path, f"trade_positions_{self.coin}_{self.fiat}_{filename}.csv")
            trade_positions.to_csv(path)
            
        # Save the actual trades
        if actual_trades is not None:
            path = os.path.join(Trading_path, f"actual_trades_{self.coin}_{self.fiat}_{filename}.csv")
            actual_trades.to_csv(path)
        
        # Return based on the requested data
        if data == "balances":
            return trade_performance
        elif data == "positions":
            return trade_positions
        elif data == "trades":
            return actual_trades
        elif data == "balances_positions":
            return trade_performance, trade_positions
        elif data == "balances_trades":
            return trade_performance, actual_trades
        elif data == "positions_trades":
            return trade_positions, actual_trades
        else:
            return trade_performance, trade_positions, actual_trades
        
    def save_performance_file(self, data=None, file=None, filename=None):
        """
        Saves performance data (DataFrame) to Google Drive without using a CSV buffer.
        """
        # Remove duplicate rows, keeping the last occurrence
        if isinstance(file, list):
            [item.drop_duplicates(keep='last', inplace=True) for item in file]
        else:
            file.drop_duplicates(keep='last', inplace=True)

        if data == "performance":
            path = os.path.join(Trading_path, f"trade_performance_{self.coin}_{self.fiat}_{filename}.csv")
            file.to_csv(path)
        elif data == "positions":
            path = os.path.join(Trading_path, f"trade_positions_{self.coin}_{self.fiat}_{filename}.csv")
            file.to_csv(path)
        elif data == "trades":
            path = os.path.join(Trading_path, f"actual_trades_{self.coin}_{self.fiat}_{filename}.csv")
            file.to_csv(path)
        else:
            # Save multiple DataFrames at once
            file1, file2, file3 = file
            
            # Save trade performance file
            
            path1 = os.path.join(Trading_path, f"trade_performance_{self.coin}_{self.fiat}_{filename}.csv")
            file1.to_csv(path1)
            
            # Save trade positions file
            path2 = os.path.join(Trading_path, f"trade_positions_{self.coin}_{self.fiat}_{filename}.csv")
            file2.to_csv(path2)
            
            # Save actual trades file
            path3 = os.path.join(Trading_path, f"Actual_trades_{self.coin}_{self.fiat}_{filename}.csv")
            file3.to_csv(path3)
            
    def configure_logger(self):
        """
        Configures a logger and uploads the log files to Google Drive.
        """

        # Create the log directory and log file name
        logger_path = utils.find_logging_path()  # Assuming this finds the path to where logs are stored locally
        current_datetime = dt.datetime.now()
        timestamp = current_datetime.strftime('%d-%m-%Y_%H-%M')
        
        log_directory = f"Algo Trader"
        log_file_name = f"Algo_trader_log_{timestamp}.log"
        self.log_file_path = os.path.join(logger_path, log_directory, log_file_name)
        
        
        if not os.path.exists(os.path.join(logger_path, log_directory)):
            os.makedirs(os.path.join(logger_path, log_directory))
        
        logger.add(self.log_file_path, rotation="500 MB", level="DEBUG")
        
        log_directory_cloud = f"Logging/Algo Trader"
        log_file_name = f"Algo_trader_log_{timestamp}.log"
        self.log_file_path_algo_trader_cloud = f"{log_directory_cloud}/{log_file_name}"
        
        try:
            # Ensure log directory exists on Google Drive
            self.gutils.create_directory_if_not_exists(log_directory_cloud )
            
            # Upload the log file to Google Drive
            self.gutils.upload_log_to_google_drive(self.log_file_path, self.log_file_path_algo_trader_cloud)
        except Exception as e:
            self.logger.error(f"Error: {e}")
            self.logger.debug(f"Exception details: {e}", exc_info=True)

#################################################################################################################################################################
#                                                               MAIN RUN
#################################################################################################################################################################

    @logger.catch
    def run_trader(self):
        last_scanner_minute = None
        n_retries = 1
        times_checked = 0
        last_trade_timestep = None
        testing = True
        constant_run = True
        log_date_check = None
        
        if self.ledger_data["current_trades"][self.current_traded_symbol] is not None:
            self.has_position = True
        else:
            self.has_position = False
        
        try:
            while True:
                try:
                    times_checked = 0 
                    current_time = dt.datetime.now()
                    # current_time = dt.datetime(2024,8,15,22,0,10)
                    current_time = current_time.replace(microsecond=0) #.strftime("%Y-%m-%d %H:%M:%S")
                    current_minute = current_time.replace(second=0, microsecond=0) #.strftime("%Y-%m-%d %H:%M:%S")
                    
                    
                    # Check if the next day is reached and create a new log file
                    if log_date_check is None:
                        log_date_check = current_time.date()

                    if current_time.date() != log_date_check:
                        self.logger.info("New day reached, creating a new log file")
                        self.logger = logger
                        self.configure_logger()
                        log_date_check =  current_time.date() 


                    # Calculate the next hour start and next hour + 5 seconds
                    if self.interval == "1h":
                        current_timestep = current_time.replace(minute=0, second=0, microsecond=0)
                        # Execute trade every hour plus 5 seconds
                    elif self.interval == "1d":
                        current_timestep = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                    elif self.interval == "5m":
                        last_5_minute = current_time.minute - (current_time.minute % 5)
                        current_timestep = current_time.replace(minute=last_5_minute, second=0, microsecond=0)
                    elif self.interval == "15m":
                        last_15_minute = current_time.minute - (current_time.minute % 15)
                        current_timestep = current_time.replace(minute=last_15_minute, second=0, microsecond=0)
                    
                    # Execute status_scanner every minute if there is a position
                    if self.has_position and last_scanner_minute != current_minute:
                        self.logger.info("Scanning status at time {}".format(current_minute))
                        self.status_scanner(datetime_input=current_minute)
                        last_scanner_minute = current_minute
                    
                    # Update next_hour_plus_5_seconds only when entering a new hour
                    if last_trade_timestep != current_timestep:
                        if self.interval == "1h":
                            next_timestep = current_timestep + dt.timedelta(hours=1)
                            next_timestep_plus_5_seconds = next_timestep + dt.timedelta(seconds=5)
                        elif self.interval == "1d":
                            next_timestep = current_timestep + dt.timedelta(days=1)
                            next_timestep_plus_5_seconds = next_timestep + dt.timedelta(seconds=5)
                        elif self.interval == "5m":
                            next_timestep = current_timestep + dt.timedelta(minutes=5)
                            next_timestep_plus_5_seconds = next_timestep + dt.timedelta(seconds=5)
                        elif self.interval == "15m":
                            next_timestep = current_timestep + dt.timedelta(minutes=15)
                            next_timestep_plus_5_seconds = next_timestep + dt.timedelta(seconds=5)
                        
                        times_checked = 0
                        last_trade_timestep = current_timestep
                    
                    # Execute trade every hour plus 5 seconds
                    if (times_checked == 0 and current_time < next_timestep_plus_5_seconds) or (current_time >= next_timestep_plus_5_seconds and times_checked < n_retries):
                        #reinit security token 
                        self.logger.info("Evaluating trade signal at time {}".format(current_time.strftime("%Y-%m-%d %H:%M:%S")))
                        self.gutils = GoogleDriveUtility(parent_folder="AlgoTrader Drive")
                        self.trade(datetime_input=current_timestep)
                        times_checked += 1
                        self.logger.info(f"Trade signal evaluated {times_checked}/{n_retries}")
                    elif constant_run and times_checked < n_retries:   
                        self.logger.info("Evaluating trade signal at time {}".format(current_time.strftime("%Y-%m-%d %H:%M:%S")))
                        self.gutils = GoogleDriveUtility(parent_folder="AlgoTrader Drive")
                        self.trade(datetime_input=current_timestep)
                        times_checked += 1
                        self.logger.info(f"Trade signal evaluated {times_checked}/{n_retries}")
                    
                    # self.logger.info("Uploading log to drive")
                    self.gutils.upload_log_to_google_drive(self.log_file_path, self.log_file_path_algo_trader_cloud)
                    
                    if self.has_position or times_checked <= 1 or constant_run:
                        current_time = dt.datetime.now()
                        # Calculate the next 30-second tick
                        if current_time.second < 30:
                            next_tick = current_time.replace(second=30, microsecond=0)
                        else:
                            next_tick = (current_time + timedelta(minutes=1)).replace(second=0, microsecond=0)
                        
                        time_remaining = (next_tick - current_time).total_seconds()
                        self.logger.info(f"Waiting for next 30s tick: {int(time_remaining)} seconds")
                        time.sleep(time_remaining)
                        times_checked += 1
                    else:
                        time_remaining = next_timestep - dt.datetime.now()
                        self.logger.info(f"Waiting for next hour: {time_remaining}")
                        time.sleep(time_remaining.total_seconds())
                except Exception as e:
                    self.logger.error(f"PLEASE CHECK: Error when running the trading bot, PLEASE CHECK: {e}")
                    time.sleep(60)  
                    
        except Exception as e:
            self.logger.error(f"PLEASE CHECK Error: {e} ")
            self.logger.debug(f"Exception details: {e}", exc_info=True)
            self.gutils.upload_log_to_google_drive(self.log_file_path, self.log_file_path_algo_trader_cloud)
            
# a = AlgoTrader()

# from main_api_trader import MainApiTrader
# from TradeBacktester import Backtester
# Define the global exception handler
def handle_exception(exc_type, exc_value, exc_traceback, algo_trader_instance):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    algo_trader_instance.gutils.upload_log_to_google_drive(algo_trader_instance.log_file_path, algo_trader_instance.log_file_path_algo_trader_cloud)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    import time
    freeze_support()
    
    #init class
    AlgoTraderKucoin = AlgoTrader()   #MainApiTrader, DataLoader
    
    # Set the global exception handler
    sys.excepthook = lambda exc_type, exc_value, exc_traceback: handle_exception(exc_type, exc_value, exc_traceback, AlgoTraderKucoin)
    
    #run main 
    AlgoTraderKucoin.run_trader()
    

    print("test completed")

# # test = a.test(KucoinTrader)
# trade_test = a.trade()
# print("test completed")