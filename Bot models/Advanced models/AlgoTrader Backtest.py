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

import mo_utils as utils
from utilsGoogleDrive import GoogleDriveUtility
from DataLoader import DataLoader
from TradeBacktester import Backtester
from StrategyEvaluator import StrategyEvaluator
from TAIndicatorCreator import TAIndicatorCreator
from KuCoin_Prices import KuCoinDataDownloader


class AlgoTrader_backtest(Backtester, DataLoader):

    def __init__(self):
        
        
        #config
        config_path = utils.find_config_path() 
        config = utils.read_config_file(os.path.join(config_path,"AlgoTrader_backtest_config.ini"))
        
        self.delete_trade_histo = utils.get_config_value(config, "backtest", "delete_trade_histo")
        
        if self.delete_trade_histo:
            if os.path.exists(os.path.join(Trading_path, "trading_ledger_backtest.json")):
                os.remove(os.path.join(Trading_path, "trading_ledger_backtest.json"))
            if os.path.exists(os.path.join(Trading_path, "trading_metrics_AlgoTrader.json")):
                os.remove(os.path.join(Trading_path, "trading_metrics_AlgoTrader.json"))

        #backtest config
        self.simulate_live_trading = utils.get_config_value(config, "backtest", "simulate_live_trading")
        self.backtest_filename = utils.get_config_value(config, "backtest", "backtest_filename")
        
        #trading pair
        self.coin = utils.get_config_value(config, "general", "coin")
        self.fiat = utils.get_config_value(config, "general", "currency")
        self.symbol = f"{self.coin}-{self.fiat}"
        self.interval =  utils.get_config_value(config, "general", "interval")

        #date input 
        start_date_str = utils.get_config_value(config, "backtest", "start_date")
        end_date_str = utils.get_config_value(config, "backtest", "end_date")

        self.start_datetime = pd.to_datetime(start_date_str, format="%Y-%m-%d %H:%M:%S")
        self.end_datetime = pd.to_datetime(end_date_str, format="%Y-%m-%d %H:%M:%S")

        self.current_datetime = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        #logger
        self.logger = logger
        self.configure_logger()

        #slippage
        self.slippage_factor = utils.get_config_value(config, "backtest", "slippage")
        self.leverage_factor = utils.get_config_value(config, "trading", "leverage_factor")
        self.atr_factor_trending = utils.get_config_value(config, "trading", "atr_factor_trending")
        self.atr_factor_ranging = utils.get_config_value(config, "trading", "atr_factor_ranging")
        
        # init kucoin trader (Backtest)   (also init self.ledger_data)
        Backtester.__init__(self, coin=self.coin, currency=self.fiat, slippage=self.slippage_factor, leverage=self.leverage_factor,logger_input=self.logger)
        
        #init kucoin price loader
        self.kucoin_price_loader = KuCoinDataDownloader(created_logger=self.logger)
        self.kucoin_price_loader.download_data(self.coin, self.fiat, self.interval, end_date=self.end_datetime, use_local_timezone=True, drop_not_complete_timestamps=False)   #download interval data in advance to avoid constant downloads
        
        if self.simulate_live_trading:
            self.kucoin_price_loader.download_data(self.coin, self.fiat, "1m", start_date=self.start_datetime, end_date=self.end_datetime, use_local_timezone=True, drop_not_complete_timestamps=False)   #download minute data in advance to avoid constant downloads
            self.logger.info(f"{'#'*1}   Simulating live trading. Minute data downloaded   {'#'*1}")
        #init Strategy Evaluator
        self.strategy_evaluator = StrategyEvaluator(logger=self.logger)
        
        filename = utils.get_config_value(config, "general", "dataset_filename")
        self.use_data_loader = utils.get_config_value(config, "general", "use_data_loader")
        self.use_multi_timeframe = utils.get_config_value(config, "general", "use_multi_timeframe")
        self.multi_timeframe = utils.get_config_value(config, "general", "multi_timeframe") if self.use_multi_timeframe else None
        self.use_atr = utils.get_config_value(config, "general", "use_atr")
        self.can_switch_sides = utils.get_config_value(config, "general", "can_switch_sides")
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
                    self.data, minute_data = self.kucoin_price_loader.download_data(self.coin, self.fiat, self.interval, start_date="2017-01-01 00:00:00", use_local_timezone=True)
                    
                    for interval in ["1d"]:
                        self.kucoin_price_loader.download_data(self.coin, self.fiat, interval, start_date="2017-01-01 00:00:00", use_local_timezone=True)
                    
                    self.data.index = pd.to_datetime(self.data.index)
                
                self.data.index = pd.to_datetime(self.data.index)
                
                if self.data.index[-1] <= self.start_datetime:
                    self.data, minute_data = self.kucoin_price_loader.download_data(self.coin, self.fiat, self.interval, start_date="2017-01-01 00:00:00", use_local_timezone=True)
                    for interval in ["1d"]:
                        self.kucoin_price_loader.download_data(self.coin, self.fiat, interval, start_date="2017-01-01 00:00:00", use_local_timezone=True)
                    
                    self.data.index = pd.to_datetime(self.data.index)
                
                self.data[f"price_{self.fiat.lower()}_{self.ticker_to_slug_mapping[self.coin].lower()}"] = self.data["close"]    
                # self.data = self.data.rename(columns={"close":f"price_{self.fiat.lower()}_{self.ticker_to_slug_mapping[self.coin].lower()}"})
                
            else:
                self.data, minute_data = self.kucoin_price_loader.download_data(self.coin, self.fiat, self.interval, start_date="2017-01-01 00:00:00", use_local_timezone=True)
                
                for interval in ["1d"]:
                        self.kucoin_price_loader.download_data(self.coin, self.fiat, interval, start_date="2017-01-01 00:00:00", use_local_timezone=True)
                
                self.data[f"price_{self.fiat.lower()}_{self.ticker_to_slug_mapping[self.coin].lower()}"] = self.data["close"]
                # self.data = self.data.rename(columns={"close":f"price_{self.fiat.lower()}_{self.ticker_to_slug_mapping[self.coin].lower()}"})

                # timestamp input is self.end_date in real calc use timestamp_input=timestam_input
            
        self.data.index = pd.to_datetime(self.data.index, format="%Y-%m-%d %H:%M:%S")

        if self.data.index[1]> self.start_datetime:
            self.start_datetime = self.data.index[1]
            
        # if self.data.index[-1]< self.end_datetime:
        #     self.end_datetime = self.data.index[-1]

        #balance coin
        self.balances = self.ledger_data["balances"]   #get ledger and balance data from TradeBacktester
        self.current_account_balance_coin = self.balances["trailing_balances"][self.coin]
        self.current_available_account_balance_coin = self.balances["trailing_balances"][self.coin]
        
        #balance curr
        self.current_account_balance_curr = self.balances["trailing_balances"][self.fiat]
        self.current_available_account_balance_curr = self.balances["trailing_balances"][self.fiat]
        
        #value coin in currency
        #self.current_coin_value_in_fiat = self.calculate_balance_in_fiat("EUR", self.coin, getSum=True)    <-- not required now, might be added to a later point in time
        
        self.current_coin_value_in_currency = self.calculate_balance_in_fiat(data=self.data, datetime_input=self.start_datetime,coin=self.coin,fiat=self.fiat, getSum=True)
        self.ledger_data["balances"]["trailing_balances"]["total_balance"] = self.current_account_balance_curr + self.current_coin_value_in_currency
        
        #trade settings
        self.taker_trade_fee = self.get_trading_fees(coin=self.coin, side="buy")
        self.maker_trade_fee = self.get_trading_fees(coin=self.coin, side="sell")
        self.max_open_positions = utils.get_config_value(config, "trading", "max_open_positions")
        # current borrow fee
        
        #ratios:
        self.investment_per_trade = utils.get_config_value(config, "trading", "investment_per_trade")
        self.stop_loss_percentage = utils.get_config_value(config, "trading", "stop_loss_percentage")
        self.take_profit_percentage = utils.get_config_value(config, "trading", "take_profit_percentage")
        self.leverage_factor = utils.get_config_value(config, "trading", "leverage_factor")
        
        # Account variables
        self.positions = self.ledger_data["positions"]
        
        if len(self.ledger_data["positions"]) ==0:
            initial_balance_value = self.current_account_balance_curr
            self.ledger_data["balances"]["initial_balance"][self.fiat] = initial_balance_value
            self.ledger_data["balances"]["initial_balance"][self.coin] = {"fiat_value":(self.current_account_balance_coin *self.get_price(data=self.data, datetime_input=self.start_datetime)),"coin_amt":self.current_account_balance_coin}
            self.ledger_data["balances"]["total_balance"] = initial_balance_value
            self.initial_balance_curr = initial_balance_value
            self.initial_balance_coin_in_curr = self.current_account_balance_coin*self.get_price(data=self.data, datetime_input=self.start_datetime, activate_slippage=False)
            self.initial_balance_coin = self.current_account_balance_coin
        else:
            self.initial_balance_curr = self.ledger_data["balances"]["initial_balance"][self.fiat]
            self.initial_balance_coin_in_curr =self.ledger_data["balances"]["initial_balance"][self.coin]["fiat_value"]
            self.initial_balance_coin = self.ledger_data["balances"]["initial_balance"][self.coin]["coin_amt"]
            self.ledger_data["balances"]["total_balance"] = self.initial_balance_curr + self.initial_balance_coin_in_curr

        self.pnl_account_abs = (self.current_account_balance_curr + self.get_price(data=self.data, datetime_input=self.start_datetime, activate_slippage=False)*self.current_account_balance_coin) - (self.initial_balance_curr + self.initial_balance_coin_in_curr)
        self.pnl_account_pct = ((self.current_account_balance_curr + self.get_price(data=self.data, datetime_input=self.start_datetime, activate_slippage=False)*self.current_account_balance_coin) - (self.initial_balance_curr + self.initial_balance_coin_in_curr)) / (self.initial_balance_curr + self.initial_balance_coin_in_curr)
        self.total_profit_balance = max(((self.current_account_balance_curr + self.get_price(data=self.data, datetime_input=self.start_datetime, activate_slippage=False)*self.current_account_balance_coin)- (self.initial_balance_curr + self.initial_balance_coin_in_curr)),0)
    
        self.logger.info(f"{'#'*1}   Initial balance in {self.fiat} is: {self.initial_balance_curr}   {'#'*1}")
        self.logger.info(f"{'#'*1}   Initial balance in {self.coin} is: {self.initial_balance_coin_in_curr}   {'#'*1}")

        self.backtest_performance = self.configure_backtest_performance_file(data="balances", datetime_input=self.start_datetime, filename=self.backtest_filename)
        self.backtest_positions = self.configure_backtest_performance_file(data="positions", datetime_input=self.start_datetime, filename=self.backtest_filename)
        self.strategy_data = pd.DataFrame()

        self.last_signal_datetime = None
        self.last_cooldown_time  = None
        self.trade_cooldown = 0
        self.position_close = None

        #for TA trading strategy initialize the TA creator signal class again and calc TA       
        # self.TA_creator.calculate_indicator("usdt_kucoin", coin=self.coin, fiat=self.fiat, interval=self.interval, 
        #                                timestamp_input=self.end_datetime, indicator_name='macd', overwrite_file=True, plot=True, 
        #                                macd={'short_ema_span': 12, 'long_ema_span': 26, 'signal_span': 9}) 
        
        #clear previous trades
        # self.ledger_data["current_trades"][self.symbol] = None

#################################################################################################################################################################
#
#                                                                  Algo trader functions
#
#################################################################################################################################################################

    # def StrategyEvaluator(self, data=None, timestamp_input=None):
    #     return evaluator.ema_cross_strategy(data=data, timestamp_input=timestamp_input, short_ema=21, long_ema=50, config=None)
    
    def trade(self, datetime_input=None):
        
        actual_datetime = datetime_input
        stop_loss_price = None
        take_profit_price = None
        coin_to_trade = None
        
        if datetime_input.minute == 59:
            print("Reached end of hour. 1m close now matching hour close")
        
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
                    

        #strategy ma hl and rsi divergence with volume    not profitable on 1h and 5 min
        # signal_ma_hl, atr, condition_index, strategy_data = self.strategy_evaluator.ma_hl_div_vol(self.coin,self.fiat, self.interval, self.interval, datetime_input, actual_datetime, self.logger, simulate_live_data=self.simulate_live_trading, verbose=True)   #datetime_input   #pass tm1 value as as current timestamp is not available yet
        # signal_strategy = "ma_hl_divergence_volume"
        # signal = signal_ma_hl

        #strategy ma hl and rsi divergence   not profitable on 1h and 5 min
        signal_ma_hl, atr, condition_index, strategy_data = self.strategy_evaluator.ma_hl_div(self.coin,self.fiat, self.interval, self.interval, datetime_input, actual_datetime, self.logger, simulate_live_data=self.simulate_live_trading, verbose=True)   #datetime_input   #pass tm1 value as as current timestamp is not available yet
        signal_strategy = "ma_hl_divergence"
        signal = signal_ma_hl

        
        #strategy fibonacci retracement     not profitable on 1h and 5 min
         # signal_fib, atr, stop_loss_price, take_profit_price = self.strategy_evaluator.fibonacci_retracement_strategy(self.coin,self.fiat, self.interval, self.interval, datetime_input, actual_datetime, self.logger, simulate_live_data=self.simulate_live_trading, verbose=True)   #datetime_input   #pass tm1 value as as current timestamp is not available yet
        # signal_strategy = "fibonacci_retracement"
        # signal = signal_fib

        # signal_ma_hl, atr, condition_index, strategy_data = self.strategy_evaluator.ma_high_low_rsi_5m_strategy(self.coin,self.fiat, self.interval, self.interval, datetime_input, actual_datetime, self.logger, simulate_live_data=self.simulate_live_trading, verbose=True)   #datetime_input   #pass tm1 value as as current timestamp is not available yet
        # signal_strategy = "ma_high_low_rsi_5m_strategy"
        # signal = signal_ma_hl

        # signal_waverider, atr, condition_index, strategy_data = self.strategy_evaluator.waverider_indicator(self.coin,self.fiat, self.interval, datetime_input, actual_datetime, self.logger, simulate_live_data=self.simulate_live_trading, verbose=True)   #datetime_input   #pass tm1 value as as current timestamp is not available yet
        # signal_strategy = "waverider test"
        # signal = signal_waverider



        # candle pattern
        # signal_candles, atr, condition_index, strategy_data = self.strategy_evaluator.candle_pattern(self.coin,self.fiat, self.interval, datetime_input, actual_datetime, self.logger, simulate_live_data=self.simulate_live_trading, verbose=True)   #datetime_input   #pass tm1 value as as current timestamp is not available yet
        # signal_strategy = "candles"
        # signal = signal_candles
        
        #kama strategy not profitable at 1h
        # signal_kama, atr, condition_index, strategy_data = self.strategy_evaluator.kama_boll(self.coin,self.fiat, self.interval, datetime_input, actual_datetime, self.logger, simulate_live_data=self.simulate_live_trading, verbose=True)   #datetime_input   #pass tm1 value as as current timestamp is not available yet
        # signal_strategy = "kama"
        # signal = signal_kama

        self.strategy_data = pd.concat([self.strategy_data, strategy_data], axis=0)

        # Check if signal is already received
        if self.last_signal_datetime is not None and self.last_signal_datetime == datetime_input and condition_index in self.trade_restrictions_conditions:
            self.logger.info(f"{'#'*1}   Signal already received for this hour. Waiting for next update   {'#'*1}")
            return
        # elif self.last_signal_datetime is not None and datetime_input == self.position_close and condition_index in self.trade_restrictions_conditions:
        #     self.logger.info(f"{'#'*1}   Position already closed this hour. Waiting for next update   {'#'*1}")
        #     return

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
            return

        else:
            coin_to_trade = self.coin if coin_to_trade is None else coin_to_trade
            symbol_to_trade = f"{coin_to_trade}-{self.fiat}"
            
            self.logger.info(f"     ")
            self.logger.info(f"{'#'*30}  Signal received for side: {signal} at time {datetime_input}  {'#'*30}")
            self.logger.info(f"     ")            

            #check for open positions or orders
            has_open_order = False
            has_open_position = False

            #balance curr
            self.current_account_balance_curr = self.balances["trailing_balances"][self.fiat]
            self.current_account_balance_coin = self.balances["trailing_balances"][coin_to_trade]

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

                initial_order_id_details = self.ledger_data["order_details"][current_order_id]["initial_order_details"] 
                check_order_triggered = self.check_order_triggered(data=self.data,datetime_input=datetime_input,order_details=initial_order_id_details, order_type=current_position["side"])
                
                if check_order_triggered:
                    has_open_order = False
                else:
                    has_open_order = True
            else:
                #track last signal and signal 
                self.trade_cooldown = self.trade_cooldown_counter+1 if (condition_index in self.cooldown_condtions) else 1    #enforce cooldown of 5 timesteps
                self.last_signal_datetime = datetime_input 
                self.last_signal_side = signal
                self.last_cooldown_time = datetime_input
                self.position_close = None

            price = self.get_price(coin=coin_to_trade, data=self.data, datetime_input=datetime_input, activate_slippage=False)

            #implement check that when atr is less than order cost no trade is executed
            if self.interval in ["1m","5m","15m","30m"] and self.use_atr:
                atr_factor = self.atr_factor_trending if condition_index in [0,1,2,3,4,5,6,7] else self.atr_factor_ranging
                potential_payoff = (((price - (atr_factor*min(atr, 750))) / price) -1) * self.ledger_data["balances"]["trailing_balances"][self.fiat]
                potential_fees = self.get_trading_fees(coin=coin_to_trade, side=signal)*self.ledger_data["balances"]["trailing_balances"][self.fiat]
                if potential_payoff > potential_fees:
                    self.logger.info(f"{'#'*1}   Potential payoff is less than fees. No trade executed.  {'#'*1}")
                    return
            
            if self.use_atr:
                if signal == "buy":
                    if condition_index in []:    #for ma hl div vol use  [0,2,3]
                        stop_loss_price = price - self.atr_factor_trending*min(atr,750)    #self.atr_factor_trending
                        take_profit_price = price + self.atr_factor_trending*min(atr,750)
                    elif condition_index in [0,1,2]:    #for ma hl div vol use  [0,2,3]
                        stop_loss_price = price - min(atr,750)    #self.atr_factor_trending
                        take_profit_price = price + self.atr_factor_trending*min(atr,750)
                    elif condition_index in []:
                        stop_loss_price = (1-(self.stop_loss_percentage/100)) * price     #sl percentage is in absolute values and not in percentage (decimal)
                        take_profit_price = (1+(self.take_profit_percentage/100)) * price
                    elif condition_index in []:  #last candle low (doji)
                        stop_loss_price = price - 1.5*min(atr,750)#self.data["low"].loc[tm1]
                        take_profit_price = price + min(atr,750)   #self.atr_factor_trending*
                    elif condition_index in []:  #second engulfed candle high (engulfing) 3
                        stop_loss_price = max(self.data.loc[tm2,["open","close"]])
                        take_profit_price = price + min(atr,750)     #self.atr_factor_ranging
                    elif condition_index in []: #only 1 atr for both
                        stop_loss_price = price - min(atr,750)   #self.atr_factor_trending
                        take_profit_price = price + min(atr,750)
                    else:
                        stop_loss_price = price - min(atr,750)   #self.atr_factor_ranging
                        take_profit_price = price + self.atr_factor_ranging*min(atr,750)
                elif signal == "sell":
                    if condition_index in []: #for ma hl div vol use  [0,2,3]   1,2,3
                        stop_loss_price = price + self.atr_factor_trending*min(atr,750)                    #self.atr_factor_trending
                        take_profit_price = price - self.atr_factor_trending*min(atr,750)
                    elif condition_index in [0,1,2]: #for ma hl div vol use  [0,2,3]   1,2,3
                        stop_loss_price = price + min(atr,750)                    #self.atr_factor_trending
                        take_profit_price = price - self.atr_factor_trending*min(atr,750)
                    elif condition_index in []:
                        stop_loss_price = (1+(self.stop_loss_percentage/100)) * price       #sl percentage is in absolute values and not in percentage (decimal)
                        take_profit_price = (1-(self.take_profit_percentage/100)) * price 
                    elif condition_index in []:  #last candle high (doji)
                        stop_loss_price = price + 1.5*min(atr,750) #self.data["high"].loc[tm1]
                        take_profit_price = price - min(atr,750)    #self.atr_factor_ranging
                    elif condition_index in []:  #second engulfed candle low (engulfing) 3
                        stop_loss_price = min(self.data.loc[tm2,["open","close"]])
                        take_profit_price = price - min(atr,750)    #self.atr_factor_ranging
                    elif condition_index in []: #only 1 atr for both
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
                    take_profit_price = (1-(self.take_profit_percentage/100)) * price   #tp percentage is in absolute values and not in percentage (decimal)

            if signal == "buy" and has_open_position ==False and has_open_order == False:

                unlevered_funds = self.current_account_balance_curr *self.investment_per_trade
                funds = unlevered_funds * (self.leverage_factor-1)
                size=None

                self.enter_margin_trade_backtest(data=self.data, datetime_input=datetime_input, coin=coin_to_trade, fiat=self.fiat,
                                                size=size, funds=funds, balance_fiat= self.current_account_balance_curr, is_long=True, order_type="market",
                                                stop_price=stop_loss_price, take_profit_price=take_profit_price)

                self.ledger_data["current_trades"][symbol_to_trade]["current strategy"]  = signal_strategy
                self.strategy_data.loc[strategy_data.index[0],"triggered"] = True 
            
            if signal == "buy" and has_open_position==True and current_position["side"]=="sell" :
                if self.can_switch_sides and condition_index in [1]: #or condition_index == 1:
                 # and  signal_strategy == self.ledger_data["current_trades"][self.symbol]["current strategy"]:   #only considers changing sides when previous trade was sell, currently two consecutive buy orders are not permitted

                    self.close_margin_position_backtest(data=self.data, datetime_input=datetime_input, coin=coin_to_trade, fiat=self.fiat, current_position=current_position)
                    self.position_close = datetime_input #set position close time to add close position to ledger
                    
                    #balance curr
                    self.current_account_balance_curr = self.balances["trailing_balances"][self.fiat]
                    self.current_account_balance_coin = self.balances["trailing_balances"][coin_to_trade]

                    unlevered_funds = self.current_account_balance_curr *self.investment_per_trade
                    funds = unlevered_funds * (self.leverage_factor-1)
                    size=None

                    self.enter_margin_trade_backtest(data=self.data, datetime_input=datetime_input, coin=coin_to_trade, fiat=self.fiat,
                                                    size=size, funds=funds, balance_fiat= self.current_account_balance_curr, is_long=True, order_type="market",
                                                    stop_price=stop_loss_price, take_profit_price=take_profit_price)

                    self.ledger_data["current_trades"][symbol_to_trade]["current strategy"]  = signal_strategy
                    self.strategy_data.loc[strategy_data.index[0],"triggered"] = True 
                else:
                    self.logger.info("{'#'*1}   Received buy signal although already short. Not changing sides. Condition index not met. Waiting for next update   {'#'*1}")
            
            elif signal == "buy" and has_open_position==True and current_position["side"]=="buy":
                self.logger.info(f"{'#'*1}   Received buy signal although already long. Waiting for next update   {'#'*1}")
                pass
            
            elif signal == "buy" and has_open_position==True and current_position["side"]=="sell" and self.can_switch_sides==False:
                self.logger.info(f"{'#'*1}   Received buy signal although already short. Not changing sides. Waiting for next update   {'#'*1}")
            

            elif signal == "sell" and has_open_position ==False and has_open_order == False:
                
                unlevered_size = (self.current_account_balance_curr *self.investment_per_trade)  / price
                size = unlevered_size * (self.leverage_factor-1)
                funds=None

                self.enter_margin_trade_backtest(data=self.data, datetime_input=datetime_input, coin=coin_to_trade, fiat=self.fiat,
                                                size=size, funds=funds, balance_fiat= self.current_account_balance_curr, is_long=False, order_type="market",
                                                stop_price=stop_loss_price, take_profit_price=take_profit_price)

                self.ledger_data["current_trades"][symbol_to_trade]["current strategy"]  = signal_strategy
                self.strategy_data.loc[strategy_data.index[0],"triggered"] = True 

             # or condition_index == 1:     
            if signal == "sell" and has_open_position==True and current_position["side"]=="buy": #and  signal_strategy == self.ledger_data["current_trades"][self.symbol]["current strategy"]:
                if self.can_switch_sides and condition_index in [1]:
                    self.close_margin_position_backtest(data=self.data, datetime_input=datetime_input, coin=coin_to_trade, fiat=self.fiat, current_position=current_position)
                    self.position_close = datetime_input #set position close time to add close position to ledger

                    #balance curr
                    self.current_account_balance_curr = self.balances["trailing_balances"][self.fiat]
                    self.current_account_balance_coin = self.balances["trailing_balances"][coin_to_trade]

                    unlevered_size = (self.current_account_balance_curr *self.investment_per_trade)  / price
                    size = unlevered_size * (self.leverage_factor-1)
                    funds=None

                    self.enter_margin_trade_backtest(data=self.data, datetime_input=datetime_input, coin=coin_to_trade, fiat=self.fiat,
                                                    size=size, funds=funds, balance_fiat= self.current_account_balance_curr, is_long=False, order_type="market",
                                                    stop_price=stop_loss_price, take_profit_price=take_profit_price)

                    self.ledger_data["current_trades"][symbol_to_trade]["current strategy"]  = signal_strategy
                    self.strategy_data.loc[strategy_data.index[0],"triggered"] = True 
                else:
                    self.logger.info("{'#'*1}   Received sell signal although already long. Not changing sides. Condition index not met. Waiting for next update   {'#'*1}")
            elif signal == "sell" and has_open_position==True and current_position["side"]=="sell":
                
                self.logger.info(f"{'#'*1}   Received sell signal although already short. Waiting for next update   {'#'*1}")
                pass
            
            elif signal == "sell" and has_open_position==True and current_position["side"]=="buy" and self.can_switch_sides==False:
                self.logger.info(f"{'#'*1}   Received sell signal although already long. Not changing sides. Waiting for next update   {'#'*1}")
                pass

    # @staticmethod
    def status_scanner(self, datetime_input=None):
        
        actual_datetime = datetime_input
        
        #check for open positions or orders
        has_position = False
        
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
        
        if self.interval == "1h":
            datetime_input= datetime_input.replace(minute=0, second=0, microsecond=0)
            tm1 = (datetime_input - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        elif self.interval == "1d":
            datetime_input= datetime_input.replace(hours=0, minute=0, second=0, microsecond=0)
            tm1 = (datetime_input - timedelta(days=1)).replace(hours=0, minute=0, second=0, microsecond=0)
        elif self.interval == "5m":
            datetime_input= datetime_input.replace(second=0, microsecond=0).replace(minute=5 * (datetime_input.minute // 5))
            tm1 = (datetime_input - timedelta(minutes=5)).replace(second=0, microsecond=0)
        elif self.interval == "15m":
            datetime_input= datetime_input.replace(second=0, microsecond=0).replace(minute=15 * (datetime_input.minute // 15))
            tm1 = (datetime_input - timedelta(minutes=15)).replace(second=0, microsecond=0)
        
            
        if self.ledger_data["current_trades"][self.current_traded_symbol] is not None:             
            current_order_id = self.ledger_data["current_trades"][self.current_traded_symbol]["order_id"]
            position = self.get_position_by_order_id(current_order_id)
            has_position = True
        else:
            position = None
            
        if has_position:  #check if stop loss or take profit is triggered
            risk_prevention_status = self.check_sl_or_tp_triggered_backtest(coin=self.current_traded_coin, fiat=self.fiat, data=self.data, datetime_input=datetime_input, position=position) 
            if risk_prevention_status != "not triggered":
                has_position = False
                self.position_close = datetime_input
                
        last_date = self.data.index[-1]
        
        if last_date > datetime_input:   #adjust data to only go until current datetime
            self.data = self.data.loc[:datetime_input]
            last_date = self.data.index[-1]
        
        if last_date < actual_datetime:
            self.data, minute_data = self.kucoin_price_loader.download_data(self.current_traded_coin, self.fiat, self.interval, start_date=last_date, end_date=actual_datetime, 
                                                                use_local_timezone=True, drop_not_complete_timestamps=False, simulate_live_data=self.simulate_live_trading, overwrite_file=False)  
            
            self.data[f"price_{self.fiat.lower()}_{self.ticker_to_slug_mapping[self.current_traded_coin].lower()}"] = self.data["close"]
            # self.data = self.data.rename(columns={"close":f"price_{self.fiat.lower()}_{self.ticker_to_slug_mapping[self.coin].lower()}"})
            self.data.index = pd.to_datetime(self.data.index)
        
        

        self.current_account_balance_coin = self.balances["trailing_balances"][self.current_traded_coin]
        self.current_available_account_balance_coin = self.balances["trailing_balances"][self.current_traded_coin]
        
        #balance curr
        self.current_account_balance_curr = self.balances["trailing_balances"][self.fiat]
        self.current_available_account_balance_curr = self.balances["trailing_balances"][self.fiat]

        if self.balances["trailing_balances"]["total_balance"] <= 1:
            self.logger.info(f"{'#'*1}   Account balance is 0. Exiting backtest   {'#'*1}")
            self.save_backtest_performance_file(data="performance", file=self.backtest_performance, filename=self.backtest_filename)
            self.save_backtest_performance_file(data="positions", file=self.backtest_positions, filename=self.backtest_filename)
            sys.exit()
        else:
            self.balances["trailing_balances"]["total_balance"] = self.calculate_balance_in_fiat(data=self.data, datetime_input=datetime_input, fiat=self.fiat, getSum=True, get_only_trade_pair_bal=True)
            if has_position and position["side"] == "sell":
                coin_value = position["size"] * self.get_price(coin=self.current_traded_coin, data=self.data, datetime_input=datetime_input, activate_slippage=False)
                trailing_position = self.balances['trailing_balances'][self.fiat] +  (self.balances['trailing_balances'][self.fiat] - coin_value)
            else:
                trailing_position = self.balances['trailing_balances']["total_balance"]

        self.logger.info(f"# STATUS SCANNER:          Trailing total balance is: {np.round(trailing_position,3)}         TOTAL PNL: {np.round((trailing_position/self.balances['initial_balance'][self.fiat]-1)*100,3)}%         WINNING TRADES: {self.trading_metrics_data['winning_trades']}         LOSING TRADES: {self.trading_metrics_data['losing_trades']}   {'#'*1}")

        # Remove duplicate entries, keeping only the last one for each hour
        self.backtest_performance = self.backtest_performance[~self.backtest_performance.index.duplicated(keep='last')]
        self.backtest_performance = pd.concat([self.backtest_performance, pd.DataFrame([self.balances["trailing_balances"]], index=[datetime_input])], axis=0)
        self.backtest_performance.loc[datetime_input, "price"] = self.data[f"price_{self.fiat.lower()}_{self.ticker_to_slug_mapping[self.current_traded_coin].lower()}"][datetime_input]
        self.backtest_performance.loc[datetime_input, "trailing_balance"] = trailing_position
        self.backtest_performance.loc[datetime_input, "winning_trades"] = self.trading_metrics_data['winning_trades']
        self.backtest_performance.loc[datetime_input, "losing_trades"] = self.trading_metrics_data['losing_trades']
        self.backtest_performance.loc[datetime_input, "total_trades"] = self.trading_metrics_data['total_trades']
        self.backtest_performance.loc[datetime_input, "total_pnl"] = np.round((trailing_position/self.balances['initial_balance'][self.fiat]-1)*100,3)

        if has_position:
            self.backtest_positions = self.backtest_positions[~self.backtest_positions.index.duplicated(keep='last') | (self.backtest_positions.index.duplicated(keep="last") & ~self.backtest_positions.duplicated(subset=['close_order_id'], keep=False) & self.backtest_positions['close_order_id'].notna())]
            position_df = pd.DataFrame([position], index=[datetime_input])
            if pd.Timestamp(position_df.time_opened.values[0]) == datetime_input and position_df["close_order_id"].values[0] not in self.backtest_positions["close_order_id"].values:
                self.backtest_positions = pd.concat([self.backtest_positions, position_df], axis=0) 
            else:
                self.backtest_positions = self.backtest_positions.combine_first(position_df)
        else:
            if self.position_close == datetime_input:
                if position is not None:
                    position_df = pd.DataFrame([position], index=[datetime_input])
                    self.backtest_positions = self.backtest_positions[~self.backtest_positions.index.duplicated(keep='last') | (self.backtest_positions.index.duplicated(keep="last") & ~self.backtest_positions.duplicated(subset=['close_order_id'], keep=False) & self.backtest_positions['close_order_id'].notna())]
                    if pd.Timestamp(position_df.time_opened.values[0]) == datetime_input and position_df["close_order_id"].values[0] not in self.backtest_positions["close_order_id"].values:
                        self.backtest_positions = pd.concat([self.backtest_positions, position_df], axis=0) 
                    else:
                        self.backtest_positions = self.backtest_positions.combine_first(position_df)
            else:
                self.backtest_positions = self.backtest_positions.combine_first(pd.DataFrame([None], index=[datetime_input]))
        
        if actual_datetime.hour == 0 and actual_datetime.minute == 0:
            if "timestamp_opened" in self.backtest_positions.columns:
                self.backtest_positions["condition_index"] = self.backtest_positions["timestamp_opened"].apply(self.get_conditions_for_hour)
            
            self.save_backtest_performance_file(data="performance", file=self.backtest_performance, filename=self.backtest_filename)
            self.save_backtest_performance_file(data="positions", file=self.backtest_positions, filename=self.backtest_filename)
            self.strategy_data.to_csv(os.path.join(Trading_path, f"strategy_data_{self.start_datetime.strftime('%Y-%m-%d_%H_%M')}_{self.end_datetime.strftime('%Y-%m-%d_%H_%M')}_{self.backtest_filename}.csv"))

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
#                                                            backtest config functions
#
#################################################################################################################################################################


    def configure_backtest_performance_file(self, data=None, datetime_input=None, balances=True, positions=True, trades=False, filename=None):
        backtest_performance = None
        backtest_positions = None
        backtest_trades = None
        
        performance_path = os.path.join(Trading_path, f"backtest_performance_{self.start_datetime.strftime('%Y-%m-%d_%H_%M')}_{self.end_datetime.strftime('%Y-%m-%d_%H_%M')}_{filename}.csv")
        # performance_path = os.path.join(Trading_path, "backtest_performance_2024-07-31_00_00_2024-11-01_00_00.csv")
        positions_path = os.path.join(Trading_path, f"backtest_positions_{self.start_datetime.strftime('%Y-%m-%d_%H_%M')}_{self.end_datetime.strftime('%Y-%m-%d_%H_%M')}_{filename}.csv")
        # positions_path = os.path.join(Trading_path, "backtest_positions_2024-07-31_00_00_2024-11-01_00_00.csv")
        trades_path = os.path.join(Trading_path, f"backtest_trades_{self.start_datetime.strftime('%Y-%m-%d_%H_%M')}_{self.end_datetime.strftime('%Y-%m-%d_%H_%M')}_{filename}.csv")

        if self.delete_trade_histo: 
            if balances:
                backtest_performance = pd.DataFrame([self.ledger_data["balances"]["trailing_balances"]], index=[datetime_input])
                backtest_performance.loc[datetime_input, "price"] = self.data[f"price_{self.fiat.lower()}_{self.ticker_to_slug_mapping[self.coin].lower()}"][datetime_input]
            if positions:
                backtest_positions = pd.DataFrame(self.ledger_data["positions"], index=[datetime_input])
            if trades:
                backtest_trades = self.evaluate_trades()
            
            if backtest_performance is not None:
                backtest_performance.to_csv(performance_path)
            if backtest_positions is not None:
                backtest_positions.to_csv(positions_path)
            if backtest_trades is not None:
                backtest_trades.to_csv(trades_path)
        elif os.path.exists(performance_path) and data=="balances":
            backtest_performance = pd.read_csv(performance_path, index_col=0)
        elif os.path.exists(positions_path) and data=="positions":
            backtest_positions = pd.read_csv(positions_path, index_col=0)
        elif os.path.exists(trades_path) and data=="trades":
            backtest_trades = pd.read_csv(trades_path, index_col=0)
        
            
        if data == "balances":
            return backtest_performance
        elif data == "positions":
            return backtest_positions
        elif data == "trades":
            return backtest_trades
        elif data == "balances_positions":
            return backtest_performance, backtest_positions
        elif data == "balances_trades":
            return backtest_performance, backtest_trades
        elif data == "positions_trades":
            return backtest_positions, backtest_trades
        else:
            return backtest_performance, backtest_positions, backtest_trades
        
    def save_backtest_performance_file(self,data=None, file=None, filename=None):
        if data == "performance":
            file.to_csv(os.path.join(Trading_path, f"backtest_performance_{self.start_datetime.strftime('%Y-%m-%d_%H_%M')}_{self.end_datetime.strftime('%Y-%m-%d_%H_%M')}_{filename}.csv"))
        elif data == "positions":
            file.to_csv(os.path.join(Trading_path, f"backtest_positions_{self.start_datetime.strftime('%Y-%m-%d_%H_%M')}_{self.end_datetime.strftime('%Y-%m-%d_%H_%M')}_{filename}.csv"))
        elif data == "trades":
             file.to_csv(os.path.join(Trading_path, f"backtest_trades_{self.start_datetime.strftime('%Y-%m-%d_%H_%M')}_{self.end_datetime.strftime('%Y-%m-%d_%H_%M')}_{filename}.csv"))
        else:
            file.to_csv(os.path.join(Trading_path, f"backtest_performance_{self.start_datetime.strftime('%Y-%m-%d_%H_%M')}_{self.end_datetime.strftime('%Y-%m-%d_%H_%M')}_{filename}.csv"))
            file.to_csv(os.path.join(Trading_path, f"backtest_positions_{self.start_datetime.strftime('%Y-%m-%d_%H_%M')}_{self.end_datetime.strftime('%Y-%m-%d_%H_%M')}_{filename}.csv"))
            file.to_csv(os.path.join(Trading_path, f"backtest_trades_{self.start_datetime.strftime('%Y-%m-%d_%H_%M')}_{self.end_datetime.strftime('%Y-%m-%d_%H_%M')}_{filename}.csv"))

    
    def configure_logger(self):
        
        logger_path = utils.find_logging_path()

        current_date = dt.datetime.now().strftime("%Y%m%d")
        log_directory = "Algo Trader Backtest"
        log_file_name = f"Algo_trader_backtest_log_{self.start_datetime.strftime('%y_%m')}_to_{self.end_datetime.strftime('%y_%m')}_{current_date}.txt"
        log_file_path = os.path.join(logger_path, log_directory, log_file_name)

        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        elif not os.path.exists(os.path.join(logger_path, log_directory)):
            os.makedirs(os.path.join(logger_path, log_directory))
        
        log_format = "{time:YYYY-MM-DD HH:mm} | {level:<8} | {function:<50} - {message}"

        self.logger.add(log_file_path, rotation="500 MB", level="INFO", format=log_format)



#################################################################################################################################################################
#                                                               backtest run
#################################################################################################################################################################

    def run_backtest(self):
        #[placeholder for websocket]
       
        check_interval = "1m"
       
        if check_interval == "1m":
            minute_dividor = 1440
        elif check_interval == "5m":
            minute_dividor = 480
        elif check_interval == "15m":
            minute_dividor = 320
         
        backtest_period = utils.generate_time_series(self.start_datetime, self.end_datetime, "1m")
        
        for timestamp in backtest_period[minute_dividor:]:
            if timestamp == dt.datetime(2023,1,1,10,0,0):
                input("Press Enter to continue...")
            self.status_scanner(datetime_input=timestamp)
            self.trade(datetime_input=timestamp)
            
# from DataLoader import DataLoader
# from TradeBacktester import Backtester


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    AlgoTraderBacktest = AlgoTrader_backtest()
    
    # start_date_str = "2019-01-01 00:00"
    # end_date_str = "2023-10-27 00:00"
    # timeseries = utils.generate_time_series(start_date_str, end_date_str, freq="H")
    AlgoTraderBacktest.run_backtest()
    print("test completed")
    
