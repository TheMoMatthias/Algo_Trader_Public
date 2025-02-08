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
# from data_download_entire_history  import *

base_path = os.path.dirname(os.path.realpath(__file__))
crypto_bot_path = os.path.dirname(base_path)
Python_path = os.path.dirname(crypto_bot_path)
Trading_bot_path = os.path.dirname(Python_path)
Trading_path = os.path.join(Trading_bot_path,"Trading")
data_path_crypto = os.path.join(Trading_bot_path, "Data", "Cryptocurrencies")
datasets_path = os.path.join(data_path_crypto, "Datasets")
csv_dataset_path = os.path.join(datasets_path, "crypto datasets", "csv")
hdf_dataset_path = os.path.join(datasets_path, "crypto datasets", "hdf5")
hist_data_download_path = os.path.join(crypto_bot_path, "Hist Data Download")
san_api_data_path = os.path.join(hist_data_download_path, "SanApi Data")
main_data_files_path = os.path.join(san_api_data_path, "Main data files")

data_loader = os.path.join(crypto_bot_path, "Data Loader")
trade_api_path =  os.path.join(crypto_bot_path,"API Trader")
backtest_path = os.path.join(crypto_bot_path, "Backtesting")
config_path = os.path.join(crypto_bot_path,"Config")
utils_path = os.path.join(Python_path, "Tools")
logging_path = os.path.join(Trading_bot_path, "Logging")
kucoin_api = os.path.join(crypto_bot_path,"Kucoin API")

sys.path.append(crypto_bot_path)
sys.path.append(trade_api_path)
sys.path.append(backtest_path)
sys.path.append(utils_path)
sys.path.append(Trading_path)
sys.path.append(config_path)
sys.path.append(logging_path)
sys.path.append(data_path_crypto)
sys.path.append(kucoin_api)
sys.path.append(data_path_crypto)
sys.path.append(datasets_path)
sys.path.append(csv_dataset_path)
sys.path.append(hdf_dataset_path)
sys.path.append(data_loader)
sys.path.append(main_data_files_path)

import mo_utils as utils

from main_api_trader import MainApiTrader as api_trader
from DataLoader import DataLoader


class AlgoTrader(MainApiTrader, DataLoader):

    def __init__(self):
        #logger
        self.logger = logger
        self.configure_logger()
        
        #config
        config_path = utils.find_config_path() 
        config = utils.read_config_file(os.path.join(config_path,"AlgoTrader_config.ini"))
        
        #trading pair
        self.coin = utils.get_config_value(config, "general", "coin")
        self.fiat = utils.get_config_value(config, "general", "currency")
        self.symbol = f"{self.coin}-{self.fiat}"
        self.slippage =  utils.get_config_value(config, "general", "slippage")
        self.leverage_factor = utils.get_config_value(config, "trading", "leverage_factor")
        exchange_name = utils.get_config_value(config, "general", "exchange")

        # init kucoin trader   (also init self.ledger_data)
        main_api_trader.MainApiTrader.__init__(self, coin=self.coin, currency=self.fiat, slippage=self.slippage_factor, leverage=self.leverage_factor,logger_input=self.logger)
        
        filename = utils.get_config_value(config, "general", "dataset_filename")
        DataLoader.DataLoader.__init__(self, logger_input=self.logger, filename=filename)
        
        csv_name = os.path.splitext(filename)[0] + "_processed" + ".csv"   
        hdf5_name = os.path.splitext(filename)[0] + "_processed" + ".h5"

        if not os.path.exists(os.path.join(csv_dataset_path, (csv_name))):
            self.data = self.main_loader(save_dataset=True)
        else:
            self.data = pd.read_csv(os.path.join(csv_dataset_path, (csv_name)))

        self.data.index = pd.to_datetime(self.data.index, format="%Y-%m-%d %H:%M:%S")
        
        #balance coin
        self.current_account_balance_coin = self.get_margin_account_details(curr=self.coin)["total"].values[0]
        self.current_available_account_balance_coin = self.get_margin_account_details(curr=self.coin)["available"].values[0]
        
        #balance curr
        self.current_account_balance_curr = self.get_margin_account_details(curr=self.coin)["total"].values[0]
        self.current_available_account_balance_curr = self.get_margin_account_details(curr=self.coin)["available"].values[0]
        
        #value coin in currency
        self.current_coin_value_in_fiat = self.calculate_balance_in_fiat("EUR", self.coin, getSum=True)
        self.current_coin_value_in_currency = self.calculate_balance_in_fiat(self.fiat, self.coin, getSum=True)
        
        #trade settings
        self.taker_trade_fee, self.taker_maker_fee = self.get_actual_trading_fees(self.symbol)
        self.interval =  utils.get_config_value(config, "general", "interval")
        self.max_open_positions = utils.get_config_value(config, "trading", "max_open_positions")
        
        
        #ratios:
        self.investment_per_trade = utils.get_config_value(config, "trading", "investment_per_trade")
        self.stop_loss_percentage = utils.get_config_value(config, "trading", "stop_loss_percentage")
        self.take_profit_percentage = utils.get_config_value(config, "trading", "take_profit_percentage")
        
        
        # Account variables
        self.positions = self.ledger_data["positions"]
        
        if len(self.ledger_data["positions"]) ==0:
            initial_balance_value = self.current_account_balance_curr
            self.ledger_data["balances"]["initial_balance"][self.fiat] = initial_balance_value
            self.ledger_data["balances"]["initial_balance"][self.coin] = {"fiat_value":(self.current_account_balance_coin *self.get_price(self.symbol)),"coin_amt":self.current_account_balance_coin}
            self.initial_balance_curr = initial_balance_value
            self.initial_balance_coin_in_curr = self.current_account_balance_coin*self.get_price(self.symbol)
            self.initial_balance_coin = self.current_account_balance_coin
            self.ledger_data["balances"]["total_balance"] = initial_balance_value
        else:
            self.initial_balance_curr = self.ledger_data["balances"]["initial_balance"][self.fiat]
            self.initial_balance_coin_in_curr =self.ledger_data["balances"]["initial_balance"][self.coin]["fiat_value"]
            self.initial_balance_coin = self.ledger_data["balances"]["initial_balance"][self.coin]["coin_amt"]
            self.ledger_data["balances"]["total_balance"] = initial_balance_value


        self.pnl_account_abs = (self.current_account_balance_curr + self.get_price(self.symbol)*self.current_account_balance_coin) - (self.initial_balance_curr + self.initial_balance_coin_in_curr)
        self.pnl_account_pct = ((self.current_account_balance_curr + self.get_price(self.symbol)*self.current_account_balance_coin) - (self.initial_balance_curr + self.initial_balance_coin_in_curr)) / (self.initial_balance_curr + self.initial_balance_coin_in_curr)
        self.total_profit_balance = max(((self.current_account_balance_curr + self.get_price(self.symbol)*self.current_account_balance_coin)- (self.initial_balance_curr + self.initial_balance_coin_in_curr)),0)
    
        # trading metrics
        self.trading_metrics_path = os.path.join(Trading_path,"trading_metrics_AlgoTrader.json") 
        self.trading_metrics_data = self.load_trading_metrics(self.trading_metrics_path)

        self.losing_trades = self.trading_metrics_data["losing_trades"]
        self.winning_trades = self.trading_metrics_data["winning_trades"]
        self.largest_loss =  self.trading_metrics_data["largest_loss"]
        self.largest_gain = self.trading_metrics_data["largest_gain"]

#################################################################################################################################################################
#
#
#                                                                  Algo trader functions
#
#################################################################################################################################################################


    def load_trading_metrics(self, metrics_file_path=None):
        if not os.path.exists(metrics_file_path):
            trading_metrics_data = {"losing_trades": 0 , "winning_trades": 0, "largest_loss":0, "largest_gain":0}
            
            with open(metrics_file_path, "w") as file:
                json.dump(trading_metrics_data, file, indent=4)

            return trading_metrics_data
        with open(metrics_file_path, "r") as file:
            trading_metrics_data = json.load(file)
            return trading_metrics_data
        
        
    def trade(self, datetime_input=None):
        
        #check of stop loss or take profit order triggered
        
        self.status_scanner()

        #get signal
                       # <- here the ML part / Strategy will be imported
        signal =  self.evaluate_signal()

        if signal == "neutral":
            
            self.logger.info("Received neutral signal. Waiting for next update")
            return

        else:
            #check for open positions or orders
            has_open_order = None
            has_open_position = None

            if self.ledger_data["current_trades"][self.symbol] is not None:
                has_open_position = True
                current_order_id = self.ledger_data["current_trades"][self.symbol]["order_id"]
                current_position = self.get_position_by_order_id(current_order_id)
                current_stop_loss_order_id = self.ledger_data["current_trades"][self.symbol]["stop_loss_order_id"]
                current_take_profit_order_id = self.ledger_data["current_trades"][self.symbol]["take_profit_order_id"]

                check_if_order_still_active = self.get_order_details_by_id(active=True, orderId=current_order_id)
                check_if_order_done = self.get_order_details_by_id(orderId=current_order_id)
                
                if check_if_order_still_active["isActive"]:
                    has_open_order = True
                elif not check_if_order_done["isActive"]:
                    has_open_order = False
            
            price = self.get_price(self.symbol)

            if signal == "buy":
                stop_loss_price = (1-self.stop_loss_percentage) * price
                take_profit_price = (1+self.take_profit_percentage) * price
            elif signal == "sell":
                stop_loss_price = (1+self.stop_loss_percentage) * price
                take_profit_price = (1-self.take_profit_percentage) * price


            if signal == "buy" and has_open_position ==False and has_open_order == False:

                unlevered_amount = self.current_account_balance_curr *self.investment_per_trade
                amount = unlevered_amount * (self.leverage_factor-1)

                self.enter_margin_trade(coin=self.coin, fiat=self.fiat,
                                        amount=amount, balance_fiat = self.current_available_account_balance_curr, is_long=True, order_type="market",
                                        stop_price=stop_loss_price, take_profit_price=take_profit_price)
            
            elif signal == "buy" and has_open_position==True and current_position["side"]=="sell":   #only considers changing sides when previous trade was sell, currently two consecutive buy orders are not permitted

                self.close_margin_position(coin=self.coin, fiat=self.fiat, order_id=current_order_id, current_position=current_position)

                unlevered_amount = self.current_account_balance_curr *self.investment_per_trade
                amount = unlevered_amount * (self.leverage_factor-1)

                self.enter_margin_trade(coin=self.coin, fiat=self.fiat,
                                        amount=amount, balance_fiat = self.current_available_account_balance_curr, is_long=True, order_type="market",
                                        stop_price=stop_loss_price, take_profit_price=take_profit_price)

            elif signal == "buy" and has_open_position==True and current_position["side"]=="buy":
                self.logger.info("Received buy signal although already long. Waiting for next update")
                pass

            elif signal == "sell" and has_open_position ==False and has_open_order == False:
                
                unlevered_amount = (self.current_account_balance_curr *self.investment_per_trade)  / price
                amount = unlevered_amount * (self.leverage_factor-1) #max(self.leverage_factor,2)
                

                self.enter_margin_trade(coin=self.coin, fiat=self.fiat,
                                        amount=amount, balance_fiat = self.current_available_account_balance_curr, is_long=False, order_type="market",
                                        stop_price=stop_loss_price, take_profit_price=take_profit_price)

            elif signal == "sell" and has_open_position==True and current_position["side"]=="buy":
                
                self.close_margin_position(coin=self.coin, fiat=self.fiat, order_id=current_order_id, current_position=current_position)
                
                unlevered_amount = (self.current_account_balance_curr *self.investment_per_trade)  / price
                amount = unlevered_amount * (self.leverage_factor-1)

                self.enter_margin_trade(coin=self.coin, fiat=self.fiat,
                                        amount=amount, balance_fiat = self.current_available_account_balance_curr, is_long=False, order_type="market",
                                        stop_price=stop_loss_price, take_profit_price=take_profit_price)

            elif signal == "sell" and has_open_position==True and current_position["side"]=="sell":
                
                self.logger.info("Received sell signal although already short. Waiting for next update")
                pass

    
    # @staticmethod
    def status_scanner(self):
        if self.ledger_data["current_trades"][self.symbol] is not None:
            current_order_id = self.ledger_data["current_trades"][self.symbol]["order_id"]
            position = self.get_position_by_order_id(current_order_id)

        risk_prevention_status = self.check_sl_or_tp_triggered(coin=self.coin, fiat=self.fiat, position=position)
        
        #balance coin
        self.current_account_balance_coin = self.get_margin_account_details(curr=self.coin)["total"].values[0]
        self.current_available_account_balance_coin = self.get_margin_account_details(curr=self.coin)["available"].values[0]
        
        #balance curr
        self.current_account_balance_curr = self.get_margin_account_details(curr=self.coin)["total"].values[0]
        self.current_available_account_balance_curr = self.get_margin_account_details(curr=self.coin)["available"].values[0]

        self.ledger_data["balances"]["trailing_balances"][self.fiat] = self.current_available_account_balance_curr
        self.ledger_data["balances"]["trailing_balances"][self.coin] = self.current_available_account_balance_coin
        self.ledger_data["balances"]["trailing_balances"]["total_balance"] = self.calculate_balance_in_fiat(fiat=self.fiat, coin=self.coin, getSum=True, get_only_trade_pair_bal=True, account_type="margin")
        

    def configure_logger(self):
        
        logger_path = utils.find_logging_path()

        #logger
        current_datetime = dt.datetime.now()
        timestamp = current_datetime.strftime('%d-%m-%Y_%H-%M')
        log_directory = "Algo Trader"
        log_file_name = f"Algo_trader_log_{timestamp}.txt"
        log_file_path = os.path.join(logger_path, log_directory, log_file_name)

        if not os.path.exists(os.path.join(logger_path, log_directory)):
            os.makedirs(os.path.join(logger_path, log_directory))

        self.logger.add(log_file_path, rotation="500 MB", level="INFO")

    def test(self):
        print("test")  

a = AlgoTrader()

from main_api_trader import MainApiTrader
from TradeBacktester import Backtester

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    AlgoTraderBacktest = AlgoTrader(MainApiTrader, DataLoader)
    print("test completed")

# test = a.test(KucoinTrader)
trade_test = a.trade()
print("test completed")