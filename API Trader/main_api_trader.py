import os
import json
import pandas as pd
import numpy as np
import kucoin
import sys
import ntplib
from time import ctime
import requests
import hashlib
import base64
import hmac
import time
import datetime as dt
from datetime import datetime, timedelta
import urllib.parse
from loguru import logger
import ccxt
import math
from io import StringIO, BytesIO
from functools import wraps

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

base_path = os.path.dirname(os.path.realpath(__file__))
crypto_bot_path = os.path.dirname(base_path)
Python_path = os.path.dirname(crypto_bot_path)
Trading_bot_path = os.path.dirname(Python_path)
Trading_path = os.path.join(Trading_bot_path, "Trading")
logging_path = os.path.join(Trading_bot_path, "Logging")
# crypto_bot_path = r"C:\Users\mauri\Documents\Trading Bot\Python\Crypto Bot"

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
        # print(f"Added: {path}")


import mo_utils as utils
from utilsAWS import S3Utility
from utilsGoogleDrive import GoogleDriveUtility


#################################################################################################################################################################
#
#
#                                                                  KUCOIN TRADER CLASS
#
#################################################################################################################################################################


class MainApiTrader:
    def __init__(self, exchange, fiat, slippage, leverage, logger_input=None):
        
        if logger_input is None:
            self.logger = logger
            self.configure_logger()
        else:
            self.logger = logger_input

        config_path = utils.find_config_path()
        self.api_config = utils.read_config_file(os.path.join(config_path,"kucoin_config.ini"))
    
        self.api_key = utils.get_config_value(self.api_config, "overview", "eins")
        self.api_secret = utils.get_config_value(self.api_config, "overview", "zwei")
        self.api_passphrase = utils.get_config_value(self.api_config, "overview", "pass")
        self.client_user_id = utils.get_config_value(self.api_config, "overview", "client_user_id")

        self.exchange = getattr(ccxt, exchange)({ "apiKey": self.api_key, "secret": self.api_secret, "password": self.api_passphrase})
        self.exchange.options["createMarketBuyOrderRequiresPrice"] = "false"

        self.fiat = fiat
        # self.coin = coin
        # self.symbol = f"{coin}-{fiat}"
        # self.ccxt_symbol = f"{coin}/{fiat}"

        self.slippage = slippage 
        self.leverage_factor = leverage

        # Ledger data 
        trading_path = utils.find_trading_path()
        
        self.ledger_path = f"{trading_path}/trading_ledger.json"
        self.ledger_data = self.load_ledger_data(self.ledger_path)

        # trading metrics
        
        self.trading_metrics_path = f"{Trading_path}/trading_metrics_AlgoTrader.json" 
        self.trading_metrics_data = self.load_trading_metrics(self.trading_metrics_path)

        self.losing_trades = self.trading_metrics_data["losing_trades"]
        self.winning_trades = self.trading_metrics_data["winning_trades"]
        self.largest_loss =  self.trading_metrics_data["largest_loss"]
        self.largest_gain = self.trading_metrics_data["largest_gain"] 

        # self.symbol_info = self.get_symbol_list(symbol=self.symbol)
        # self.minSize = float(self.symbol_info["baseMinSize"].values[0])
        # self.minFunds = float(self.symbol_info["quoteMinSize"].values[0])
        # self.coinIncrement = self.get_rounding_precision(self.symbol_info["baseIncrement"].values[0])
        # self.fiatIncrement= self.get_rounding_precision(self.symbol_info["quoteIncrement"].values[0])
        
        #retrieve all assets and create mapping with san coin names
        # all_asset_file_path = f"{main_data_files_path}/all_assets.xlsx" 
        self.all_assets = pd.read_excel(os.path.join(main_data_files_path, "all_assets.xlsx"), header=0)
        self.ticker_to_slug_mapping = dict(zip(self.all_assets['ticker'], self.all_assets['slug']))
        
        self.logger.info("MainApiTrader initialized")
        time.sleep(0.5)
        
#################################################################################################################################################################
#
#
#                                                                  TRADING FUNCTIONS
#
#################################################################################################################################################################
    
    def enter_margin_trade(self, coin=None, fiat=None, size=None, funds=None, balance_fiat = None, is_long=None, order_type=None, limit_price=None, stop_price=None, take_profit_price=None):
        try:
            #cancel all existing stop loss and take profit orders
            
            #symbol
            coin = str(coin)
            fiat = str(fiat)
            symbols  = self.get_symbols(coin=coin, fiat=fiat)
            symbol, symbol_ccxt = symbols["symbol"], symbols["ccxt_symbol"]
            size = float(size) if size is not None else None
            funds = float(funds) if funds is not None else None
            increments = self.get_increments(symbol)
            mins = self.get_min_sizes(symbol)
            
            is_long = bool(is_long)
            order_type = str(order_type)
            limit_price = float(limit_price) if limit_price is not None else None
            stop_price = float(stop_price) if stop_price is not None else None
            take_profit_price = float(take_profit_price) if take_profit_price is not None else None
            
            self.cancel_all_orders(symbols=symbols, is_stop=True)    #adjust and delete if you wanna allow for multiple positions
            
            # Create the initial margin order
            if is_long is not None:
                side = 'buy' if is_long else 'sell'
            else:
                self.logger.info("No side specificed aborting trade")
                return
            
            #function for HF:   initial_balance = self.get_margin_account_details(curr=fiat)["available"].values[0]
            initial_balance = self.get_account_list(curr=fiat, account_type="margin")["available"].values[0]
            price = self.get_price(symbol=symbol_ccxt)
            
            if side == "buy":
                self.modify_leverage_multiplier(leverage_factor=self.leverage_factor)
                available_balance = initial_balance * self.leverage_factor
    
                if funds > available_balance:
                    funds = funds + (available_balance - funds)

                funds = self.round_down(funds, increments["fiat"])
                
                if funds < mins["fiat"]:
                    self.logger.error("funds is less than minimum funds required for trade")
                    return
            else:
                self.modify_leverage_multiplier(leverage_factor=self.leverage_factor+1)
                available_balance = (initial_balance*self.leverage_factor) / price

                if size > available_balance:
                    size = size + (available_balance - size)
                    size = size *(1-self.get_trading_fees(symbols, "0")[0])
                
                size = self.round_down(size, increments["coin"])
                
                if size < mins["coin"]:
                    self.logger.error("size is less than minimum size required for trade")
                    return
                
            if self.leverage_factor<=1 and side == "buy":
               initial_order_id, borrowSize = self.execute_auto_borrow_margin_order(symbols=symbols, funds=funds, size=size, side=side, order_type=order_type,use_leverage=False, increments=increments)
            else:
                initial_order_id, borrowSize = self.execute_auto_borrow_margin_order(symbols=symbols, funds=funds, size=size, side=side, order_type=order_type, increments=increments) #removed price as no limit order strategy

            # Calculate the invested amount in USDT
            # if funds is None:
            #     current_price = self.get_price(symbol)
            #     invested_amount = size * ([limit_price if not np.isnull(limit_price) else current_price])
            # else:
            #     invested_amount = funds
    
            order_details = self.get_order_details_by_id(symbols, orderId=initial_order_id)
            
            # coin_balance = self.get_margin_account_details(curr=coin)["available"].values[0]
            
            if side == "buy":                
                execution_price = self.calc_price_from_details(dealFunds=order_details["dealFunds"][0],dealSize=order_details["dealSize"][0])
            else:
                execution_price = self.calc_price_from_details(dealFunds=order_details["dealFunds"][0],dealSize=order_details["dealSize"][0])
            
            fees = order_details["fee"][0]

            size = order_details["dealSize"][0] if side=="buy" else size
            funds = order_details["dealFunds"][0]
            
            sl_size =  order_details["dealSize"][0]
            tp_size = order_details["dealSize"][0]
            
            if side == "buy":    
                if size == None:
                    raise ("size is none please check and adjust invested amount calculation")
                    
            else:
                fee_rate = self.get_actual_trading_fees(symbol)[0]
                sl_funds = (funds+fees) * (stop_price/execution_price)    #*(1+fee_rate)
                tp_funds = (funds+fees) * (take_profit_price/execution_price)    #*(1+fee_rate)

                if funds == None:
                    raise ("funds is none please check and adjust invested amount calculation")
                                                                                                           
            # Create a stop loss order if stop_price is specified
            stop_loss_order = None
            take_profit_order = None
            
            if stop_price:
                stop_loss_side = 'sell' if is_long else 'buy'
                if side == "buy":
                    stop_loss_order_id = self.execute_stop_order(symbols=symbols, size=sl_size, order_type="market", side=stop_loss_side, stopPrice=stop_price, stop="loss", auto_repay=True, increments=increments)
                else:
                    stop_loss_order_id = self.execute_stop_order(symbols=symbols, size=size, order_type="limit", side=stop_loss_side, stopPrice=stop_price, price=(stop_price* (1 + self.slippage)),stop="entry",auto_repay=True, increments=increments)
                    
                stop_loss_order = True

            # Create a take-profit order if specified
            if take_profit_price:
                take_profit_side = 'sell' if is_long else 'buy'
                if side =="buy":
                    take_profit_order_id = self.execute_stop_order(symbols=symbols, size=tp_size, order_type="market", side=take_profit_side, stopPrice=take_profit_price, stop="entry",auto_repay=True, increments=increments)
                else:
                    take_profit_order_id = self.execute_stop_order(symbols=symbols, size=size, order_type="limit", side=take_profit_side, stopPrice=take_profit_price, price=(take_profit_price* (1 + self.slippage)), stop="loss",auto_repay=True, increments=increments)
                take_profit_order = True


            #create fallback mechanism in case the price should fall below sl threshold
            fallback_side = 'sell' if is_long else 'buy'

            if side == "buy":
                fallback_price = stop_price * 0.8
                fallback_order_id = self.execute_stop_order(symbols=symbols, size=sl_size, order_type="market", side=fallback_side, stopPrice=fallback_price, stop="loss", auto_repay=True, increments=increments)
            else:
                fallback_price = stop_price * 1.2
                fallback_order_id = self.execute_stop_order(symbols=symbols, size=size, order_type="limit", side=fallback_side, stopPrice=fallback_price, price=(fallback_price* (1 + self.slippage)),stop="entry",auto_repay=True,increments=increments)
                
            

            #timestamp_now = datetime.now().timestamp()*1000
            # Add order information to the ledger
            position_info = {
                'order_id':                     initial_order_id,
                'size':                         size,
                'funds':                        funds,
                'side':                         side,
                'pair':                         symbol,
                'execution_price':              execution_price,
                'status':                       'open',
                'timestamp_opened':             str(order_details["createdAt"][0]),
                'timestamp_closed':             None,
                'time_opened':                  str(order_details["createdAt"][0]),
                'time_closed':                  None,  
                'invested_usdt':                (order_details["dealFunds"][0]+fees),
                'initial_balance' :             initial_balance,
                'closing_amount_invested':      None,
                'closing_balance':              None,
                'stop_loss_order_id':           stop_loss_order_id if stop_loss_order else None,
                'take_profit_order_id':         take_profit_order_id if take_profit_order else None,
                'rebuy_order_id':               None,
                'convert_order_id':             None,
                "borrowSize" :                  borrowSize,
                "repaid" :                      "open_borrowing",
                "close_order_id" :              None, 
                "fees_opening_trade":           fees,
                "fees_closing_trade":           None, 
                "pnl":                          None,
                "owed_tax":                     None             
            }

            self.ledger_data["current_trades"][symbol] = {"order_id":initial_order_id, "stop_loss_order_id":stop_loss_order_id, "take_profit_order_id":take_profit_order_id}
            balance_fiat = self.get_account_list(curr=fiat, account_type="margin")["available"].values[0]       #self.get_margin_account_details(curr=fiat)["available"].values[0]
            balance_coin = self.get_account_list(curr=coin, account_type="margin")["available"].values[0]
            
            self.ledger_data["balances"]["trailing_balances"][fiat] = balance_fiat
            self.ledger_data["balances"]["trailing_balances"][coin] = balance_coin
            self.ledger_data["balances"]["trailing_balances"]["total_balance"] = self.calculate_balance_in_fiat(fiat=fiat, coin=coin, getSum=True, get_only_trade_pair_bal=True, account_type="margin")
            
            self.ledger_data["order_details"][initial_order_id] = {}
            self.ledger_data["order_details"][initial_order_id]["initial_order_details"] = order_details

            self.update_position(position_info)
            
            
        except Exception as e:
            self.logger.error(f"An error occurred entering the trade: {e}")
            raise Exception(f"An error occurred entering the trade: {e}")
    
    def close_margin_position(self,coin=None,fiat="USDT",order_id=None, current_position=None):
        """
        closes margin orders by order id
        """
        coin = str(coin)
        fiat = str(fiat)
        symbols  = self.get_symbols(coin=coin, fiat=fiat)
        symbol, symbol_ccxt = symbols["symbol"], symbols["ccxt_symbol"]
        order_id = str(order_id)
        increments = self.get_increments(symbol)
        mins = self.get_min_sizes(symbol)
        
        try:
            position = current_position #self.get_position_by_order_id(order_id)
            
            if not position:
                raise Exception("Position not found")

            symbol = position['pair']
            size = position['size']
            funds = position['funds']
            is_long = position['side'] == 'buy'
            
            # Determine the side of the order to close the position
            side = 'sell' if is_long else 'buy'
            
            closing_funds = None
            closing_size = self.round_down(size, increments["coin"])   #using two times size bc we have market sell (size) and limit buy (size + price)

            if side == "sell":
                borrowed_asset = fiat
                borrowed_amount = funds
            else:
                borrowed_asset = coin
                borrowed_amount = size   #might be used instead of BorrowAmount returned from initial execute auto borrow order

            risk_prevention_status = self.check_sl_or_tp_triggered(coin=coin, fiat=fiat, position=position)

            if risk_prevention_status =="not triggered":
                # Place an order to close the position, the take profit order and the stop loss order
                if side == "sell":
                    close_order_id, borrowSize = self.execute_auto_repay_margin_order(symbols=symbols, size=closing_size, funds=closing_funds, order_type="market", side=side, increments=increments)
                else:
                    limit_price = self.get_price(symbol=symbol) * (1 + self.slippage)
                    debt = self.get_margin_account_details(curr=borrowed_asset)["liability"].values[0]
                    closing_size = max(closing_size, debt)
                    close_order_id, borrowSize = self.execute_auto_repay_margin_order(symbols=symbols, size=closing_size, funds=closing_funds, order_type="limit", side=side, price=limit_price,increments=increments)
                self.logger.info(f"{'#'*30}   Closing order executed with order id: {close_order_id}    {'#'*30}")
                
                self.cancel_all_orders(symbols=symbols)

                #get close order details
                close_order_details = self.get_order_details_by_id(symbols=symbols, orderId=close_order_id)
                
                
                if side == "sell":
                    closing_amount_invested = close_order_details["dealFunds"][0]
                    pnl_trade = closing_amount_invested - position['invested_usdt']
                elif side =="buy":
                    coin_balance_after_closing = self.get_margin_account_details(curr=coin)["available"].values[0]
                    closing_amount_invested = close_order_details["dealSize"][0]
                    pnl_trade = closing_amount_invested - closing_size

                    if pnl_trade < 0:
                        
                        if pnl_trade <mins["coin"]:
                            pnl_trade += (1.5*mins["coin"])-pnl_trade
                        pnl_trade = self.round_down(pnl_trade, increments["coin"])
                        
                        rebuy_price = self.get_price(symbol=symbol) * (1 + self.slippage)
                        rebuy_order_id= self.execute_margin_order(symbols=symbols, size=abs(pnl_trade), order_type="limit", side=side,price=rebuy_price,increments=increments)
                        position["rebuy_order_id"] = rebuy_order_id
                        rebuy_order_details = self.get_order_details_by_id(symbols=symbols, orderId=rebuy_order_id)
                    else:
                        rebuy_order_details = {"fee":0} 
                        rebuy_order_details["dealFunds"] = 0
                        

                #only applicable if HF trading
                #double check if all borrowed funds have been repaid
                borrowed_asset_details = self.get_margin_account_details(curr=borrowed_asset)
                liabilities = borrowed_asset_details["liability"].values[0]
                available = borrowed_asset_details["available"].values[0]
                
                if liabilities >  0:
                    size_repay = liabilities
                    
                    if liabilities > available and side =="buy":
                        owed_amount = liabilities - available
                        
                        if owed_amount <mins["coin"]:
                            owed_amount += (1.5*mins["coin"])-pnl_trade
                        
                        owed_amount = self.round_down(owed_amount, increments["coin"])
                        
                        rebuy_price_debt = self.get_price(symbol=symbol) * (1 + self.slippage)
                        rebuy_order_debt_id = self.execute_margin_order(symbols=symbols, size=owed_amount, order_type="limit", side=side,price=rebuy_price_debt, increments=increments)

                    repaid_order_id, repaid_amount = self.repay_funds(symbols=symbols, curr=borrowed_asset, size=size_repay)
                    self.logger.info(f"Repaid order of amount {repaid_amount}")
                    
                #when short convert coin into usdt
                if side == "buy":
                    coin_balance_after_trade = self.get_account_list(curr=coin, account_type="margin")["available"].values[0]
                    coin_balance_after_trade = self.round_down(coin_balance_after_trade, increments["coin"])
                    
                    if coin_balance_after_trade > 0:
                        #potentially also implement the min size adjustment here, but previous checks should prevent this
                        convert_order_id, convert_borrow_size = self.execute_margin_order(symbols=symbols, size=coin_balance_after_trade, order_type="market", side="sell", increments=increments)#
                        position["convert_order_id"] = convert_order_id
                        convert_order_id_details = self.get_order_details_by_id(symbols=symbols, orderId=convert_order_id)
                    else:
                        position["convert_order_id"] = None
                        convert_order_id_details = {"fee":0}
                        convert_order_id_details["dealFunds"] = 0
                        # convert_order_id_details["funds"] = 0
                
                # Update the position as closed in the ledger
                position["close_order_id"] = close_order_id
                position['status'] = 'closed'
                position['timestamp_closed'] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                position['time_closed'] =      dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                position["repaid"] = "repaid"
                
                if side =="buy" and pnl_trade <0:
                    fees = { "sum":     sum([close_order_details["fee"][0], rebuy_order_details["fee"][0], convert_order_id_details["fee"][0]]), 
                            "single":   [close_order_details["fee"][0], rebuy_order_details["fee"][0], convert_order_id_details["fee"][0]]}
                elif side =="buy" and pnl_trade >=0:
                    fees = { "sum":     sum([close_order_details["fee"][0], convert_order_id_details["fee"][0]]), 
                            "single":   [close_order_details["fee"], convert_order_id_details["fee"]]}                
                else:
                    fees = close_order_details["fee"][0]
                
                position["fees_closing_trade"] = fees["sum"] if isinstance(fees, dict) else fees

                closing_balances = self.get_account_list(account_type="margin")
                closing_balance_fiat = closing_balances[closing_balances["currency"]==fiat]["available"].values[0] #self.get_margin_account_details(curr=fiat)["available"].values[0]
                closing_balance_coin = closing_balances[closing_balances["currency"]==coin]["available"].values[0]
                position["closing_balance"] = closing_balance_fiat
                
                if side == "sell":
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = pnl_trade   
                else:
                    closing_amount_invested = close_order_details["funds"][0] - convert_order_id_details["dealFunds"][0] + rebuy_order_details["funds"][0]
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = (position['funds'] - closing_amount_invested)

                if position["pnl"] > 0:
                    self.logger.info(f"{'#'*30} Position closed with profit of {position['pnl']} {fiat}  {'#'*30}")
                    position["owed_tax"] = position["pnl"] * 0.25
                    self.trading_metrics_data["winning_trades"] += 1

                    if position["pnl"] > self.trading_metrics_data["largest_gain"]:
                        self.trading_metrics_data["largest_gain"] = position["pnl"]

                elif position["pnl"] < 0:
                    self.logger.info(f"{'#'*30}   Position closed with loss of {position['pnl']} {fiat}    {'#'*30}")
                    position["owed_tax"] = position["pnl"] * 0.25
                    self.trading_metrics_data["losing_trades"] += 1

                    if position["pnl"] < self.trading_metrics_data["largest_loss"]:
                        self.trading_metrics_data["largest_loss"] = position["pnl"]

                self.save_trading_metrics(self.trading_metrics_path, self.trading_metrics_data)  

                #set open position to None
                self.ledger_data["current_trades"][symbol] = None

                #update balances
                self.ledger_data["balances"]["trailing_balances"][fiat] = closing_balance_fiat
                self.ledger_data["balances"]["trailing_balances"][coin] = closing_balance_coin
                self.ledger_data["balances"]["trailing_balances"]["total_balance"] = self.calculate_balance_in_fiat(fiat=fiat, coin=coin, getSum=True, get_only_trade_pair_bal=True, account_type="margin")

                self.ledger_data["order_details"][position["order_id"]]["sl_order_details"] = close_order_details
                
                if side == "buy":
                    self.ledger_data["order_details"][position["order_id"]]["rebuy_order_details"] = rebuy_order_details
                    self.ledger_data["order_details"][position["order_id"]]["usdt_convert_order_details"] = convert_order_id_details

                self.update_position(position)

                return 
            else:
                if risk_prevention_status == "tp_cancel":
                    closed_by = "stop loss"
                else:
                    closed_by = "take profit"
                logger.info(f"Position already closed by {closed_by}")
                
                liabilities = self.get_margin_account_details(curr=borrowed_asset)["liability"].values[0]
                
                if liabilities >  0:
                    size_repay = liabilities
                    
                    repaid_order_id, repaid_amount = self.repay_funds(symbols=symbols, curr=borrowed_asset, size=size_repay)
                    self.logger.info(f"Repaid order of amount {repaid_amount}")
                return
        
        except Exception as e:
            self.logger.error(f"An error occurred closing the position: {e}")
            raise Exception(f"An error occurred closing the position: {e}")
        

    def get_trading_fees(self, symbols, currency_type=0):
        """
        REQ:            currencyType:       string    0: crypto-currency, 1: fiat currency   default is 0
        link:           https://www.kucoin.com/docs/rest/funding/trade-fee/basic-user-fee-spot-margin-trade_hf

        Returns:
        takerFeeRate:   base taker fee
        makerFeeRate:   base maker fee   	
        """
        try:
            symbol = symbols["symbol_ccxt"]
            response = self.exchange.fetch_trading_fee(symbol)
            if response["info"]["code"]=="200000":
                takerFeeRate = float(response["taker"])  
                makerFeeRate = float(response["maker"])
                return takerFeeRate, makerFeeRate
            
        except Exception as e:
            self.logger.error(f"An error occurred when retrieving trading fees: {e}")
            return {"error": f"An exception occurred: {str(e)}"}
        

    def get_actual_trading_fees(self, symbol):
        """
        REQ:            symbols:       string e.g. BTC-USDT
        link:           https://www.kucoin.com/docs/rest/funding/trade-fee/trading-pair-actual-fee-spot-margin-trade_hf

        Returns:
        takerFeeRate:   Actual taker fee rate for the trading pair
        makerFeeRate:   Actual maker fee rate for the trading pair
        """
        try:
            endpoint = f'/api/v1/trade-fees'
            url = 'https://api.kucoin.com' + endpoint

            params ={"symbols" : symbol}

            headers = self.create_sign_in(api_key=self.api_key, api_secret=self.api_secret, api_passphrase=self.api_passphrase, 
                                        method="GET",url=url, endpoint=endpoint, data="", params=params)

            response = requests.get(url, headers=headers, params=params)    
            response_json = response.json()

            # Check if response is successful
            if response.status_code == 200:
                takerFeeRate = float(response_json["data"][0]["takerFeeRate"])  
                makerFeeRate = float(response_json["data"][0]["makerFeeRate"])
                return takerFeeRate, makerFeeRate
            else:
                return {"error": f"Failed to fetch data: {response.status_code}", "details": response_json}
        
        except Exception as e:
            self.logger.error(f"An error occurred when retrieving actual trading pair fees: {e}")
            return {"error": f"An exception occurred: {str(e)}"}

    


#################################################################################################################################################################
#
#
#                                                                       Margin Order Functions (V1)
#
#################################################################################################################################################################
    

    def execute_auto_borrow_margin_order(self, symbols=None, funds = None, size=None, side=None, price=None, order_type="market",use_leverage=True, increments=None):
        """
        ONLY FOR STANDARD MARGIN TRADING NOT HIGH-FREQUENCY MARGIN_V2 !!!!!!!!!

        REQ:    side:                 "buy" or "sell"
        REQ:    order_type:           "limit" or "market"

        Additional params for MARKET ORDERS where EITHER ONE IS REQUIRED TO BE SPECIFIED:
        REQ:    size:                   String  used when creating sell orders
        REQ:    funds:                  String  used when creating buy orders
        
        Additonial params for LIMIT ORDERS:
        REQ:    price:                  String  used when creating limit orders

        Return:
        orderNo:                orderid
        borrowSize:             borrow amount
        loanApplyId:            
        """
        try:
            size = self.round_down(abs(size),increments["coin"]) if size is not None else None
            funds = self.round_down(abs(funds),increments["fiat"]) if funds is not None else None
            price = self.round_down(price, 1) if price is not None else None
            
            differentiating_exchanges = ["KuCoin"]
            
            params = {"tradeType": "MARGIN_TRADE", 
                      "autoBorrow": True}       #MARGIN_TRADE or MARGIN_V2_TRADE
            
            if side == "buy" and not use_leverage:
                params["autoBorrow"] = False
            
            exchange_name = self.exchange.name

            if exchange_name in differentiating_exchanges:
                if exchange_name == "KuCoin":
                    
                    if size:
                        params["size"] =  size
                
                    if funds and order_type=="market": 
                        params["funds"] = int(funds)

                    if price and order_type == "limit":
                        params["price"] = price
                    
                    if order_type == "limit":
                        params["type"] = order_type 
                    
                    if side=="buy":
                        amount = 0
                    else:
                        amount = size
            else:
                if side=="buy":
                    amount = funds/self.get_price(symbol=symbols["symbol_ccxt"])
                else: 
                    amount = size
                params = params
                # amount = 0 
            
            if order_type == "limit" and price:
                order  = self.exchange.create_order(symbol=symbols["symbol_ccxt"], amount=size,  side=side, type=order_type, price=price, params=params)
            else:
                order = self.exchange.create_order(symbol=symbols["symbol_ccxt"], amount=amount,  side=side, type=order_type, params=params)
            
            order_id  = order["info"]["orderId"]
            borrowSize = float(order["info"]["borrowSize"]) if not order["info"]["borrowSize"] is None else 0

            return order_id, borrowSize
        except Exception as e:
            self.logger.error(f"An error occurred while creating auto borrow order: {e}")
            return 
        
    def execute_auto_repay_margin_order(self, symbols=None, funds = None, size=None, side=None, price=None, order_type="market", increments=None):
        """
        ONLY FOR STANDARD MARGIN TRADING NOT HIGH-FREQUENCY MARGIN_V2 !!!!!!!!!

        REQ:    side:                  "buy" or "sell"
        REQ:    order_type:           "limit" or "market"

        Additional params for MARKET ORDERS where EITHER ONE IS REQUIRED TO BE SPECIFIED:
        REQ:    size:                   String  used when creating sell orders
        REQ:    funds:                  String  used when creating buy orders
        
        Additonial params for LIMIT ORDERS:
        REQ:    price:                  String  used when creating limit orders

        LINK:   https://www.kucoin.com/docs/rest/margin-trading/orders/place-margin-order
        
        Return:
        orderNo:                orderid
        borrowSize:             borrow amount
        loanApplyId:            
        """
        try:
            size = self.round_down(abs(size),increments["coin"]) if size is not None else None
            funds = self.round_down(abs(funds),increments["fiat"]) if funds is not None else None
            price = self.round_down(price, 1) if price is not None else None
            
            differentiating_exchanges = ["KuCoin"]
            
            params = {"tradeType": "MARGIN_TRADE", 
                      "autoRepay": True}       #MARGIN_TRADE or MARGIN_V2_TRADE

            exchange_name = self.exchange.name

            if exchange_name in differentiating_exchanges:
                if exchange_name == "KuCoin":
                    
                    if size:
                        params["size"] =  size
                
                    if funds and order_type=="market": 
                        params["funds"] = funds

                    if price and order_type == "limit":
                        params["price"] = price
                    
                    if order_type == "limit":
                        params["type"] = order_type 
                    
                    if side=="buy":
                        amount = 0
                    else:
                        amount = size
            
            else:
                if side=="buy":
                    amount = funds/self.get_price(ccxt_symbol=symbols["symbol_ccxt"])
                else: 
                    amount = size
                params = params
                # amount = 0 

            if order_type == "limit" and price:
                order  = self.exchange.create_order(symbol=symbols["symbol_ccxt"], amount=size,  side=side, type=order_type, price=price, params=params)
            else:
                order = self.exchange.create_order(symbol=symbols["symbol_ccxt"], amount=amount,  side=side, type=order_type, params=params)
            
            order_id  = order["info"]["orderId"]
            borrowSize = float(order["info"]["borrowSize"]) if not order["info"]["borrowSize"] is None else 0

            return order_id, borrowSize
        except Exception as e:
            self.logger.error(f"An error occurred while creating auto repay order: {e}")
            return 
        

    def execute_margin_order(self, symbols=None, size=None, funds=None, order_type="market", side=None, price=None, increments=None):
        """
        ONLY FOR STANDARD MARGIN TRADING NOT HIGH-FREQUENCY MARGIN_V2 !!!!!!!!!

        REQ:    side:                  "buy" or "sell"
        REQ:    order_type:           "limit" or "market"

        Additional params for MARKET ORDERS where EITHER ONE IS REQUIRED TO BE SPECIFIED:
        REQ:    size:                   String  used when creating sell orders
        REQ:    funds:                  String  used when creating buy orders
        
        Additonial params for LIMIT ORDERS:
        REQ:    price:                  String  used when creating limit orders

        Return:
        orderNo:                orderid
        borrowSize:             borrow amount
        loanApplyId:            
        """
        try:
            size = self.round_down(abs(size),increments["coin"]) if size is not None else None
            funds = self.round_down(abs(funds),increments["fiat"]) if funds is not None else None
            price = self.round_down(price, 1) if price is not None else None

            differentiating_exchanges = ["KuCoin"]
            
            params = {"tradeType": "MARGIN_TRADE"}     #MARGIN_TRADE or MARGIN_V2_TRADE

            exchange_name = self.exchange.name

            if exchange_name in differentiating_exchanges:
                if exchange_name == "KuCoin":
                    
                    if size:
                        params["size"] =  size
                
                    if funds and order_type=="market": 
                        params["funds"] = funds

                    if price and order_type == "limit":
                        params["price"] = price
                    
                    if order_type == "limit":
                        params["type"] = order_type 
                    
                    if side=="buy":
                        amount = 0
                    else:
                        amount = size
            
            else:
                if side=="buy":
                    amount = funds/self.get_price(ccxt_symbol=symbols["symbol_ccxt"])
                else: 
                    amount = size
                params = params
                # amount = 0 

            if order_type == "limit" and price:
                order  = self.exchange.create_order(symbol=symbols["symbol_ccxt"], amount=size,  side=side, type=order_type, price=price, params=params)
            else:
                order = self.exchange.create_order(symbol=symbols["symbol_ccxt"], amount=amount,  side=side, type=order_type, params=params)
            
            order_id  = order["info"]["orderId"]
            borrowSize = float(order["info"]["borrowSize"]) if not order["info"]["borrowSize"] is None else 0

            return order_id, borrowSize
        except Exception as e:
            self.logger.error(f"An error occurred while creating margin order: {e}")
            return
        
    
    def execute_stop_order(self, symbols=None, size=None, funds=None, order_type="market", side=None, price=None, stopPrice=None, stop=None, tradeType="MARGIN_TRADE", auto_repay=False, increments=None):
        """
        ONLY FOR STANDARD MARGIN TRADING NOT HIGH-FREQUENCY MARGIN_V2 !!!!!!!!!

        REQ:    side:                  "buy" or "sell"
        REQ:    order_type:           "limit" or "market"
        REQ:   stopPrice:             String  stop price
        REQ:   stop:                  String  stop type (loss or entry), loss price drops below stop price, entry price rises above stop price


        Additional params for MARKET ORDERS where EITHER ONE IS REQUIRED TO BE SPECIFIED:
        REQ:    size:                   String  used when creating sell orders
        REQ:    funds:                  String  used when creating buy orders


        Additonial params for LIMIT ORDERS:
        REQ:    price:                  String  used when creating limit orders

        Return:
        orderNo:                orderid
        borrowSize:             borrow amount
        loanApplyId:            
        """
        try:
            size = self.round_down(abs(size),increments["coin"]) if size is not None else None
            funds = self.round_down(abs(funds),increments["coin"]) if funds is not None else None
            stopPrice = self.round_down(stopPrice, 1)
            price = self.round_down(price, 1) if price is not None else None
            
            differentiating_exchanges = ["KuCoin"]
        
            params = {}      

            exchange_name = self.exchange.name

            if exchange_name in differentiating_exchanges:
                if exchange_name == "KuCoin":

                    params["tradeType"] = tradeType
                    params["stopPrice"] = stopPrice
                    params["stop"] = stop
                    amount = 0

                    if size:
                        params["size"] =  size
                
                    if funds and order_type=="market": 
                        params["funds"] = funds

                    if auto_repay: 
                        params["autoRepay"] = True
                    
                    if price and order_type == "limit":
                        params["price"] = price
                    
                    if order_type == "limit":
                        params["type"] = order_type      
            
                    if side=="buy":
                        amount = 0
                    else:
                        amount = size
                    
            else:
                if side=="buy":
                    amount = funds/self.get_price(symbol=symbols["symbol_ccxt"])
                else: 
                    amount = size
                params = params
                # amount = 0 

            if order_type == "limit" and price:
                order  = self.exchange.create_order(symbol=symbols["symbol_ccxt"], amount=size,  side=side, type=order_type, price=price, params=params)
            else:
                order = self.exchange.create_order(symbol=symbols["symbol_ccxt"], amount=amount, side=side, type=order_type, params=params)
            
            order_id  = order["info"]["orderId"]
        
            return order_id
        except Exception as e:
            self.logger.error(f"An error occurred while creating stop order: {e}")
            return
        

    def cancel_order_by_order_id(self, symbols=None, order_id=None,is_stop=False, tradeType ="MARGIN_TRADE"):
        """
        REQ:        symbol:         e.g. BTC-USDT
        REQ:        orderId         String

        Returns:
        orderId:                orderId of the cancelled order
        """
        params = {"tradeType": tradeType} 
        symbol = symbols["symbol_ccxt"]

        try:
            if is_stop:
                params["stop"] = True
                
            cancelled_order = self.exchange.cancelOrder(order_id, params=params)

            return cancelled_order
        except Exception as e:
            self.logger.error(f"An error occurred while cancelling order {order_id}: {e}")
            return
    
    def cancel_all_orders(self, symbols=None, order_ids=None, is_stop=None, tradeType="MARGIN_TRADE"):
        """
        REQ:        symbol      e.g. BTC-USDT
        
        Kucoint API:
        Link:       https://www.kucoin.com/docs/rest/spot-trading/orders/cancel-order-by-orderid

        Returns:
        orderId:                orderId of the cancelled order
        """
        try:            
            symbol = symbols["symbol_ccxt"]
            params = {"tradeType": tradeType}

            if is_stop:
                params["stop"] = True

            if order_ids:
                params["orderIds"] = order_ids

            cancelled_orders = self.exchange.cancel_all_orders(symbol=symbol, params=params)
            
            return cancelled_orders            
        except Exception as e:
            self.logger.error(f"An error occurred while cancelling all orders: {e}")
            return


    def check_order_triggered(self, symbols, order_id):
        """
        REQ:        order_id.       individual orderId by trade
        REQ:        symbol:         e.g. BTC-USDT
        
        Returns:    True if order still active, False otherwise
        """
        try:
            order_details = self.get_order_details_by_id(symbols=symbols, orderId=order_id)

            if bool(order_details['isActive'][0]):
                return False  # Order is still active, not triggered
            elif not bool(order_details['isActive'][0]) and bool(order_details["cancelExist"][0]):
                return "cancelled"  # Order has been cancelled
            elif not bool(order_details['isActive'][0]) and not bool(order_details["cancelExist"][0]):
                return True  # Order has been executed 

        except Exception as e:
            self.logger.error(f"An error occurred while checking order status: {e}")
            return
        
    def check_sl_or_tp_triggered(self, coin=None, fiat=None, position=None):
        """
        REQ:            Order id:           Order Id to check, use initial buy / sell order from positions tracker
        REG:            symbol:             Symbol to query order details

        Returns:

        Order that has been canceled:       Either TP, SL or not triggered if no stop loss or take profit has been reached yet

        Functionality:                      

        Function queries if any of SL or TP has been triggered. If yes it will get the details and use that to update the positions tracker. It also calculates the correct tax amount. 
        """
        coin = str(coin)
        fiat = str(fiat)
        symbols  = self.get_symbols(coin=coin, fiat=fiat)
        increments = self.get_increments(symbol)
        mins = self.get_min_sizes(symbol)

        order_id = str(position["order_id"])
        symbol = str(position["pair"])
        side_initial_position = str(position["side"])
        size = float(position['size']) if position['size'] is not None else None
        funds = float(position['funds']) if position['funds'] is not None else None

        stop_loss_order_id = position.get('stop_loss_order_id')
        take_profit_order_id = position.get('take_profit_order_id')

        check_initial_order_triggered = self.check_order_triggered(symbols, order_id=order_id)
        check_stop_loss_triggered   = self.check_order_triggered(symbols, stop_loss_order_id)
        check_take_profit_triggered = self.check_order_triggered(symbols, take_profit_order_id)
        
        if check_initial_order_triggered and check_stop_loss_triggered and check_stop_loss_triggered != "cancelled":
            try:
                #cancel tp order
                try:
                    self.cancel_order_by_order_id(symbols=symbols, order_id=take_profit_order_id, is_stop=True)
                except Exception as e:
                    print(f"An error occurred while cancelling take profit order: {e}")
                    print("Cancelling all open orders")
                    self.cancel_all_orders(symbols=symbols, is_stop=True)
                
                self.logger.info(f"{'#'*30}   Stop loss order executed with order id: {stop_loss_order_id}    {'#'*30}")

                sl_order_details =self.get_order_details_by_id(symbols, stop_loss_order_id)

                #timestamp_now = datetime.now().timestamp()*1000    check later in backtest if createdAt is timestamp when order was filled else use timestamp now
                
                #update position respectively
                side = sl_order_details["side"][0]
                
                if side == "sell":
                    borrowed_asset = fiat
                    borrowed_amount = funds
                    closing_amount_invested = sl_order_details["dealFunds"][0]
                    pnl_trade = closing_amount_invested - position['invested_usdt']
                elif side =="buy":
                    borrowed_asset = coin
                    borrowed_amount = size
                    closing_amount_invested = sl_order_details["dealSize"][0]
                    pnl_trade = closing_amount_invested - size

                    if pnl_trade < 0:
                        if pnl_trade < mins["coin"]:
                            pnl_trade += (1.5*mins["coin"])-pnl_trade
                        pnl_trade = self.round_down(pnl_trade, increments["coin"])
                        
                        rebuy_price = self.get_price(symbol=symbol) * (1 + self.slippage)
                        rebuy_order_id = self.execute_margin_order(symbols=symbols, size=abs(pnl_trade), order_type="limit", side=side,price=rebuy_price, increments=increments)
                        position["rebuy_order_id"] = rebuy_order_id
                        #get convert order details and rebuy_order_details
                        rebuy_order_details = self.get_order_details_by_id(symbols, orderId=rebuy_order_id)
                    else:
                        rebuy_order_details = {"fee":0}
                        rebuy_order_details["dealFunds"] = 0
                        
                borrowed_asset_details = self.get_margin_account_details(curr=borrowed_asset) 
                liabilities = borrowed_asset_details["liability"].values[0]
                available = borrowed_asset_details["available"].values[0]
                
                if liabilities >  0:
                    size_repay = liabilities
                    
                    if liabilities > available and side =="buy":
                        owed_amount = liabilities - available
                        
                        if owed_amount <mins["coin"]:
                            owed_amount += (1.5*mins["coin"])-pnl_trade
                        
                        owed_amount = self.round_down(owed_amount, increments["coin"])
                        
                        rebuy_price_debt = self.get_price(symbol=symbol) * (1 + self.slippage)
                        rebuy_order_debt_id = self.execute_margin_order(symbols=symbols, size=owed_amount, order_type="limit", side=side,price=rebuy_price_debt, increments=increments)
                        
                    repaid_order_id, repaid_amount = self.repay_funds(symbols=symbols, curr=borrowed_asset, size=size_repay)
                    self.logger.info(f"Repaid order of amount {repaid_amount}")
                    
                #when short convert coin into usdt
                if side == "buy":
                    coin_balance_after_trade = self.get_account_list(curr=coin, account_type="margin")["available"].values[0]
                    coin_balance_after_trade = self.round_down(coin_balance_after_trade, increments["coin"])
                    
                    if coin_balance_after_trade > 0:
                        convert_order_id = self.execute_margin_order(symbols=symbols, size=coin_balance_after_trade, order_type="market", side="sell",increments=increments)#
                        position["convert_order_id"] = convert_order_id
                        convert_order_id_details = self.get_order_details_by_id(symbols, orderId=convert_order_id)
                    else:
                        position["convert_order_id"] = None
                        convert_order_id_details = {"fee":0}
                        convert_order_id_details["dealFunds"] = 0

                #update position
                position["close_order_id"] = stop_loss_order_id
                position['status'] = 'closed'
                position['timestamp_closed'] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                position['time_closed'] =      dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                position["repaid"] = "repaid"
                
                if side =="buy" and pnl_trade <0:
                    fees = { "sum":     sum([sl_order_details["fee"][0], rebuy_order_details["fee"][0], convert_order_id_details["fee"][0]]), 
                            "single":   [sl_order_details["fee"][0], rebuy_order_details["fee"][0], convert_order_id_details["fee"][0]]}
                elif side =="buy" and pnl_trade >=0:
                    fees = { "sum":     sum([sl_order_details["fee"][0], convert_order_id_details["fee"][0]]), 
                            "single":   [sl_order_details["fee"][0], convert_order_id_details["fee"][0]]}                
                else:
                    fees = sl_order_details["fee"][0]
                
                position["fees_closing_trade"] = fees["sum"] if isinstance(fees, dict) else fees

                closing_balances = self.get_account_list(account_type="margin")
                closing_balance_fiat = closing_balances[closing_balances["currency"]==fiat]["available"].values[0] #self.get_margin_account_details(curr=fiat)["available"].values[0]
                closing_balance_coin = closing_balances[closing_balances["currency"]==coin]["available"].values[0]
                position["closing_balance"] = closing_balance_fiat

                if side == "sell":
                    closing_amount_invested = sl_order_details["dealFunds"][0]
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = (closing_amount_invested - position['invested_usdt'])    #position['closing_balance']
                else:
                    closing_amount_invested = sl_order_details["funds"][0] - convert_order_id_details["dealFunds"][0] + rebuy_order_details["funds"][0]
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = (position['funds']- closing_amount_invested)
                
                if position["pnl"] > 0:
                    self.logger.info(f"{'#'*30} Position closed with profit of {position['pnl']} {fiat}  {'#'*30}")
                    position["owed_tax"] = position["pnl"] * 0.25
                    self.trading_metrics_data["winning_trades"] += 1

                    if position["pnl"] > self.trading_metrics_data["largest_gain"]:
                        self.trading_metrics_data["largest_gain"] = position["pnl"]

                elif position["pnl"] < 0:
                    self.logger.info(f"{'#'*30}   Position closed with loss of {position['pnl']} {fiat}    {'#'*30}")
                    position["owed_tax"] = position["pnl"] * 0.25
                    self.trading_metrics_data["losing_trades"] += 1

                    if position["pnl"] < self.trading_metrics_data["largest_loss"]:
                        self.trading_metrics_data["largest_loss"] = position["pnl"]

                self.save_trading_metrics(self.trading_metrics_path, self.trading_metrics_data)  

                #set open position to None
                self.ledger_data["current_trades"][symbol] = None
                
                #update balances
                self.ledger_data["balances"]["trailing_balances"][self.fiat] = closing_balance_fiat
                self.ledger_data["balances"]["trailing_balances"][coin] = closing_balance_coin
                self.ledger_data["balances"]["trailing_balances"]["total_balance"] = self.calculate_balance_in_fiat(fiat=fiat, coin=coin, getSum=True, get_only_trade_pair_bal=True, account_type="margin")
                
                self.ledger_data["order_details"][order_id]["sl_order_details"] = sl_order_details
                
                if side == "buy":
                    self.ledger_data["order_details"][order_id]["rebuy_order_details"] = rebuy_order_details
                    self.ledger_data["order_details"][order_id]["usdt_convert_order_details"] = convert_order_id_details
                    
                self.update_position(position)

                return str("tp_cancel")
            except:
                self.logger.error("take profit order could not be canceled")
        
        elif check_initial_order_triggered and check_take_profit_triggered and check_take_profit_triggered != "cancelled":
            try:
                #cancel sl order
                try:
                    self.cancel_order_by_order_id(symbols=symbols, order_id=take_profit_order_id, is_stop=True)
                except Exception as e:
                    print(f"An error occurred while cancelling take profit order: {e}")
                    print("Cancelling all open orders")
                    self.cancel_all_orders(symbols=symbols, is_stop=True)

                self.logger.info(f"{'#'*30}    Take profit order executed with order id: {take_profit_order_id}    {'#'*30}")
                tp_order_details =self.get_order_details_by_id(symbols, take_profit_order_id)

                #update position respectively
                side = tp_order_details["side"][0]
                
                if side == "sell":
                    borrowed_asset = fiat
                    borrowed_amount = funds
                    closing_amount_invested = tp_order_details["dealFunds"][0]
                    pnl_trade = closing_amount_invested - position['invested_usdt']
                elif side =="buy":
                    borrowed_asset = coin
                    borrowed_amount = size
                    closing_amount_invested = tp_order_details["dealSize"][0]
                    pnl_trade = closing_amount_invested - size

                    if pnl_trade < 0:
                        
                        if pnl_trade <mins["coin"]:
                            pnl_trade += (1.5*mins["coin"])-pnl_trade
                        pnl_trade = self.round_down(pnl_trade, increments["coin"])
                        
                        rebuy_price = self.get_price(symbol=symbol) * (1 + self.slippage)
                        rebuy_order_id = self.execute_margin_order(symbols=symbols, size=abs(pnl_trade), order_type="limit", side=side,price=rebuy_price,increments=increments)
                        position["rebuy_order_id"] = rebuy_order_id
                        rebuy_order_details = self.get_order_details_by_id(symbols, orderId=rebuy_order_id)
                    else:
                        rebuy_order_details = {"fee":0}
                        rebuy_order_details["dealFunds"] = 0


                borrowed_asset_details = self.get_margin_account_details(curr=borrowed_asset) 
                liabilities = borrowed_asset_details["liability"].values[0]
                available = borrowed_asset_details["available"].values[0]
                
                if liabilities >  0:
                    size_repay = liabilities
                    
                    if liabilities > available and side =="buy":
                        owed_amount = liabilities - available
                        
                        if owed_amount <mins["coin"]:
                            owed_amount += (1.5*mins["coin"])-pnl_trade
                        
                        owed_amount = self.round_down(owed_amount, increments["coin"])
                        
                        rebuy_price_debt = self.get_price(symbol=symbol) * (1 + self.slippage)
                        rebuy_order_debt_id = self.execute_margin_order(symbols=symbols, size=owed_amount, order_type="limit", side=side,price=rebuy_price_debt, increments=increments)

                    repaid_order_id, repaid_amount = self.repay_funds(symbols=symbols, curr=borrowed_asset, size=size_repay)
                    self.logger.info(f"Repaid order of amount {repaid_amount}")
                    
                #when short convert coin into usdt
                if side == "buy":
                    coin_balance_after_trade = self.get_account_list(curr=coin, account_type="margin")["available"].values[0]
                    coin_balance_after_trade = self.round_down(coin_balance_after_trade, increments["coin"])
                    
                    if coin_balance_after_trade > 0:
                        convert_order_id = self.execute_margin_order(symbols=symbols, size=coin_balance_after_trade, order_type="market", side="sell",increments=increments)#
                        position["convert_order_id"] = convert_order_id
                        convert_order_id_details = self.get_order_details_by_id(symbols, orderId=convert_order_id)
                    else:
                        position["convert_order_id"] = None
                        convert_order_id_details = {"fee":0}
                        convert_order_id_details["dealFunds"] = 0
                

                position["close_order_id"] = position.get("take_profit_order_id")
                position['status'] = 'closed'
                position['timestamp_closed'] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                position['time_closed'] =      dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                position["repaid"] = "repaid"
                
                if side =="buy" and pnl_trade <0:
                    fees = { "sum":     sum([sl_order_details["fee"][0], rebuy_order_details["fee"][0], convert_order_id_details["fee"][0]]), 
                            "single":   [sl_order_details["fee"], rebuy_order_details["fee"], convert_order_id_details["fee"]]}
                elif side =="buy" and pnl_trade >=0:
                    fees = { "sum":     sum([sl_order_details["fee"][0], convert_order_id_details["fee"][0]]), 
                            "single":   [sl_order_details["fee"][0], convert_order_id_details["fee"][0]]}                
                else:
                    fees = sl_order_details["fee"][0]
                
                position["fees_closing_trade"] = fees["sum"] if isinstance(fees, dict) else fees

                closing_balances = self.get_account_list(account_type="margin")
                closing_balance_fiat = closing_balances[closing_balances["currency"]==fiat]["available"].values[0] #self.get_margin_account_details(curr=fiat)["available"].values[0]
                closing_balance_coin = closing_balances[closing_balances["currency"]==coin]["available"].values[0]
                position["closing_balance"] = closing_balance_fiat

                if side == "sell":
                    closing_amount_invested = tp_order_details["dealFunds"][0]
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = (closing_amount_invested - position['invested_usdt'])    #position['closing_balance']
                else:
                    closing_amount_invested = tp_order_details["funds"][0] - convert_order_id_details["dealFunds"][0] + rebuy_order_details["funds"][0]
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = (position['funds'] - closing_amount_invested)
                
                if position["pnl"] > 0:
                    self.logger.info(f"{'#'*30} Position closed with profit of {position['pnl']} {fiat}  {'#'*30}")
                    position["owed_tax"] = position["pnl"] * 0.25
                    self.trading_metrics_data["winning_trades"] += 1

                    if position["pnl"] > self.trading_metrics_data["largest_gain"]:
                        self.trading_metrics_data["largest_gain"] = position["pnl"]

                elif position["pnl"] < 0:
                    self.logger.info(f"{'#'*30}   Position closed with loss of {position['pnl']} {fiat}    {'#'*30}")
                    position["owed_tax"] = position["pnl"] * 0.25
                    self.trading_metrics_data["losing_trades"] += 1

                    if position["pnl"] < self.trading_metrics_data["largest_loss"]:
                        self.trading_metrics_data["largest_loss"] = position["pnl"]
                
                self.save_trading_metrics(self.trading_metrics_path, self.trading_metrics_data)

                #set open position to None
                self.ledger_data["current_trades"][symbol] = None
                
                #update balances
                self.ledger_data["balances"]["trailing_balances"][self.fiat] = closing_balance_fiat
                self.ledger_data["balances"]["trailing_balances"][coin] = closing_balance_coin
                self.ledger_data["balances"]["trailing_balances"]["total_balance"] = self.calculate_balance_in_fiat(fiat=fiat, coin=coin, getSum=True, get_only_trade_pair_bal=True, account_type="margin")

                self.ledger_data["order_details"][order_id]["tp_order_details"] = tp_order_details
                
                if side == "buy":
                    self.ledger_data["order_details"][order_id]["rebuy_order_details"] = rebuy_order_details
                    self.ledger_data["order_details"][order_id]["usdt_convert_order_details"] = convert_order_id_details

                self.update_position(position)

                return str("sl_cancel")
            except:
                self.logger.error("Stop loss order could not be canceled")
    
        elif check_initial_order_triggered and not check_stop_loss_triggered and not check_take_profit_triggered:
            return str("not triggered")

    @retry(max_retries=5, delay=2, backoff=2)
    def get_order_details_list(self, symbols, orderId=None, get_open_orders=False, side=None, is_stop=False, tradeType="MARGIN_TRADE", type=None, startAt=None,endAt=None):
        """ 
        OPT:        orderid:    query specific order ID
        OPT:        symbol:     e.g. BTC-USDT
        OPT:        active:     "done" or "active"
        OPT:        side:       "buy" or "sell"
        OPT:        tradeType:  default is "MARGIN_TRADE" but also available "TRADE", "MARGIN_ISOLATED_TRADE"
        OPT:        type:       "market", "limit", "limit_stop", "market_stop"
        OPT:        startAt:    date of when to start to query orders    provide string object of format "Y-m-d h:m"
        OPT:        endAt:      date of when to end to query orders      provide string object of format "Y-m-d h:m"

        Link:       https://www.kucoin.com/docs/rest/spot-trading/orders/get-order-list

        Returns:
        id:     	        Order ID, the ID of an order.
        symbol:	:           symbol
        opType: 	        Operation type: DEAL
        type:   	        order type
        side:   	        transaction direction,include buy and sell
        price:  	        order price
        size:   	        order quantity
        funds:  	        order funds
        dealFunds: 	        executed size of funds
        dealSize:  	        executed quantity
        fee:    	        fee
        feeCurrency:    	charge fee currency
        stp:            	self trade prevention,include CN,CO,DC,CB
        stop:           	stop type, include entry and loss
        stopTriggered:  	stop order is triggered or not
        stopPrice:      	stop price
        timeInForce:       	time InForce,include GTC,GTT,IOC,FOK
        postOnly:       	postOnly
        hidden:         	hidden order
        iceberg:        	iceberg order
        visibleSize:    	displayed quantity for iceberg order
        cancelAfter:    	cancel orders time，requires timeInForce to be GTT
        channel:        	order source
        clientOid:      	user-entered order unique mark
        remark:         	remark
        tags:           	tag order source
        isActive:       	order status, true and false. If true, the order is active, if false, the order is fillled or cancelled
        cancelExist:    	order cancellation transaction record
        createdAt:      	create time
        tradeType:      	The type of trading
        """
        # time.sleep(1)
        retry_count = 0
        max_retries = 5
        order_details = None
        try:
            symbol = symbols["symbol_ccxt"]
            
            params={"tradeType":tradeType}
            
            if tradeType and tradeType != "MARGIN_TRADE":
                params = {"tradeType":tradeType}

            if endAt:
                end_timestamp = self.convert_datetime_to_timestamp(endAt)
                params["till"] = end_timestamp

            if side:
                params["side"] = side

            if is_stop: 
                params["stop"] = is_stop

            if type:
                params["type"] = type

            if get_open_orders:
                while order_details is None and retry_count < max_retries:
                    if startAt:
                        start_timestamp = self.convert_datetime_to_timestamp(startAt)
                        order_details = self.exchange.fetch_open_orders(symbol, start_timestamp, params=params)
                    else:
                        order_details = self.exchange.fetch_open_orders(symbol, params=params)
            else:
                while order_details is None and retry_count < max_retries:
                    if startAt:
                        start_timestamp = self.convert_datetime_to_timestamp(startAt)
                        order_details = self.exchange.fetch_closed_orders(symbol, start_timestamp, params=params)
                    else:
                        order_details = self.exchange.fetch_closed_orders(symbol, params=params)
            
            df = pd.DataFrame()

            for item in range(len(order_details)):
                tmp_df = pd.DataFrame(order_details[item]["info"], index=[0])
                df = pd.concat([df, tmp_df], axis=0)
            
            if orderId:
                df =  df[df.id==orderId]
            
            df.createdAt = [self.convert_timestamp_to_datetime(x) for x in df.createdAt]
            return df 
            
        except Exception as e:
            self.logger.info(f"An error occurred when retrieving order details list: {e}")
            raise Exception(f"An error occurred when retrieving order details list: {str(e)}")
    

    @retry(max_retries=5, delay=2, backoff=2)
    def get_order_details_by_id(self, symbols=None, orderId=None, is_stop=False, tradeType="MARGIN_TRADE"):
        """ 
        OPT:        orderid:    query specific order ID
        OPT:        is_stop:    True if stop order
        OPT:        tradeType:  default is "MARGIN_TRADE" but also available "TRADE", "MARGIN_ISOLATED_TRADE"
        OPT:        type:       "market", "limit", "limit_stop", "market_stop"
        
        Kucoin API:
        Link:       https://www.kucoin.com/docs/rest/spot-trading/orders/get-order-list

        Returns:
        id:     	        Order ID, the ID of an order.
        symbol:	:           symbol
        opType: 	        Operation type: DEAL
        type:   	        order type
        side:   	        transaction direction,include buy and sell
        price:  	        order price
        size:   	        order quantity
        funds:  	        order funds
        dealFunds: 	        executed size of funds
        dealSize:  	        executed quantity
        fee:    	        fee
        feeCurrency:    	charge fee currency
        stp:            	self trade prevention,include CN,CO,DC,CB
        stop:           	stop type, include entry and loss
        stopTriggered:  	stop order is triggered or not
        stopPrice:      	stop price
        timeInForce:       	time InForce,include GTC,GTT,IOC,FOK
        postOnly:       	postOnly
        hidden:         	hidden order
        iceberg:        	iceberg order
        visibleSize:    	displayed quantity for iceberg order
        cancelAfter:    	cancel orders time，requires timeInForce to be GTT
        channel:        	order source
        clientOid:      	user-entered order unique mark
        remark:         	remark
        tags:           	tag order source
        isActive:       	order status, true and false. If true, the order is active, if false, the order is fillled or cancelled
        cancelExist:    	order cancellation transaction record
        createdAt:      	create time
        tradeType:      	The type of trading
        """
        # time.sleep(1)
        retry_count = 0
        max_retries = 5
        order_details = None
        try:
            symbol = symbols["symbol_ccxt"]
            params={"tradeType":tradeType}
            
            if tradeType and tradeType != "MARGIN_TRADE":
                params = {"tradeType":tradeType}

            if is_stop:
                params["stop"] = is_stop

            while order_details is None and retry_count< max_retries:
                order_details = self.exchange.fetchOrder(orderId, symbol, params)
                time.sleep(1)
                retry_count += 1
                
            df = pd.DataFrame(order_details["info"], index=[0])
            
            if is_stop:
                if df["side"].values[0] == "buy":
                        df = df.rename(columns={"takerFeeRate":"fee"})
                else:
                    df = df.rename(columns={"makerFeeRate":"fee"})
                df = df.rename(columns={"funds":"dealFunds", "size":"dealSize"})

            df = df.apply(pd.to_numeric, errors='coerce')
            
            return df
           
        except Exception as e:
            self.logger.info(f"An error occurred when retrieving order details: {e}")
            raise Exception(f"An error occurred when retrieving order details: {e}")
  

    def get_filled_order_list(self, orderId=None, symbol=None, side=None,order_type="market", tradeType="MARGIN_TRADE"):
        """ 
        OPT:        orderId:    orderId
        OPT:        symbol:     e.g. BTC-USDT
        OPT:        side:       "buy" or "sell"
        OPT:        order type: market, limit, 
        OPT:        tradeType:  default is "MARGIN_TRADE" but also available "TRADE", "MARGIN_ISOLATED_TRADE"

        Link:       https://www.kucoin.com/docs/rest/spot-trading/fills/get-filled-list

        Returns:

        symbol: 	        symbol
        tradeId:   	        trade id, it is generated by Matching engine.
        orderId:   	        Order ID, unique identifier of an order.
        counterOrderId: 	counter order id.
        side:           	transaction direction,include buy and sell.
        price:          	order price
        size:           	order quantity
        funds:          	order funds
        type:           	order type,e.g. limit,market,stop_limit.
        fee:               	fee
        feeCurrency:    	charge fee currency
        stop:           	stop type, include entry and loss
        liquidity:      	include taker and maker
        forceTaker:        	forced to become taker, include true and false
        createdAt:      	create time
        tradeType:      	The type of trading : TRADE（Spot Trading）, MARGIN_TRADE (Margin Trading).
        """
        
        try:
            endpoint = f'/api/v1/fills'
            url = 'https://api.kucoin.com' + endpoint

            params={"tradeType":tradeType}

            if orderId:
                params["orderId"] = orderId

            if symbol:
                params["symbol"] = symbol

            if side:
                params["side"] = side
            
            if order_type:
                params["type"] = order_type

            if tradeType and tradeType != "MARGIN_TRADE":
                params = {"tradeType":tradeType}
            
            headers = self.create_sign_in(api_key=self.api_key, api_secret=self.api_secret, api_passphrase=self.api_passphrase, 
                                        method="GET", url=url, endpoint=endpoint, data="", params=params)

            response = requests.get(url, headers=headers, params=params)
            response_json = response.json()

            if response.status_code == 200:
                response_df = response_json["data"]

                return response_df
            else:
                return {"error": f"Failed to fetch data: {response.status_code}", "details": response_json}
        
        except Exception as e:
            print(f"An error occurred for getting the filled orders: {e}")
            return {"error": f"An exception occurred: {str(e)}"}
    
    
    
    def modify_leverage_multiplier(self, symbol=None, leverage_factor=1,isIsolated=False):
        """
        REQ:    clientOid:  clientOid, the unique identifier created by the client, use of UUID, with a maximum length of 128 bits.
        REQ:    currency:   Currency
        REQ:    from:       Payment Account Type: main, trade, trade_hf, margin, isolated, margin_v2, isolated_v2
        REQ:    to:         Receiving Account Type: main, trade, trade_hf, margin, isolated, margin_v2, isolated_v2, contract
        REQ:    amount:     Transfer amount, the precision being a positive integer multiple of the Currency Precision

        OPT:    from-tag:   Trading pair, required when the payment account type is isolated, e.g.: BTC-USDT
        OPT:    to_tag:     Trading pair, required when the payment account type is isolated, e.g.: BTC-USDT
        
        """

        clientOid = self.client_user_id

        endpoint = '/api/v3/position/update-user-leverage'  #/api/v1/margin/order/test
        url = 'https://api.kucoin.com' + endpoint

        data = {"leverage": leverage_factor,
            "isIsolated": isIsolated
            }

        if symbol and isIsolated:
            data["symbol"] = symbol
        
        data_json=json.dumps(data)

        headers = self.create_sign_in(api_key=self.api_key, api_secret=self.api_secret, api_passphrase=self.api_passphrase, 
                                    method="POST",url=url, endpoint=endpoint, data=data_json)
        
        request = requests.post(url, data=data_json, headers=headers)
        request_json = request.json()
        
        if request.status_code == 200:
            self.logger.info(f"Leverage multiplier has been updated to {leverage_factor}")
        else:
            self.logger.info(f"Failed to update leverage multiplier: {request_json}")
#################################################################################################################################################################
#
#
#                                                                  MARKET FUNCTIONS
#
#################################################################################################################################################################

    def get_fiat_price(self, base, currencies):
        """"
        calculates fiat values for each currency.
        base:           EUR, USD, JPY
        currencies:     BTC, ETH (multiple allowed when comma seperated)
        
        """
        
        try:
            # symbol = currencies +"/"+ base
            # price_1 = self.get_price(symbol=symbol)
            
            endpoint = '/api/v1/prices'
            url = 'https://api.kucoin.com' + endpoint

            params = {"base": base, "currencies": currencies}

            headers = self.create_sign_in(api_key=self.api_key, api_secret=self.api_secret, api_passphrase=self.api_passphrase, 
                                        method="GET", url=url, endpoint=endpoint, data="")

            response = requests.get(url, headers=headers, params=params)
            response_json = response.json()
            
            if response.status_code == 200:
                price_df = pd.DataFrame.from_dict(response_json)
                price_df.drop(columns=["code"], inplace=True)
                price_df["data"] = price_df["data"].astype(float)
                price_df = price_df.T
                return price_df
            else:
                return {"error": f"Failed to fetch data: {response.status_code}", "details": response_json}
        
        except Exception as e:
            print(f"An error occurred while retrieving USDT: {e}")
            return {"error": f"An exception occurred: {str(e)}"}
        
    def get_price(self, symbol=None, close=False, open=False, high=False, low=False, volume=False, timestamp=False, since=None, limit=None):
        """"
        calculates fiat values for each currency.
        symbol:         e.g. BTC/USDT
        """
        ccxt_symbol = symbol
        if "-" in symbol:
            ccxt_symbol = ccxt_symbol.replace("-", "/")
        
        try:
            if close:
                return self.exchange.fetchTicker(ccxt_symbol)["close"]
            elif open:
                return self.exchange.fetchTicker(ccxt_symbol)["open"]
            elif high:
                return self.exchange.fetchTicker(ccxt_symbol)["high"]
            elif low:
                return self.exchange.fetchTicker(ccxt_symbol)["low"]
            elif volume:
                return self.exchange.fetchTicker(ccxt_symbol)["volume"]
            elif timestamp:
                return self.exchange.fetchTicker(ccxt_symbol)["timestamp"]
            else:
                return self.exchange.fetchTicker(ccxt_symbol)["last"]
        except Exception as e:
            print(f"An error occurred while retrieving data for {ccxt_symbol} : {e}")
            return {"error": f"An exception occurred: {str(e)}"}

    def get_ohlc(self, symbol, timeframe="1m", startAt=None, endAt=None, limit=1500, onlyLast=True):
        """
        REQ:           symbol:         trading pair e.g. BTC-USDT
        REQ:           timeframe:      1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 1w, 1M
        OPT:           since:          timestamp in milliseconds
        OPT:           until:          timestamp in milliseconds
        OPT:           limit:          limit the amount of data returned
        """
        params = {}
        
        ccxt_symbol = symbol
        if "-" in symbol:
            ccxt_symbol = ccxt_symbol.replace("-", "/")
        
        if startAt is not None and pd.isinstance(startAt, dt.datetime):
            startAt = pd.Timestamp(startAt).tz_localize("Europe/Berlin", ambiguous=True).tz_convert("UTC")
            startAt = self.convert_datetime_to_timestamp(startAt)
        else:
            time_now = dt.datetime.now()
            if timeframe == "1d":
                now = time_now.replace(hour=0, minute=0,second=0,microsecond=0)
            elif timeframe == "1w":
                now = time_now.replace(hour=0, minute=0,second=0,microsecond=0)
                now = now - dt.timedelta(days=now.weekday())
            elif timeframe == "1h":
                now = time_now.replace(minute=0,second=0,microsecond=0).replace(hour=1 * (time_now.hour // 1))
            elif timeframe == "1m":
                now = time_now.replace(second=0,microsecond=0).replace(minute=1 * (time_now.minute // 1))
            elif timeframe == "15m":
                now = time_now.replace(second=0, microsecond=0).replace(minute=15 * (time_now.minute // 15))
                now = now - dt.timedelta(minutes=now.minute % 15)
            elif timeframe == "5m":
                now = time_now.replace(second=0, microsecond=0).replace(minute=5 * (time_now.minute // 5))
                now = now - dt.timedelta(minutes=now.minute % 5)
            elif timeframe == "30m":
                now = time_now.replace(second=0,microsecond=0).replace(minute=30 * (time_now.minute // 30))
                now = now - dt.timedelta(minutes=now.minute % 30)
            elif timeframe == "4h":
                now = time_now.replace(minute=0,second=0,microsecond=0).replace(hour=4 * (time_now.hour // 4))
                now = now - dt.timedelta(hours=now.hour % 4)
                now = now + dt.timedelta(hours=1) #add 1 hour since gmt+1 is 1 hour ahead not 20 but 21 for example
            elif timeframe == "6h":
                now = time_now.replace(minute=0,second=0,microsecond=0).replace(hour=6 * (time_now.hour // 6))
                now = now - dt.timedelta(hours=now.hour % 6)
                now = now + dt.timedelta(hours=1) #add 1 hour since gmt+1 is 1 hour ahead not 20 but 21 for example
            
            if timeframe not in ["1w"]:    
                now = pd.Timestamp(now).tz_localize("Europe/Berlin", ambiguous=True).tz_convert("UTC")
            startAt = self.convert_datetime_to_timestamp(now)
        
        if endAt is not None:
            if timeframe not in ["1w"]:
                endAt = pd.Timestamp(endAt).tz_localize("Europe/Berlin", ambiguous=True).tz_convert("UTC")
            endAt = self.convert_datetime_to_timestamp(endAt)
            params = {"endAt":endAt, "startAt":startAt, "type":timeframe}
        
        try:
            if bool(params):
                data = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, startAt, limit, params)
            else:
                data = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, startAt, limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = [self.convert_timestamp_to_datetime(x) for x in df["timestamp"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            if onlyLast:
                return df.iloc[-1]
            return df
        except Exception as e:
            print(f"An error occurred while retrieving data for {ccxt_symbol} : {e}")
            return {"error": f"An exception occurred: {str(e)}"}

        
        
    def calc_price_from_details(self, dealSize=None, dealFunds=None):

        price = dealFunds / dealSize  
        price = round(price, 2)
        return price
    
    def get_symbol_list(self, symbol=None, specific_param=None):
        """
        OPT:            symbol:             Symbol to be queried, else all will be returned
        OPT:            specific_param:     Select a param from below to only return that value
        
        link:           https://www.kucoin.com/docs/rest/spot-trading/market-data/get-symbols-list

        Returns:
        symbol:     	    unique code of a symbol, it would not change after renaming
        name:       	    Name of trading pairs, it would change after renaming
        baseCurrency:	    Base currency,e.g. BTC.
        quoteCurrency:	    Quote currency,e.g. USDT.
        market:     	    The trading market.
        baseMinSize:	    The minimum order quantity requried to place an order.
        quoteMinSize:	    The minimum order funds required to place a market order.
        baseMaxSize:	    The maximum order size required to place an order.
        quoteMaxSize:	    The maximum order funds required to place a market order.
        baseIncrement:	    Quantity increment: The quantity for an order must be a positive integer multiple of this increment. Here, the size refers to the quantity of the base currency for the order. For example, for the ETH-USDT trading pair, if the baseIncrement is 0.0000001, the order quantity can be 1.0000001 but not 1.00000001.
        quoteIncrement:	    Quote increment: The funds for a market order must be a positive integer multiple of this increment. The funds refer to the quote currency amount. For example, for the ETH-USDT trading pair, if the quoteIncrement is 0.000001, the amount of USDT for the order can be 3000.000001 but not 3000.0000001.
        priceIncrement:	    Price increment: The price of an order must be a positive integer multiple of this increment. For example, for the ETH-USDT trading pair, if the priceIncrement is 0.01, the order price can be 3000.01 but not 3000.001.
        feeCurrency:	    The currency of charged fees.
        enableTrading:	    Available for transaction or not.
        isMarginEnabled:	Available for margin or not.
        priceLimitRate:	    Threshold for price portection
        minFunds:       	the minimum spot and margin trading amounts
        """
        try:
            endpoint = f'/api/v2/symbols'
            url = 'https://api.kucoin.com' + endpoint

            headers = self.create_sign_in(api_key=self.api_key, api_secret=self.api_secret, api_passphrase=self.api_passphrase, 
                                        method="GET",url=url, endpoint=endpoint, data="")

            response = requests.get(url, headers=headers)    
            response_json = response.json()

            # Check if response is successful
            if response.status_code == 200:
                df = pd.DataFrame()
                
                for item in response_json["data"]:
                    tmp_df = pd.DataFrame(item, index=[0]).from_dict(item, orient='index').T
                    df = pd.concat([df, tmp_df], axis=0)
                  
                if symbol is not None:
                    df = df[df.symbol == symbol]
                    
                if specific_param is not None:
                    return df[specific_param]
                else:
                    return df
                
            else:
                return {"error": f"Failed to fetch data: {response.status_code}", "details": response_json}
        
        except Exception as e:
            print(f"An error occurred when retrieving account details: {e}")
            return {"error": f"An exception occurred: {str(e)}"}
    
#################################################################################################################################################################
#
#
#                                                                  API & SUPPORT FUNCTIONS 
#
#################################################################################################################################################################

    def round_up(self, value, n):
        factor = 10 ** n
        return math.ceil(value * factor) / factor

    def round_down(self, value, n):
        increment_str = str(value)
        if 'e-' in increment_str or 'E-' in increment_str:
            value = float(increment_str)
            precision = abs(int(increment_str.split('e-')[1]))
        else:
            if '.' in increment_str:
                precision = len(increment_str.split('.')[1])
            else:
                precision = 0
            
        if precision>n:
            factor = 10 ** n
            rounded_value =  math.floor(value * factor) / factor        
        else:
            rounded_value = value

        # Format the result to have at most n decimal places
        formatted_value = format(rounded_value, f'.{n}f')
        return float(formatted_value)
    
    def get_rounding_precision(self, value):
        # Convert the increment to a string and split at the decimal point
        increment_str = str(value)
        if '.' in increment_str:
            # Count the number of digits after the decimal point to get the precision
            precision = len(increment_str.split('.')[1])
        else:
            # If there's no decimal point, no rounding is needed
            precision = 0
            
        return int(precision)
    
    def get_increments(self, symbol):
        """
        REQ:        symbol:     trading pair e.g. BTC-USDT

        Returns:    increments:  dictionary of increments for coin, quote and price
        """
        symbol_details = self.get_symbol_list(symbol=symbol)
        increments = {}
        increments["coin"] = self.get_rounding_precision(symbol_details["baseIncrement"].values[0])
        increments["fiat"] = self.get_rounding_precision(symbol_details["quoteIncrement"].values[0])
        increments["price"] = self.get_rounding_precision(symbol_details["priceIncrement"].values[0])

        return increments
    
    def get_min_sizes(self, symbol):
        """
        REQ:        symbol:     trading pair e.g. BTC-USDT

        Returns:    mins:       dictionary of minimum size and funds minSize, minFunds
        
        """
        symbol_details = self.get_symbol_list(symbol=symbol)
        mins = {}
        mins["coin"] = float(symbol_details["baseMinSize"].values[0])
        mins["fiat"] = float(symbol_details["quoteMinSize"].values[0])
        
        return mins
    
    def get_symbols(self, coin, fiat):
        """
        REQ:        coin:       coin symbol
        REQ:        fiat:       fiat symbol

        Returns:    symbols:     trading pair e.g BTC-USDT and ccxt format BTC/USDT
        """
        symbols = {}
        symbols["symbol"] = coin + "-" + fiat
        symbols["symbol_ccxt"] = coin + "/" + fiat
        return symbol
    
    
    def get_server_timestamp(self):
            """
            Link:       https://www.kucoin.com/docs/rest/spot-trading/market-data/get-server-time

            Returns:    returns server time in iso format
            """
            try:
                endpoint = f'/api/v1/timestamp'
                url = 'https://api.kucoin.com' + endpoint


                headers = self.create_sign_in(api_key=self.api_key, api_secret=self.api_secret, api_passphrase=self.api_passphrase, 
                                            method="GET", url=url, endpoint=endpoint, data="")

                timestamp = requests.get(url, headers=headers)
                timestamp = timestamp.json()

                return timestamp["data"]
            
            except Exception as e:
                print(f"An error occurred when retrieving server timestamp: {e}")


    def get_ntp_time(self):
        try:
            return int(time.time() * 1000)
             # Return time in milliseconds
        except Exception as e:
            print(f"Failed to get system time: {e}. using NTP time instead")
              # Fallback to local time
            client = ntplib.NTPClient()
            response = client.request('pool.ntp.org', version=3)
            return int(response.tx_time * 1000) 
    
     
    def create_sign_in(self, api_key, api_secret, api_passphrase, method, url, endpoint, data='', params=None):
        """
        Generates the headers required for Kucoin API requests.

        :param api_key: Your Kucoin API key.
        :param api_secret: Your Kucoin API secret.
        :param api_passphrase: Your Kucoin API passphrase.
        :param method: The HTTP method (e.g., 'GET', 'POST').
        :param endpoint: The API endpoint (e.g., '/api/v1/accounts').
        :param data: The request body, if any (for POST requests).
        :return: A dictionary containing the headers.
        """
        # os.system('w32tm /resync') #comment out if code throws error executing, might requre admin privilges
        now = self.get_ntp_time()
        # now2 = int(time.time() * 1000)
        # now = utils.get_ntp_time(in_ms=True)
        str_to_sign = str(now) + method + endpoint

        if method in ['GET', 'DELETE'] and params:
            # Sort and encode parameters
            sorted_params = sorted(params.items())
            encoded_query_string = urllib.parse.urlencode(sorted_params)
            str_to_sign += '?' + encoded_query_string  # Add '?' before query string
        elif data:
            str_to_sign += data

        signature = base64.b64encode(hmac.new(api_secret.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest()) #.decode()

        passphrase = base64.b64encode(hmac.new(api_secret.encode('utf-8'), api_passphrase.encode('utf-8'), hashlib.sha256).digest()) #.decode()

        headers = {
            "KC-API-SIGN": signature,
            "KC-API-TIMESTAMP": str(now),
            "KC-API-KEY": api_key,
            "KC-API-PASSPHRASE": passphrase,
            "KC-API-KEY-VERSION": "2"
        }

        if method == 'POST':
            headers["Content-Type"] = "application/json"

        return headers
    
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
#                                                                  ACCOUNT FUNCTIONS
#
#################################################################################################################################################################

    def inner_transfer(self, currency=None, from_acc=None, to_acc=None, amount=None, from_tag=None, to_tag=None):
        """
        REQ:    clientOid:  clientOid, the unique identifier created by the client, use of UUID, with a maximum length of 128 bits.
        REQ:    currency:   Currency
        REQ:    from:       Payment Account Type: main, trade, trade_hf, margin, isolated, margin_v2, isolated_v2
        REQ:    to:         Receiving Account Type: main, trade, trade_hf, margin, isolated, margin_v2, isolated_v2, contract
        REQ:    amount:     Transfer amount, the precision being a positive integer multiple of the Currency Precision

        OPT:    from-tag:   Trading pair, required when the payment account type is isolated, e.g.: BTC-USDT
        OPT:    to_tag:     Trading pair, required when the payment account type is isolated, e.g.: BTC-USDT
        
        """

        clientOid = self.client_user_id

        endpoint = '/api/v2/accounts/inner-transfer'  #/api/v1/margin/order/test
        url = 'https://api.kucoin.com' + endpoint

        data = {
            "clientOid": clientOid,
            "currency": currency,
            "from": from_acc,
            "to": to_acc,
            "amount": amount
            }
    
        data_json=json.dumps(data)

        headers = self.create_sign_in(api_key=self.api_key, api_secret=self.api_secret, api_passphrase=self.api_passphrase, 
                                    method="POST",url=url, endpoint=endpoint, data=data_json)
        
        requests.post(url, data=data_json, headers=headers)


    def inner_flex_transfer(self, currency=None,amount=None, from_acc=None, to_acc=None):

        clientOid = self.client_user_id
        type = "INTERNAL"

        endpoint = '/api/v3/accounts/universal-transfer'  #/api/v1/margin/order/test
        url = 'https://api.kucoin.com' + endpoint


        data = {
            "clientOid": clientOid,
            "currency": currency,
            "fromAccountType": from_acc,
            "toAccountType": to_acc,
            "amount": amount,
            "type": type
            }
        
        isolated_param = currency +"-USDT"
        if from_acc  == "ISOLATED" or from_acc == "ISOLATED_V2":
            data["fromAccountTag"] =   isolated_param

        if to_acc == "ISOLATED" or to_acc == "ISOLATED_V2":
            data["toAccountTag"] = isolated_param

        data_json=json.dumps(data)

        headers = self.create_sign_in(api_key=self.api_key, api_secret=self.api_secret, api_passphrase=self.api_passphrase, 
                                    method="POST",url=url, endpoint=endpoint, data=data_json)
        
        requests.post(url, data=data_json, headers=headers)


    def get_account_list(self, curr=None, account_type=None):
        """
        OPT:            currency:       e.g. KCS
        OPT:            type:           main, trade, margin, trade_hf

        Link:           https://www.kucoin.com/docs/rest/account/basic-info/get-account-list-spot-margin-trade_hf
        
        Returns:

        id:             id of account:
        currency:   	Currency
        type:       	Account type:，main、trade、trade_hf、margin
        balance:    	Total funds in the account
        available:  	Funds available to withdraw or trade
        holds:      	Funds on hold (not available for use)
        """
        try:
            df = pd.DataFrame(self.exchange.fetchAccounts())

            if curr is not None:
                df = df[df["currency"]==curr]

            if account_type is not None:
                df = df[df["type"]==account_type]
                
            value_selection = df["info"]
            if len(value_selection) == 0:
                value_selection_index = value_selection.index.values[0]
                df = pd.DataFrame(dict(value_selection[value_selection_index]), index=range(len(value_selection_index)))
            else:
                df = pd.DataFrame()
                value_selection_index = value_selection.index.tolist()
                for i in value_selection_index:
                    tmp_df = pd.DataFrame(value_selection[i], index=[0])
                    df = pd.concat([df, tmp_df], axis=0)
                    
            # Pass an index of [0] to create the DataFrame from scalar values

            df[["balance","available","holds"]] = df[["balance","available","holds"]].astype(float)
            return df
                    
        except Exception as e:
            print(f"An error occurred: {e}")
            return {"error": f"An exception occurred: {str(e)}"}


    def get_account_balance(self, margin_mode=None):
        """
        REQ:            accountid:       string
        link:           https://www.kucoin.com/docs/rest/account/basic-info/get-account-detail-spot-margin-trade_hf

        Returns:
        currency:   	The currency of the account
        balance:    	Total funds in the account
        holds:      	Funds on hold (not available for use)
        available:  	Funds available to withdraw or trade
        """
        try:
            params = {}

            if margin_mode:
                params["marginMode"] = margin_mode
            balance = self.exchange.fetchBalance()
        except Exception as e:
            print(f"An error occurred when retrieving account details: {e}")
            return {"error": f"An exception occurred: {str(e)}"}
        
        
    def get_margin_account_details(self, accounts=True, assets=False, liabilities=False, debtRatio=False, curr=None):
        """
        OPT:            accounts:           Boolean, True or False for getting margin account list
        OPT:            assets:             Boolean, True or False for getting total margin assets
        OPT:            liabilities:        Boolean, True or False for getting total margin liabilities
        OPT:            debtRatio:          Boolean, True or False for getting margin debt Ratio
        OPT:            curr:               Coin to query e.g. BTC
        link:           https://www.kucoin.com/docs/rest/funding/funding-overview/get-account-detail-cross-margin

        Returns:
        totalAssetOfQuoteCurrency:      total assets in margin hf 
        totalLiabilityOfQuoteCurrency:  total liabilities in margin hf
        debtRatio:                      debtRatio
        status:                         Position status; EFFECTIVE-effective, BANKRUPTCY-bankruptcy liquidation, LIQUIDATION-closing, REPAY-repayment, BORROW borrowing
        currency:   	                The currency of the account
        liability:                      how many funds are borrowed 
        total:    	                    Total funds in the account
        available:  	                Funds available to withdraw or trade
        hold:      	                    Funds on hold (not available for use)
        maxBorrowSize:                  max borrow size
        borrowEnabled:                  if max borrowing is enabled
        repayEnabled:                   Support repay or not
        transferInEnabled               transferInEnabled
        """
        try:

            
            endpoint = f'/api/v3/margin/accounts'
            url = 'https://api.kucoin.com' + endpoint

            headers = self.create_sign_in(api_key=self.api_key, api_secret=self.api_secret, api_passphrase=self.api_passphrase, 
                                        method="GET",url=url, endpoint=endpoint, data="")

            response = requests.get(url, headers=headers)    
            response_json = response.json()

            # Check if response is successful
            if response.status_code == 200:
                df = response_json["data"]

                if accounts:
                    df = pd.DataFrame(df["accounts"])
                    df[["total","available","hold","liability", "maxBorrowSize"]] = df[["total","available","hold","liability", "maxBorrowSize"]].astype(float)
                    
                if curr is not None:
                    df = df[df["currency"] == curr]
                    return df
                
                elif assets:
                    return df['totalAssetOfQuoteCurrency']
                elif liabilities:
                    return df['totalLiabilityOfQuoteCurrency']
                elif debtRatio:
                    return df['debtRatio']
                else:
                    return df
            else:
                return {"error": f"Failed to fetch data: {response.status_code}", "details": response_json}
            
        except Exception as e:
            print(f"An error occurred when retrieving account details: {e}")
            return {"error": f"An exception occurred: {str(e)}"}

    
    def get_isolated_margin_account_details(self, accounts=True, assets=False, liabilities=False, debtRatio=False, curr=None):
        """
        OPT:            accounts:           Boolean, True or False for getting margin account list
        OPT:            assets:             Boolean, True or False for getting total margin assets
        OPT:            liabilities:        Boolean, True or False for getting total margin liabilities
        OPT:            debtRatio:          Boolean, True or False for getting margin debt Ratio
        OPT:            curr:               Coin to query e.g. BTC
        link:           https://www.kucoin.com/docs/rest/funding/funding-overview/get-account-detail-isolated-margin

        Returns:
        totalAssetOfQuoteCurrency:      total assets in margin hf 
        totalLiabilityOfQuoteCurrency:  total liabilities in margin hf
        debtRatio:                      debtRatio
        status:                         Position status; EFFECTIVE-effective, BANKRUPTCY-bankruptcy liquidation, LIQUIDATION-closing, REPAY-repayment, BORROW borrowing
        currency:   	                The currency of the account
        liability:                      how many funds are borrowed 
        total:    	                    Total funds in the account
        available:  	                Funds available to withdraw or trade
        hold:      	                    Funds on hold (not available for use)
        maxBorrowSize:                  max borrow size
        borrowEnabled:                  if max borrowing is enabled
        repayEnabled:                   Support repay or not
        transferInEnabled               transferInEnabled
        """
        try:
            endpoint = f'/api/v3/isolated/accounts'
            url = 'https://api.kucoin.com' + endpoint

            headers = self.create_sign_in(api_key=self.api_key, api_secret=self.api_secret, api_passphrase=self.api_passphrase, 
                                        method="GET",url=url, endpoint=endpoint, data="")

            response = requests.get(url, headers=headers)    
            response_json = response.json()

            # Check if response is successful
            if response.status_code == 200:
                df = response_json["data"]

                if accounts:
                    df = pd.DataFrame(df["accounts"])
                    df[["total","available","hold","liability", "maxBorrowSize"]] = df[["total","available","hold","liability", "maxBorrowSize"]].astype(float)
                    
                    if curr is not None:
                      df = df[df["currency"] == curr]
                    return df
                elif assets:
                    return df['totalAssetOfQuoteCurrency']
                elif liabilities:
                    return df['totalLiabilityOfQuoteCurrency']
                elif debtRatio:
                    return df['debtRatio']
                else:
                    return df
            else:
                return {"error": f"Failed to fetch data: {response.status_code}", "details": response_json}
        
        except Exception as e:
            print(f"An error occurred when retrieving account details: {e}")
            return {"error": f"An exception occurred: {str(e)}"}


    # def calculate_pnl(self):
    #     try:
    #         pnl_report = []
    #         for position in self.ledger_data["positions"]:
    #             # Assuming each position has 'symbol', 'size', 'side', and 'entry_price'
    #             execution_price = self.get_usdt_price("USDT", position['symbol'])
    #             entry_price = position['entry_price']
    #             size = position['size']

    #             # Calculate PnL
    #             if position['side'] == 'buy':  # Long position
    #                 pnl = (current_price - entry_price) * size
    #             else:  # Short position
    #                 pnl = (entry_price - current_price) * size

    #             pnl_report.append({
    #                 'trade_id': position['trade_id'],
    #                 'pair': position['pair'],
    #                 'size': size,
    #                 'entry_price': entry_price,
    #                 'current_price': current_price,
    #                 'pnl': pnl
    #             })

    #         return pnl_report
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #         return []



    def calculate_balance_in_fiat(self,  fiat=None, coin=None, use_only_available=False, getSum=False, get_only_trade_pair_bal = False, hf=False, account_type=None):
        """
        REQ:        fiat:                   e.g. fiat currency to calculate value for e.g. EUR or USD, or USDT
        OPT:        coin:                   specify if you want to calc the value for a specific coin list or single string
        OPT:        use_only_available:     use only the available balance in case you should be in a tradem, def is False
        OPT:        getSum:                 Calculates the sum of all queried balances
        OPT:        get_only_trade_pair_bal:Returns only the balance of the trade pair
        OPT:        hf:                     Default set to True, if true returns only balance in HF account, else all other balances
        OPT:        account_type:           main, trade, trade_hf, margin, isolated, margin_v2, isolated_v2
        """
        if hf==True:
            account_balances = self.get_margin_account_details()
            balance = "available" if use_only_available else "total"
        else:
            account_balances = self.get_account_list(account_type=account_type)
            balance = "available" if use_only_available else "balance"
            # account_balances[["balance","available","holds"]] = account_balances[["balance","available","holds"]].astype(float)

        account_balances = account_balances.loc[account_balances[balance]!=0]
        account_balances = account_balances.groupby(["currency"]).sum(numeric_only=True)
        
        if coin is not None:
            if get_only_trade_pair_bal:
               account_balances = account_balances.loc[account_balances.index.isin([coin, fiat])]

            else:
                account_balances = account_balances.loc[account_balances.index==coin]
        
        fiat_balance_df = pd.DataFrame(columns=account_balances.index, index=["value"])
        total_balance = 0

        for coin in account_balances.index:
            asset_amount = account_balances.loc[coin, balance]
            asset_price = 0
            if fiat == "EUR" or fiat == "USD":
                asset_price = self.get_fiat_price(fiat, coin).loc["data",coin]
            else:
                if coin != fiat and coin not in ["EUR","USD"]:
                    symbol = coin + "-" + fiat
                    asset_price = self.get_price(symbol)
                elif coin == fiat:
                    asset_price = 1
                elif coin in ["EUR","USD"]:
                    asset_price = self.get_fiat_price(coin, fiat).loc["data",fiat]
            balance_value = asset_amount * asset_price
            if getSum:
                total_balance += balance_value
            else:
                fiat_balance_df.loc["value",coin] = balance_value

        if getSum:
            return total_balance
        else:
            return fiat_balance_df
    

    def update_historical_values(self, current_total_value):
        # Update historical values in the ledger
        balances = self.ledger_data.get("balances")
        now = datetime.now()

        # Update the historical values
        balances["last_saved_value"] = current_total_value
        balances["one_day_ago_value"] = balances.get(now - timedelta(days=1).strftime("%Y-%m-%d"), current_total_value)
        balances["one_week_ago_value"] = balances.get(now - timedelta(weeks=1).strftime("%Y-%m-%d"), current_total_value)
        balances["one_month_ago_value"] = balances.get((now - timedelta(days=30)).strftime("%Y-%m-%d"), current_total_value)

        # Save to ledger
        # self.ledger_data["historical_values"] = balances
        self.save_data(self.ledger_path)


    def calculate_total_pnl(self):
        try:
            # Current total value: Sum of current positions in USDT + current account balance in USDT
            current_total_value = self.calculate_current_positions_value() + self.calculate_current_account_balance()

            # Load historical values
            historical_values = self.ledger_data.get("balances", {})
            last_saved_value = historical_values.get("last_saved_value", 0)
            one_day_ago_value = historical_values.get("one_day_ago_value", 0)
            one_week_ago_value = historical_values.get("one_week_ago_value", 0)
            one_month_ago_value = historical_values.get("one_month_ago_value", 0)
            inception_value = historical_values.get("inception_value", 0)

            # Calculate P&L
            pnl = {
                "since_last_save": current_total_value - last_saved_value,
                "since_one_day_ago": current_total_value - one_day_ago_value,
                "since_one_week_ago": current_total_value - one_week_ago_value,
                "since_one_month_ago": current_total_value - one_month_ago_value,
                "since_inception": current_total_value - inception_value
            }

            # Update last saved value
            self.update_historical_values(current_total_value)

            return pnl
        except Exception as e:
            print(f"An error occurred: {e}")
            return {}

#################################################################################################################################################################
#
#
#                                                                  MARGIN FUNCTIONS
#
#################################################################################################################################################################


    def borrow_funds(self, symbols, currency=None, size =None, symbol=None, isisolated=None):
        """
        REQ:        currency:   borrowed currency   e.g. KCS
        REQ:        size:       borrowed amount
        OPT:        symbol:     Trading-pair, mandatory for isolated margin account
        OPT:        isisolated: true-isolated, false-cross, default: cross
        Link:        https://www.kucoin.com/docs/rest/margin-trading/margin-trading-v3-/margin-borrowing   

        Returns:
        orderNo:        Borrow order number
        actualSize:     actual borrowed amount
        """
        try:
            symbol = symbols["symbol_ccxt"]

            differentiating_exchanges = ["KuCoin"]
            
            params = {}
            exchange_name = self.exchange.name

            if exchange_name in differentiating_exchanges:
                if exchange_name == "KuCoin":
    
                    if symbol:
                        params["symbol"] = symbol.replace("/", "-")
                    
                    if isisolated:
                        params["isisolated"] = isisolated
            
            amount = size
            borrow_order = self.exchange.borrowCrossMargin(currency, amount, params)
            borrow_order_id = borrow_order["id"]
            borrow_amount = borrow_order["amount"]

            return borrow_order_id,borrow_amount
            
        
        except Exception as e:
            print(f"An error occurred when borrowing funds of size {size}: {e}")
            return {"error": f"An exception occurred: {str(e)}"}



    def repay_funds(self, symbols=None, curr = None, size =None, symbol=None, isisolated=None):
        """
        REQ:        currency:   borrowed currency e.g. KCS
        REQ:        size:       repayment amount
        OPT:        symbol:     Trading-pair, mandatory for isolated margin account
        OPT:        isisolated: true-isolated, false-cross, default: cross
        Link:       https://www.kucoin.com/docs/rest/margin-trading/margin-trading-v3-/repayment

        Returns:
        orderNo:        Repayment order number
        actualSize:     actual repayment amount
        """
        try:
            symbol = symbols["symbol_ccxt"]
            differentiating_exchanges = ["KuCoin"]
            
            params = {}
            exchange_name = self.exchange.name

            if exchange_name in differentiating_exchanges:
                if exchange_name == "KuCoin":
                    
                    if symbol and isisolated:
                        params["symbol"] = symbol.replace("/", "-")
                    
                    if isisolated:
                        params["isisolated"] = isisolated
            
            repay_order = self.exchange.repayCrossMargin(curr, size, params)
            repay_order_id = repay_order["id"]
            borrow_amount = repay_order["amount"]

            return repay_order_id,borrow_amount
        
        except Exception as e:
            print(f"An error occurred when repaying funds of size {size}.: {e}")
            return {"error": f"An exception occurred: {str(e)}"}
        
    def pay_off_all_debts(self, curr=None, symbol=None, isisolated=False):
        """
        REQ:        currency:   borrowed currency e.g. KCS
        OPT:        symbol:     Trading-pair, mandatory for isolated margin account
        OPT:        isisolated: true-isolated, false-cross, default: cross
        Link:       https://www.kucoin.com/docs/rest/margin-trading/margin-trading-v3-/repayment

        Returns:
        orderNo:        Repayment order number
        actualSize:     actual repayment amount
        """
        try:
            borrowed_asset_details = self.get_margin_account_details(curr=curr) 
            symbols = self.get_symbols(coin=curr, fiat=self.fiat)
            increments = self.get_increments(symbol=symbols["symbol"])
            mins = self.get_min_sizes(symbol=symbols["symbol"])
            
            liabilities = borrowed_asset_details["liability"].values[0]
            available = borrowed_asset_details["available"].values[0]
            
            if liabilities >  0:
                size_repay = liabilities
                
                if liabilities > available:
                    owed_amount = liabilities - available
                    
                    if owed_amount <mins["coin"]:
                        owed_amount += (mins["coin"]-owed_amount)
                    
                    owed_amount = self.round_down(owed_amount, increments["coin"])
                    
                    rebuy_price_debt = self.get_price(symbol=symbol) * (1 + self.slippage)
                    rebuy_order_debt_id = self.execute_margin_order(symbols=symbols, size=owed_amount, order_type="limit", side="buy",price=rebuy_price_debt,increments=increments)

                repaid_order_id, repaid_amount = self.repay_funds(symbols=symbols, curr=curr, size=size_repay)
                self.logger.info(f"Repaid order of amount {repaid_amount}")
            
            return {"message": "All debts paid off"}
        
        except Exception as e:
            print(f"An error occurred when paying off all debts: {e}")
            return {"error": f"An exception occurred: {str(e)}"}


    def get_margin_borrowing_history(self, coin=None, isisolated=None, symbol=None, orderno=None):
        """"
        REQ:        coin        e.g. KCS
        OPT:        isisolated:     true-isolated, false-cross, default-cross
        OPT:        symbol:         trading-pair, mandatory for isolated margin
        OPT:        orderNo:        OrderNumber
        Link:       https://www.kucoin.com/docs/rest/margin-trading/margin-trading-v3-/get-margin-borrowing-history
        
        Returns:
        orderNo:	        Borrow order ID
        symbol: 	        Isolated margin trading pair; empty for cross margin
        currency:	        Currency
        size:   	        Initiated borrowing amount
        actualSize:         borrowed amount
        status:             Status
        createdTime:    	Time of borrowing
        """
        try:
            endpoint = f'/api/v3/margin/borrow'
            url = 'https://api.kucoin.com' + endpoint
            
            params = {"currency": coin}
            
            if symbol is not None:
                params["symbol"] = symbol

            if isisolated:
                params["isisolated"] = isisolated
            
            if orderno is not None:
                params["orderNo"] = orderno

            headers = self.create_sign_in(api_key=self.api_key, api_secret=self.api_secret, api_passphrase=self.api_passphrase, 
                                        method="GET",url=url, endpoint=endpoint, data="", params=params)

            response = requests.get(url, headers=headers, params=params)    
            response_json = response.json()

            # Check if response is successful
            if response.status_code == 200:
                return response_json["data"]
            else:
                return {"error": f"Failed to fetch data: {response.status_code}", "details": response_json}
        
        except Exception as e:
            print(f"An error occurred when retrieving borrowing history: {e}")
            return {"error": f"An exception occurred: {str(e)}"}
        
    def get_margin_repayment_history(self, coin=None, isisolated=None, symbol=None, orderno=None):
        """"
        REQ:        coin:       e.g.: KCS
        OPT:        isisolated:     true-isolated, false-cross, default-cross
        OPT:        symbol:         trading-pair, mandatory for isolated margin
        OPT:        orderNo:        OrderNumber
        Link:       https://www.kucoin.com/docs/rest/margin-trading/margin-trading-v3-/get-repayment-history
        
        Returns:
        orderNo:	        Borrow order ID
        symbol: 	        Isolated margin trading pair; empty for cross margin
        currency:	        Currency
        size:   	        Initiated borrowing amount
        principal:      	Principal to be paid
        interest:       	Interest to be paid
        status:         	Status Repaying, Completed, Failed
        createdTime:    	Time of repayment
        """
        try:
            endpoint = f'/api/v3/margin/repay'
            url = 'https://api.kucoin.com' + endpoint
            params = {"currency": coin}
            
            if symbol is not None:
                params["symbol"] = symbol

            if isisolated:
                params["isisolated"] = isisolated
            
            if orderno is not None:
                params["orderNo"] = orderno

            headers = self.create_sign_in(api_key=self.api_key, api_secret=self.api_secret, api_passphrase=self.api_passphrase, 
                                        method="GET",url=url, endpoint=endpoint, data="", params=params)

            response = requests.get(url, headers=headers, params=params)    
            response_json = response.json()

            # Check if response is successful
            if response.status_code == 200:
                #df = pd.DataFrame(response_json["data"])
                return response_json["data"]
            else:
                return {"error": f"Failed to fetch data: {response.status_code}", "details": response_json}
        
        except Exception as e:
            print(f"An error occurred when retrieving repayment history: {e}")
            return {"error": f"An exception occurred: {str(e)}"}

    def check_if_is_still_borrowing(self, coin=None, fiat=None, symbol=None):
        """
        checks borrowing and repayment history for order id of trading pair and identifies if it still borrowing.
        
        OPT:                fiat    e.g. USDT
        OPT:                coin    e.g. BTC
        OPT:                if isosolated: provide symbol

        Returns: 
        True, False:        True if still borrowing, False if not borrowing
        Amount:             Borrowed amount if still borrowing, else 0
        
        """

        if fiat is not None:
            borrow_histo_fiat = self.get_margin_borrowing_history(coin=fiat)
            repay_histo_fiat = self.get_margin_repayment_history(coin=fiat)
            
            if coin is None:
                borrow_histo_coin = None
                repay_histo_coin = None
            
            if borrow_histo_fiat["totalNum"] ==0:
                self.logger.info("No borrowing history found for fiat")
                notborrowing = True
            
        if coin is not None: 
            borrow_histo_coin = self.get_margin_borrowing_history(coin=coin)
            repay_histo_coin = self.get_margin_repayment_history(coin= coin)
            
            if fiat is None:
                borrow_histo_fiat = None
                repay_histo_fiat = None
            
            if borrow_histo_coin["totalNum"] ==0:
                self.logger.info("No borrowing history found for coin")
                notborrowing = True
        
        if fiat is not None and coin is not None:
            if notborrowing and borrow_histo_fiat["totalNum"] ==0 or notborrowing and borrow_histo_coin["totalNum"] ==0:
                return False
            
        else:
            
            borrowed_amount_fiat = 0
            repaid_amount_fiat = 0

            borrowed_amount_coin = 0 
            repaid_amount_coin = 0 

            if fiat is not None:
                for item in borrow_histo_fiat['items']:
                    if item["currency"] == fiat and item["status"]=="SUCCESS":
                        borrowed_amount_fiat += float(item["actualSize"])
                for item in repay_histo_fiat['items']:
                    if item["currency"] == fiat and item["status"]=="SUCCESS":
                        repaid_amount_fiat += float(item["principal"])
                
                outstanding_borrowed_amount_fiat = borrowed_amount_fiat - repaid_amount_fiat
                
            if coin is not None:
                for item in borrow_histo_coin["items"]:   
                    if item["currency"] == coin and item["status"]=="SUCCESS":
                        borrowed_amount_coin += float(item["actualSize"])    
                for item in repay_histo_coin["items"]:
                    if item["currency"] == coin and item["status"]=="SUCCESS":
                        repaid_amount_coin += float(item["principal"])

                outstanding_borrowed_amount_coin = borrowed_amount_coin - repaid_amount_coin

            if fiat is not None and coin is not None:
                if outstanding_borrowed_amount_fiat >0 and outstanding_borrowed_amount_coin<=0:
                    self.logger.info(f"Borrowed fiat amount is {outstanding_borrowed_amount_fiat}")
                    return outstanding_borrowed_amount_fiat
                elif outstanding_borrowed_amount_coin >0 and outstanding_borrowed_amount_fiat<=0:
                    self.logger.info(f"Borrowed coin amount is {outstanding_borrowed_amount_coin}")
                    return outstanding_borrowed_amount_coin
                elif outstanding_borrowed_amount_fiat>0 and outstanding_borrowed_amount_coin>0:
                    self.logger.info(f"Borrowed fiat amount is {outstanding_borrowed_amount_fiat}, borrowed coin amount is {outstanding_borrowed_amount_coin}")
                    return outstanding_borrowed_amount_fiat, outstanding_borrowed_amount_coin
            elif fiat is not None and coin is None:
                if outstanding_borrowed_amount_fiat >0:
                    self.logger.info(f"Borrowed fiat amount is {outstanding_borrowed_amount_fiat}")
                    return outstanding_borrowed_amount_fiat
                else:
                    return False
            elif coin is not None and fiat is None:
                if outstanding_borrowed_amount_coin >0:
                    self.logger.info(f"Borrowed coin amount is {outstanding_borrowed_amount_coin}")
                    return outstanding_borrowed_amount_coin
                else:
                    return False
            else:
                return False


    def get_paid_fees_per_symbol(self, currency="USDT", symbol=None, order_id=None):
        position = self.get_position_by_order_id(self.ledger_data["current_trades"][symbol]["order_id"])
        


#################################################################################################################################################################
#
#
#                                                                  POSITION Functions Internal
#
#################################################################################################################################################################

    def load_ledger_data(self, filepath):
        def restore_value(value):
            """Helper function to restore serialized JSON values to their original types."""
            if isinstance(value, dict) and "year" in value:
                # Heuristic to restore pandas Timestamp (depends on your data specifics)
                try:
                    return pd.to_datetime(value)
                except Exception:
                    return value
            elif isinstance(value, dict):
                # Heuristic to restore pd.Series or pd.DataFrame
                if all(isinstance(k, str) and isinstance(v, dict) for k, v in value.items()):
                    # Likely a DataFrame
                    return pd.DataFrame(value)
                else:
                    # Likely a Series
                    return pd.Series(value)
            else:
                return value

        def restore_dict(d):
            """Recursively restores all values in the dictionary to their original types."""
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = restore_dict(value)
                else:
                    d[key] = restore_value(value)
            return d

        if not os.path.exists(filepath):
            ledger = {
                "positions": [], 
                "balances": {
                    "initial_balance":{}, 
                    "trailing_balances":{}, 
                    "total_balance":0
                }, 
                "current_trades":{},   #self.symbol:None
                "order_details":{}
            }

            account_list_margin_hf = self.get_margin_account_details(accounts=True)
            account_list_margin_hf = account_list_margin_hf[account_list_margin_hf["available"] > 0][["currency","total","available"]]
            
            for crypto in account_list_margin_hf["currency"]: 
                ledger["balances"]["initial_balance"][crypto] = account_list_margin_hf[account_list_margin_hf["currency"] == crypto]["available"]
                ledger["balances"]["trailing_balances"][crypto] = account_list_margin_hf[account_list_margin_hf["currency"] == crypto]["available"]
                
                if crypto != self.fiat:
                    symbol = crypto + "-" + self.fiat
                    ledger["current_trades"][symbol] = None
            
            ledger["balances"]["trailing_balances"][self.fiat] = account_list_margin_hf[account_list_margin_hf["currency"] == self.fiat]["available"]
            ledger["balances"]["trailing_balances"]["total_balance"] = self.calculate_balance_in_fiat(
                fiat=self.fiat, 
                getSum=True, 
                get_only_trade_pair_bal=True, 
                account_type="margin"
            )
            
            return ledger 
        else:
            try:
                with open(filepath, 'r') as file:
                    ledger_json = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {filepath}: {e}")
                return None
            except IOError as e:
                print(f"Error loading data from {filepath}: {e}")
                return None
            return restore_dict(ledger_json)    #json.loads()
        
    def save_data(self, filepath):

        def convert_value(value):
            """Helper function to convert non-serializable objects to serializable formats."""
            if isinstance(value, pd.Timestamp):
                return value.isoformat()
            elif isinstance(value, pd.Series):
                return value.to_dict()
            elif isinstance(value, pd.DataFrame):
                return value.to_dict(orient='records')
            else:
                return value

        def convert_dict(d):
            """Recursively converts all values in the dictionary to JSON serializable formats."""
            for key, value in d.items():
                if isinstance(value, dict):
                    convert_dict(value)
                else:
                    d[key] = convert_value(value)
        
        ledger_converted = self.ledger_data.copy()
        
        # Convert Timestamp objects to strings
        for position in ledger_converted["positions"]:
            if 'timestamp_opened' in position and isinstance(position['timestamp_opened'], pd.Timestamp):
                position['timestamp_opened'] = position['timestamp_opened'].isoformat()
            if 'timestamp_closed' in position and isinstance(position['timestamp_closed'], pd.Timestamp):
                position['timestamp_closed'] = position['timestamp_closed'].isoformat()
        
        # Convert all entries in the ledger data to serializable formats
        convert_dict(ledger_converted)
        
        try:
            with open(filepath, 'w') as file:
                json.dump(ledger_converted, file, indent=4)
        except IOError as e:
            print(f"Error saving data to {filepath}: {e}")
        
        g_drive_filepath = "Trading/trading_ledger.json"
        
        # Convert the ledger data to JSON string
        ledger_json_g_drive = json.dumps(ledger_converted, indent=4)
        
        # Try saving the file to g drive for upload purposes
        try:
            self.gutils.save_file(ledger_json_g_drive, g_drive_filepath)
        except Exception as e:
            print(f"Error saving data to {filepath}: {e}")
            return False
        
        
        

    def load_trading_metrics(self, metrics_file_path=None):
        
        
        if not os.path.exists(metrics_file_path):
            trading_metrics_data = {"losing_trades": 0 , "winning_trades": 0, "largest_loss": 0, "largest_gain": 0}
            
            with open(metrics_file_path, "w") as file:
                    json.dump(trading_metrics_data, file, indent=4)

            return trading_metrics_data
        with open(metrics_file_path, "r") as file:
            trading_metrics_data = json.load(file)
            return trading_metrics_data
        
    def save_trading_metrics(self, metrics_file_path=None, trading_metrics_data=None):
        with open(metrics_file_path, "w") as file:
            json.dump(trading_metrics_data, file, indent=4)

    def update_position(self, position_info):
        """"
        REQ:    position_info:      dict containing the info for each trade 

        Updates the below position information:
        
        'order_id':                     initial_order_id,
        'size':                         size
        'funds':                        funds
        'side':                         side
        'pair':                         symbol
        'execution_price':                execution_price
        'status':                       "open" / "closed"
        'timestamp_opened':             datetime.now().timestamp()*1000
        'timestamp_closed':             datetime.now().timestamp()*1000
        'time_opened':                  date str opened,
        'time_closed':                  date str closed,
        'invested_usdt':                invested_amount
        'initial_balance' :             initial_balance,
        'closing_amount_invested':      closing amount invested
        'closing_balance':              balance after position close
        'stop_loss_order_id':           stop_loss_order_id if stop_loss_order else None
        'take_profit_order_id':         take_profit_order_id if take_profit_order else None
        'rebuy_order_id':               When short and pnl is negative we need to rebuy to cover debt,
        'convert_order_id':             When short we need to convert to usdt,
        "borrowSize" :                  borrowSize
        "repaid" :                      "open_borrowing"
        "close_order_id" :              None,
        "fees_opening_trade":           fees paid for starting trade,
        "fees_closing_trade":           fees paid for ending trade, 
        "pnl":                          None,
        "owed_tax:                      None
        """
        # Assuming position_info contains a unique identifier like 'trade_id'
        existing_positions = [pos for pos in self.ledger_data["positions"] if pos.get('order_id') == position_info.get('order_id')]
        
        if existing_positions:
            # Update existing position
            index = self.ledger_data["positions"].index(existing_positions[0])
            self.ledger_data["positions"][index] = position_info
        else:
            # Add new position
            self.ledger_data["positions"].append(position_info)

        self.save_data(self.ledger_path)


    def get_position_by_order_id(self, order_id):
        for position in self.ledger_data.get("positions", []):
            if position.get('order_id') == order_id:
                return position
        return None
    
    def delete_position_and_orders(self, order_id):
        # Deleting the position associated with the given order_id
        self.ledger_data["positions"] = [position for position in self.ledger_data.get("positions", []) if position.get('order_id') != order_id]

        # Also, delete any stop loss or take profit orders associated with this position
        # Assuming these are tracked separately in the ledger
        # If they are part of the position data, they will already be deleted in the above line
        self.ledger_data["orders"] = [order for order in self.ledger_data.get("orders", []) if order.get('associated_trade_id') != order_id]

        self.save_data(self.ledger_path)
    
    def calculate_current_positions_value(self, symbol):
        total_positions_value = 0
        for position in self.ledger_data.get("positions", []):
            if position.get('status') == 'open':  # Check if the position is still open
                current_price = self.get_price(symbol)
                size = position['size']
                total_positions_value += current_price * size
        return total_positions_value

#################################################################################################################################################################
#
#
#                                                                  Logger
#
#################################################################################################################################################################


    def configure_logger(self):
        
        logger_path = utils.find_logging_path()

        #logger
        current_datetime = dt.datetime.now()
        timestamp = current_datetime.strftime('%d-%m-%Y_%H-%M')
        log_directory = "Kucoin Trader log"
        log_file_name = f"Kucoin_trader_log_{timestamp}.txt"
        log_file_path = os.path.join(logger_path, log_directory, log_file_name)

        if not os.path.exists(os.path.join(logger_path, log_directory)):
            os.makedirs(os.path.join(logger_path, log_directory))

        self.logger.add(log_file_path, rotation="500 MB", level="INFO")


#################################################################################################################################################################
#
#
#                                                                  BACKTEST
#
#################################################################################################################################################################

exchange = 'kucoin'
coin = "BTC"
fiat = "USDT"
slippage = 0.01
leverage = 1
symbol = coin + "-"+ fiat

trader = MainApiTrader(exchange, coin, fiat,slippage, leverage)

# margin_details = trader.get_margin_account_details()
# current_price = trader.get_ohlc(symbol=symbol, timeframe="1d")

# take_profit_percentage = 0.04
# stop_loss_percentage = 0.02
# stop_loss_price = current_price*(1-stop_loss_percentage)
# take_profit_price = current_price*(1+take_profit_percentage)

# symbol_info = trader.get_symbol_list(symbol=symbol)
# minSize = trader.symbol_info["baseMinSize"].values[0]
# minFunds = trader.symbol_info["quoteMinSize"].values[0]
# coinIncrement = trader.get_rounding_precision(symbol_info["baseIncrement"].values[0])
# fiatIncrement= trader.get_rounding_precision(symbol_info["quoteIncrement"].values[0])
# increments = trader.get_increments(symbol)
# mins = trader.get_min_sizes(symbol)

# side = "buy"
# funds = (margin_details[margin_details["currency"]==fiat]["available"].values[0]*0.8) #trader.round_down(margin_details[margin_details["currency"]==fiat]["available"].values[0],5)
# funds = trader.round_down(funds, fiatIncrement)

# fee_rate=trader.get_trading_fees("0")[0]
# size = ((funds/current_price )*(1-fee_rate))*(1-stop_loss_percentage) #stop_loss_price)
# size = trader.round_down(size, coinIncrement)
# stopPrice= 
# price = 24.95


#works
# account_list = trader.get_account_list(curr = coin)
# account_list = trader.get_account_list(curr = coin, account_type="margin")
# test_limit_order = trader.execute_margin_order(symbol=symbol, size=0.5, order_type="limit", side=side,price=14.39)
# trader.execute_auto_borrow_margin_order(funds=funds, side=side)
# trader.inner_transfer(currency="USDT", from_acc="trade", to_acc="margin", amount="10")
# acc_list = trader.get_account_list()  #curr="BTC", type="MARGIN"  type="margin"
# margin_account_list = trader.get_margin_account_details()
# margin_avax_balance = trader.get_margin_account_details(curr="AVAX")

#does not work
# trader.inner_flex_transfer(currency="USDT",  amount=1, from_acc="MARGIN", to_acc="TRADE")
# trader.inner_flex_transfer(currency="USDT",  amount=2, from_acc="MARGIN", to_acc="MAIN")
# acc_details = trader.get_account_balance("61411ad12455d9000628a080")

#################################################################################################################################################################
#                                     PRICE FUNCTIONS AND BALANCE IN FIAT
#################################################################################################################################################################

#works
# usd_price = trader.get_fiat_price(base="EUR", currencies="USDT")
# usdt_price =  trader.get_price("AVAX-USDT")
#eur_price = trader.get_price("BTC-EUR")
# calc_fiat = trader.calculate_balance_in_fiat(fiat="USDT", getSum=True)
# calc_usdt_sum = trader.calculate_balance_in_fiat(coin=coin, fiat="USDT", getSum=True, account_type="margin")
# calc_usdt_individual = trader.calculate_balance_in_fiat(coin=coin, fiat="USDT", account_type="margin", get_only_trade_pair_bal=True)

#################################################################################################################################################################
#                                     BORROWING AND REPAYMENT HISTORY
#################################################################################################################################################################
# date_string = "2024-04-04 00:00"
# date_ob = dt.datetime(2024,3,12,0,0)

#works
# margin_borrow_histo = trader.get_margin_borrowing_history("USDT")
#margin_repay_histo = trader.get_margin_repayment_history("USDT")
# check = trader.check_if_is_still_borrowing("AVAX", "USDT")
# margin_details = trader.get_margin_account_details()
# taker_fee, maker_fee = trader.get_trading_fees(0)
# act_taker_fee, act_maker_fee= trader.get_actual_trading_fees(symbol)
# margin_details_fiat = trader.get_margin_account_details(curr="USDT")["available"]
# margin_details_coin = trader.get_margin_account_details(curr=coin)["available"]
# get_account_details = trader.get_account_list(account_type="margin")
# filled_list = trader.get_filled_order_list()

# timestamp_1 = trader.convert_datetime_to_timestamp(date_string)
# timestamp_2 = trader.convert_datetime_to_timestamp(date_ob)
# date = trader.convert_timestamp_to_datetime(datetime.now().timestamp()*1000)


#does not work
# test_order_id, test_order_borrow_size = trader.execute_auto_borrow_margin_order_test(symbol=symbol, funds=funds, side="buy")
# test_order_details_hf = trader.get_hf_order_details_by_orderId(order_id="65f59247b7253400073ff561", symbol=symbol)

#works
# all_symbols = trader.get_symbol_list(symbol=symbol, specific_param=["symbol","baseMinSize","quoteMinSize"])

# trader.cancel_all_orders()
# limit_buy_price = current_price
# limit_buy_order_id, limit_borrow_size = trader.execute_auto_borrow_margin_order(size=size,price=limit_buy_price, side=side, order_type="limit", use_leverage=False)

#stop limit order
# stop_buy_id = trader.execute_stop_order(size=size, order_type="limit", side="buy", stopPrice=stopPrice, stop="entry", price=price)

#stop market order
# stop_buy_id = trader.execute_stop_order(funds=funds, order_type="market", side="buy", stopPrice=stopPrice, stop="loss")

# al_order_details = trader.get_order_details_list()
# stop_buy_id = 'vs8hopgiha2qbnj4003upuu3'
# print(stop_buy_id) 
# order_details_margin = trader.get_order_details_by_id(stop_buy_id)
# order_details_stop_margin = trader.get_order_details_by_id(stop_buy_id, is_stop=True)
# order_details_margin = trader.get_order_details_by_id(stop_buy_id)

# order_details_margin_active = trader.get_order_details_by_id(active=True)
# stop_buy_details = trader.get_stop_order_details(stop_buy_id)
# check = trader.check_order_triggered(stop_buy_id)
# cancel_stop = trader.cancel_order_by_order_id(stop_buy_id)
# check2 = trader.check_order_triggered(stop_buy_id)
# trader.cancel_stop_order()


###################################    BACKTEST    ########################################

# exchange = 'kucoin'
# coin = "AVAX"
# fiat = "USDT"
# slippage = 0.01
# leverage = 1
# symbol = coin + "-"+ fiat

# trader = MainApiTrader(exchange, coin, fiat,slippage, leverage)

# margin_details = trader.get_margin_account_details()
# current_price = trader.get_price(symbol=symbol)

# take_profit_percentage = 0.04
# stop_loss_percentage = 0.02
# stop_loss_price = current_price*(1-stop_loss_percentage)
# take_profit_price = current_price*(1+take_profit_percentage)
# investment_per_trade = 0.9

# symbol_info = trader.get_symbol_list(symbol=symbol)
# minSize = trader.symbol_info["baseMinSize"].values[0]
# minFunds = trader.symbol_info["quoteMinSize"].values[0]
# coinIncrement = trader.get_rounding_precision(symbol_info["baseIncrement"].values[0])
# fiatIncrement= trader.get_rounding_precision(symbol_info["quoteIncrement"].values[0])


# side = "buy"
# funds = (margin_details[margin_details["currency"]==fiat]["available"].values[0]*investment_per_trade) #trader.round_down(margin_details[margin_details["currency"]==fiat]["available"].values[0],5)
# funds = trader.round_down(funds, fiatIncrement)

# fee_rate=trader.get_trading_fees("0")[0]
# size = ((funds/current_price )*(1-fee_rate))*(1-stop_loss_percentage) #stop_loss_price)
# size = trader.round_down(size, coinIncrement)



############################################## test go long ########################################
# trader.cancel_all_orders()
# trader.cancel_all_orders(is_stop=True)
# trader.modify_leverage_multiplier(symbol=symbol, leverage_factor=leverage)
# use_leverage=(True if leverage>1 else False)
# test_auto_borrow_order_id, borrowSize = trader.execute_auto_borrow_margin_order(funds=funds, side="buy",use_leverage=use_leverage)

# order_details = trader.get_order_details_by_id(orderId=test_auto_borrow_order_id)
# trade_size = order_details["dealSize"][0]
# trade_funds = order_details["dealFunds"][0]
# margin_details = trader.get_margin_account_details(curr=coin)["available"].values[0]
# price = trader.get_price(symbol)

# stop_loss_price = (1-stop_loss_percentage) * price
# take_profit_price = (1+take_profit_percentage) * price

# stop_loss_order_id = trader.execute_stop_order(size=trade_size, order_type="market", side="sell", stopPrice=stop_loss_price, stop="loss", auto_repay=True)
# take_profit_order_id = trader.execute_stop_order(size=trade_size, order_type="market", side="sell", stopPrice=take_profit_price, stop="entry",auto_repay=True)

# test_auto_repay_order_id, borrowSize = trader.execute_auto_repay_margin_order(size=trade_size, side="sell", order_type="limit", price=(price-0.3))
# trader.cancel_all_orders()
# trader.cancel_all_orders(is_stop=True)

#####################################      #test go short    ###################################
# trader.cancel_all_orders(is_stop=True)
# trader.cancel_all_orders()
# # price = trader.get_price(symbol)

# use_leverage=(True if leverage>1 else False)
# trader.modify_leverage_multiplier(symbol=symbol, leverage_factor=2)
# test_auto_borrow_order_id, borrowSize = trader.execute_auto_borrow_margin_order(size=size, side="sell",use_leverage=use_leverage)

# order_details = trader.get_order_details_by_id(orderId=test_auto_borrow_order_id)
# trade_size = order_details["dealSize"]
# trade_funds = order_details["dealFunds"]
# margin_details = trader.get_margin_account_details(curr=coin)["available"].values[0]

# price = trader.get_price(symbol)
# stop_loss_price = (1+stop_loss_percentage) * price
# take_profit_price = (1-take_profit_percentage) * price

# stop_loss_order_id = trader.execute_stop_order(size=trade_size, order_type="limit", side="buy", stopPrice=stop_loss_price, price=(take_profit_price*(1+slippage)), stop="entry",auto_repay=True)
# take_profit_order_id = trader.execute_stop_order(size=trade_size, order_type="limit", side="buy", stopPrice=take_profit_price, price=(take_profit_price*(1+slippage)),stop="loss",auto_repay=True)

# price = trader.get_price(symbol)
# test_auto_repay_order_id, borrowSize = trader.execute_auto_repay_margin_order(size=trade_size, side="buy", order_type="limit", price=(price*(1+slippage)))
# trader.cancel_all_orders(is_stop=True)
# trader.cancel_all_orders()



# debt = trader.get_margin_account_details() #curr="PYTH"
# debt = debt[debt.liability>0]
# debt.set_index("currency", inplace=True)
# liab = debt.loc["PYTH", "liability"]
# trader.repay_funds(curr="PYTH",size=liab)
# test_order_details = trader.get_order_details_by_id(orderId='65f5c6394a3a69000761f39d')



# test_if_is_still_borrowing = trader.check_if_is_still_borrowing(currency="AVAX")

# filled_orders = trader.get_filled_hf_orders_list("BTC-USDT", trade_type="MARGIN_TRADE")
# test = trader.get_account_list(curr="QNT",type="trade")["available"]


# test_order = trader.execute_auto_borrow_hf_margin_order_test(symbol, size=size, order_type="market", side=side)
print("API Class successfully initialized")