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
from concurrent.futures import ProcessPoolExecutor
import random
import string

base_path = os.path.dirname(os.path.realpath(__file__))
crypto_bot_path = os.path.dirname(base_path)
Python_path = os.path.dirname(crypto_bot_path)
Trading_bot_path = os.path.dirname(Python_path)
Trading_path = os.path.join(Trading_bot_path, "Trading")

# Data Paths
data_path_crypto = os.path.join(Trading_bot_path, "Data", "Cryptocurrencies")
datasets_path = os.path.join(data_path_crypto, "Datasets")
csv_dataset_path = os.path.join(datasets_path, "crypto datasets", "csv")
hdf_dataset_path = os.path.join(datasets_path, "crypto datasets", "hdf5")
hist_data_download_path = os.path.join(crypto_bot_path, "Hist Data Download")
san_api_data_path = os.path.join(hist_data_download_path, "SanApi Data")
main_data_files_path = os.path.join(san_api_data_path, "Main data files")

# Strategy and Trading API Paths
strategy_path = os.path.join(crypto_bot_path, "Trading Strategies")
trade_api_path = os.path.join(crypto_bot_path, "API Trader")
backtest_path = os.path.join(crypto_bot_path, "Backtesting")
kucoin_api = os.path.join(crypto_bot_path, "Kucoin API")

# Config and Utility Paths
config_path = os.path.join(crypto_bot_path, "Config")
utils_path = os.path.join(Python_path, "Tools")
logging_path = os.path.join(Trading_bot_path, "Logging")
data_loader = os.path.join(crypto_bot_path, "Data Loader")

sys.path.append(crypto_bot_path)
sys.path.append(trade_api_path)
sys.path.append(backtest_path)
sys.path.append(utils_path)
sys.path.append(Trading_path)
sys.path.append(config_path)
sys.path.append(logging_path)
sys.path.append(data_path_crypto)
sys.path.append(kucoin_api)
sys.path.append(datasets_path)
sys.path.append(csv_dataset_path)
sys.path.append(hdf_dataset_path)
sys.path.append(data_loader)
sys.path.append(main_data_files_path)
sys.path.append(strategy_path)

import mo_utils as utils
import KuCoin_websocket as websocket

#################################################################################################################################################################
#
#
#                                                                  KUCOIN BACKTEST CLASS
#
#################################################################################################################################################################



class Backtester():
    def __init__(self, coin, currency, slippage, leverage, logger_input=None):
        """
        Initialisiert den Trader mit dem gewünschten Handelsmodus: 'spot' oder 'margin'.
        """

        if logger is None:
            self.logger = logger
            self.configure_logger()
        else:
            self.logger = logger_input

        config_path = utils.find_config_path()
        self.api_config = utils.read_config_file(os.path.join(config_path,"kucoin_backtest_config.ini"))

        trading_path = utils.find_trading_path()
        self.ledger_path = os.path.join(trading_path, "trading_ledger_backtest.json")
        self.ledger_data = self.load_ledger_data(self.ledger_path)

        self.balances = self.ledger_data["balances"]    #pd.DataFrame.from_dict({currency: 0, coin: })
        self.slippage = slippage
        self.leverage = leverage

        # self.coin = coin
        self.fiat = currency

         #retrieve all assets and create mapping with san coin names 
        self.all_assets = pd.read_excel(os.path.join(main_data_files_path, "all_assets.xlsx"), header=0)
        self.ticker_to_slug_mapping = dict(zip(self.all_assets['ticker'], self.all_assets['slug']))

        # trading metrics
        self.trading_metrics_path = os.path.join(Trading_path,"trading_metrics_AlgoTrader.json") 
        self.trading_metrics_data = self.load_trading_metrics(self.trading_metrics_path)

        self.losing_trades = self.trading_metrics_data["losing_trades"]
        self.winning_trades = self.trading_metrics_data["winning_trades"]
        self.largest_loss =  self.trading_metrics_data["largest_loss"]
        self.largest_gain = self.trading_metrics_data["largest_gain"]
        self.total_trades = self.trading_metrics_data["total_trades"]
        
    
        self.client_oid = utils.get_config_value(self.api_config,"overview","client_user_id") #self.generate_random_string(10)
                               # still needs to be adjusted, variable is passed from AlgoTraderBacktest
        self.time_sleep_factor_long = 0
        self.time_sleep_factor_short = 0
         
        self.logger.info(f"{'#'*1}   Backtester initialized for {coin}-{currency}   {'#'*1}")
################################################################################################################################################################
#
#
#                                                                  TRADING FUNCTIONS
#
#################################################################################################################################################################


    def enter_margin_trade_backtest(self, data=None, datetime_input=None, coin=None, fiat=None, size=None, funds=None, balance_fiat=None, is_long=None, order_type=None, limit_price=None, stop_price=None, take_profit_price=None):
        
        try:
            #symbol
            symbol = coin + "-" + fiat

            # Create the initial margin order
            if is_long is not None:
                side = 'buy' if is_long else 'sell'
            else:
                self.logger("No side specificed aborting trade")
                return
        
            initial_order_id, borrowSize, order_details = self.execute_auto_borrow_margin_order_backtest(data=data, datetime_input=datetime_input, coin=coin, fiat=fiat, size=size, funds=funds, balance_fiat=balance_fiat, order_type=order_type, side=side) #removed price as no limit order strategy

            price = self.get_price(coin=coin,data=data, datetime_input=datetime_input, activate_slippage=False)
            stop_loss_order_id = self.generate_random_string(24)
            take_profit_order_id = self.generate_random_string(24)
            timestamp  = self.convert_datetime_to_timestamp(datetime_input)
            fees = order_details["fee"] 
            current_price = order_details["price"]

            if side == "buy":
                sl_size = order_details["dealSize"]
                tp_size = order_details["dealSize"]

                sl_funds = None
                tp_funds = None 

                if order_details["dealSize"] == None:
                    raise ("size is none please check and adjust invested amount calculation")     
            else:
                sl_size = order_details["dealSize"] 
                tp_size = order_details["dealSize"]
                
                sl_funds = None
                tp_funds = None 

                # sl_funds = order_details["dealFunds"]   # + ((order_details["dealFunds"]+order_details["fee"]) * round((1-(stop_price/order_details["price"])),4))          #*(1-self.get_trading_fees(coin=coin, side="sell"))
                # tp_funds = order_details["dealFunds"]   # + ((order_details["dealFunds"] + order_details["fee"]) * round((1-(take_profit_price/order_details["price"])),4)) #(1-self.get_trading_fees(coin=coin, side="sell")) 
                
                # if order_details["dealFunds"] == None:
                #     raise ("funds is none please check and adjust invested amount calculation")

                if order_details["dealSize"] == None:
                    raise ("size is none please check and adjust invested amount calculation")

            # Create a stop loss order if stop_price is specified
            stop_loss_order = None
            take_profit_order = None

            if stop_price:
                stop_loss_side = 'sell' if is_long else 'buy'
                stop_loss_order_id, borrowSize_sl, sl_order_details = self.execute_auto_repay_margin_order_backtest(data=data, datetime_input=datetime_input, coin=coin, fiat=fiat, size=sl_size, funds=sl_funds, order_type="limit", side=stop_loss_side, price=stop_price)
                stop_loss_order = True

            # Create a take-profit order if specified
            if take_profit_price:
                take_profit_side = 'sell' if is_long else 'buy'
                take_profit_order_id, borrowSize_tp, tp_order_details = self.execute_auto_repay_margin_order_backtest(data=data, datetime_input=datetime_input, coin=coin, fiat=fiat,  size=tp_size, funds=tp_funds,order_type="limit", side=take_profit_side, price=take_profit_price)
                take_profit_order = True

            
            # Add order information to the ledger
            position_info = {
                'order_id':                     initial_order_id,
                'size':                         order_details["dealSize"],
                'funds':                        order_details["dealFunds"],
                'side':                         side,
                'pair':                         symbol,
                'current_price':                float(current_price),
                'status':                       'open',
                'timestamp_opened':             str(order_details["createdAt"]),
                'timestamp_closed':             None,
                'time_opened':                  str(order_details["createdAt"]),
                'time_closed':                  None,
                'invested_usdt':                (order_details["dealFunds"]+fees),
                'initial_balance' :             self.balances["trailing_balances"][fiat],
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
                "owed_tax":                     None}

            self.ledger_data["current_trades"][symbol] = {"order_id":initial_order_id, "stop_loss_order_id":stop_loss_order_id, "take_profit_order_id":take_profit_order_id}
            
            if side == "buy":
                balance_fiat = self.balances["trailing_balances"][fiat] -  order_details["funds"]
                balance_coin = self.balances["trailing_balances"][coin] + order_details["dealSize"]
            else:
                balance_fiat = order_details["dealFunds"]  #self.get_margin_account_details(curr=fiat)["available"].values[0]
                balance_coin = self.balances["trailing_balances"][coin] 

            self.balances["trailing_balances"][fiat] = balance_fiat
            self.balances["trailing_balances"][coin] = balance_coin
            self.balances["trailing_balances"]["total_balance"] = self.calculate_balance_in_fiat(data=data, datetime_input=datetime_input, fiat=fiat, coin=coin, getSum=True, get_only_trade_pair_bal=True)
            
            self.ledger_data["order_details"][initial_order_id] = {}
            self.ledger_data["order_details"][initial_order_id]["initial_order_details"] = order_details
            self.ledger_data["order_details"][initial_order_id]["sl_order_details"] = sl_order_details
            self.ledger_data["order_details"][initial_order_id]["tp_order_details"] = tp_order_details

            self.update_position(position_info)

            self.trading_metrics_data['total_trades'] += 1

            self.save_trading_metrics(self.trading_metrics_path, self.trading_metrics_data)

            position = self.get_position_by_order_id(initial_order_id)
            
            
            # self.backtest_performance = pd.concat([self.backtest_performance,pd.DataFrame([self.balances["trailing_balances"]], index=[datetime_input])], axis=0, ignore_index=True)
            # self.backtest_positions = pd.concat([self.backtest_positions, pd.DataFrame(position, index=[datetime_input])], axis=0, ignore_index=True)
            self.backtest_performance = self.backtest_performance.combine_first(pd.DataFrame([self.balances["trailing_balances"]], index=[datetime_input]))
            self.backtest_positions = self.backtest_positions.combine_first(pd.DataFrame(position, index=[datetime_input]))
            
            self.save_backtest_performance_file(data="performance", file=self.backtest_performance)
            self.save_backtest_performance_file(data="positions", file=self.backtest_positions)

            time.sleep(self.time_sleep_factor_long)
            return order_details, sl_order_details, tp_order_details
            
        except Exception as e:
            self.logger.error(f"An error occurred when entering the trade: {e}")
            return 
        
    
    def close_margin_position_backtest(self,data=None, datetime_input=None, coin=None,fiat=None, current_position=None):
        """
        closes margin orders by order id
        """
        try:
            fiat = fiat if fiat is not None else self.fiat 
            position = current_position #self.get_position_by_order_id(order_id)
            
            if not position:
                raise Exception("Position not found")

            symbol = position['pair']
            size = position['size']
            funds = position['funds']
            is_long = position['side'] == 'buy'
            timestamp  = self.convert_datetime_to_timestamp(datetime_input)
            borrowAmount = position["borrowSize"]
            trailing_balance_coin =  self.balances["trailing_balances"][coin] 
            trailing_balance_fiat =  self.balances["trailing_balances"][fiat] 
            initial_order_id = position["order_id"] 

            sl_order_details = self.ledger_data["order_details"][initial_order_id]["sl_order_details"]
            tp_order_details = self.ledger_data["order_details"][initial_order_id]["tp_order_details"]

            # Determine the side of the order to close the position
            side = 'sell' if is_long else 'buy'
            
            if side == "sell":
                borrowed_asset = fiat
                borrowed_amount = funds
                closing_size = size
                closing_funds = None
            else:
                borrowed_asset = coin
                borrowed_amount = size   #might be used instead of BorrowAmount returned from initial execute auto borrow order
                closing_size = None
                closing_funds = funds

            risk_prevention_status = self.check_sl_or_tp_triggered_backtest(data=data, datetime_input=datetime_input, coin=coin, fiat=fiat, position=position)        #still needs to be coded 

            if risk_prevention_status =="not triggered":
                # Place an order to close the position, the take profit order and the stop loss order
                close_order_id, borrowSize, close_order_details  = self.execute_auto_repay_margin_order_backtest(data=data, datetime_input=datetime_input, coin=coin, fiat=fiat, size=closing_size, funds=closing_funds, order_type="market", side=side)
                self.logger.info(f"Closed position with id: {position['order_id']}")
                self.logger.info(f"Cancelled stop loss order with id: {position.get('stop_loss_order_id')}")
                self.logger.info(f"Cancelled take profit order with id: {position.get('stop_loss_order_id')}")
                
                time.sleep(self.time_sleep_factor_short)
                close_order_details = close_order_details  
                
                if side == "sell":
                    closing_amount = close_order_details["dealFunds"] 
                    pnl_trade = closing_amount - position['invested_usdt']

                    self.balances["trailing_balances"][fiat]  += close_order_details["dealFunds"]
                    self.balances["trailing_balances"][coin]  = self.balances["trailing_balances"][coin] - closing_size

                elif side =="buy":
                    closing_amount = close_order_details["size"]
                    pnl_trade = closing_amount - size

                    if np.round(pnl_trade,8) < 0:
                        rebuy_price = self.get_price(coin=coin, data=data, datetime_input=datetime_input) * (1+ self.slippage)
                        rebuy_order_id, rebuy_order_details = self.execute_margin_order_backtest(data=data, datetime_input=datetime_input, coin=coin, fiat=fiat, funds=None, size=abs(pnl_trade), order_type="limit", side=side, price=rebuy_price)
                        position["rebuy_order_id"] = rebuy_order_id
                        self.balances["trailing_balances"][coin] = pnl_trade + rebuy_order_details["size"]  #initial balance in coin is not considered as we borrow funds for going short  - close_order_details["dealSize"]
                        self.balances["trailing_balances"][fiat] = self.balances["trailing_balances"][fiat] + (self.balances["trailing_balances"][fiat] - close_order_details["funds"]) - rebuy_order_details["dealFunds"] - close_order_details["fee"]  #- E.g. 1000 - 988 (as being short this would be profit) - 0.5 (fee) - 0.5 (fee)

                        usdt_convert_order_details = {"fee":0}
                        usdt_convert_order_details["dealFunds"] = 0
                        usdt_convert_order_details["funds"] = 0
                    
                    elif np.round(pnl_trade,8) == 0:
                        rebuy_order_details = {"fee":0}
                        rebuy_order_details["dealFunds"] = 0
                        rebuy_order_details["funds"] = 0

                        usdt_convert_order_details = {"fee":0}
                        usdt_convert_order_details["dealFunds"] = 0
                        usdt_convert_order_details["funds"] = 0

                        self.balances["trailing_balances"][coin] = pnl_trade  
                        self.balances["trailing_balances"][fiat] = self.balances["trailing_balances"][fiat] + (self.balances["trailing_balances"][fiat] - close_order_details["funds"])  - close_order_details["fee"] 
                    
                    else:
                        # initial balance in coin - close_order_details["dealSize"]
                        rebuy_order_details = {"fee":0}
                        rebuy_order_details["dealFunds"] = 0
                        rebuy_order_details["funds"] = 0

                        self.balances["trailing_balances"][coin] = pnl_trade  

                        #when short convert coin into usdt
                        coin_balance_after_trade = self.balances["trailing_balances"][coin]
                        usdt_convert_order_id, usdt_convert_order_details = self.execute_margin_order_backtest(data=data, datetime_input=datetime_input, coin=coin, fiat=fiat, size=coin_balance_after_trade, order_type="market", side="sell")
                        position["convert_order_id"] = usdt_convert_order_id
                        self.balances["trailing_balances"][fiat] =  self.balances["trailing_balances"][fiat] + (self.balances["trailing_balances"][fiat] - close_order_details["funds"]) + usdt_convert_order_details["dealFunds"] - close_order_details["fee"]  #+ usdt_convert_order_details["dealFunds"] - (close_order_details["fee"]) 
                        self.balances["trailing_balances"][coin] = pnl_trade - usdt_convert_order_details["size"]

                # Update the position as closed in the ledger
                position["close_order_id"] = close_order_id
                position['status'] = 'closed'
                position['timestamp_closed'] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                position['time_closed'] =      dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                position["repaid"] = "repaid"
                
                if side =="buy" and pnl_trade <0:
                    fees = { "sum":     sum([close_order_details["fee"], rebuy_order_details["fee"], usdt_convert_order_details["fee"]]), 
                            "single":   [close_order_details["fee"], rebuy_order_details["fee"], usdt_convert_order_details["fee"]]}
                elif side =="buy" and pnl_trade >=0:
                    fees = { "sum":     sum([close_order_details["fee"], usdt_convert_order_details["fee"]]), 
                            "single":   [close_order_details["fee"], usdt_convert_order_details["fee"]]}                
                else:
                    fees = close_order_details["fee"]

                position["fees_closing_trade"] = fees["sum"] if isinstance(fees, dict) else fees
                position["closing_balance"] = self.balances["trailing_balances"][fiat]
                
                if side == "sell":
                    closing_amount_invested = close_order_details["dealFunds"]
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = (closing_amount_invested - position['invested_usdt'])    #position['closing_balance']
                else:
                    time.sleep(self.time_sleep_factor_short)
                    closing_amount_invested = close_order_details["funds"] - usdt_convert_order_details["dealFunds"] + rebuy_order_details["funds"]    # if profit trade as short then   e.g. 980 - 5 + 0.5 = 975.5  which is then subtracted from initial invested amt 1000 - 975.5 = 24.5
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = (position['funds'] - (closing_amount_invested+fees["sum"]))    #add fees to subtract invested amt
   
                if position["pnl"] > 0:
                    self.logger.info(f"{'#'*1} Position closed with profit of {position['pnl']} {fiat}  {'#'*1}")
                    position["owed_tax"] = position["pnl"] * 0.25
                    self.trading_metrics_data["winning_trades"] += 1

                    if position["pnl"] > self.trading_metrics_data["largest_gain"]:
                        self.trading_metrics_data["largest_gain"] = position["pnl"]

                elif position["pnl"] < 0:
                    self.logger.info(f"{'#'*1}   Position closed with loss of {position['pnl']} {fiat}    {'#'*1}")
                    position["owed_tax"] = position["pnl"]*0.25
                    self.trading_metrics_data["losing_trades"] += 1

                    if position["pnl"] < self.trading_metrics_data["largest_loss"]:
                        self.trading_metrics_data["largest_loss"] = position["pnl"]
                
                self.save_trading_metrics(self.trading_metrics_path, self.trading_metrics_data)   

                self.ledger_data["current_trades"][symbol] = None
                self.ledger_data["balances"]["trailing_balances"]["total_balance"] = self.calculate_balance_in_fiat(data=data, datetime_input=datetime_input, fiat=fiat, coin=coin, getSum=True, get_only_trade_pair_bal=True)
                self.logger.info(f"{'#'*1} Current total balance in USDT is {self.ledger_data['balances']['trailing_balances']['total_balance']} {'#'*1}")

                if side == "buy":
                    self.ledger_data["order_details"][initial_order_id]["rebuy_order_details"] = rebuy_order_details
                    self.ledger_data["order_details"][initial_order_id]["usdt_convert_order_details"] = usdt_convert_order_details

                self.update_position(position)
                time.sleep(self.time_sleep_factor_long)
                return 
            else:
                if risk_prevention_status == "tp_cancel":
                    closed_by = "stop loss"
                else:
                    closed_by = "take profit"
                logger.info(f"Position already closed by {closed_by}")
                return
        
        except Exception as e:
            self.logger.error(f"An error occurred when closing the position: {e}")
            return None
        


#################################################################################################################################################################
#
#
#                                                                       Margin Order Functions (V1)
#
#################################################################################################################################################################

    def execute_auto_borrow_margin_order_backtest(self, data=None, datetime_input=None, coin=None, fiat=None, size=None, funds=None, balance_fiat=None, order_type="market", side=None, price=None):
        """
        ONLY FOR STANDARD MARGIN TRADING NOT HIGH-FREQUENCY MARGIN_V2 !!!!!!!!!

        REQ:    clientOid:      String
        REQ:    side:           "buy" or "sell"
        REQ:    symbol:         e.g. BTC-USDT
        REQ:    type:           "limit" or "market"

        OPT:    stp:            self-trade-prevention  (CN, CO, CB, DC)
        OPT:    isIsolated:     TRUE-isolated margin, FALSE -cross margin   def: false
        OPT:    autoBorrow:     True or False
        OPT:    autoRepay:      True or False
        Link:           https://www.kucoin.com/docs/rest/margin-trading/orders/place-margin-order

        Additional params for LIMIT ORDERS:
        REQ:    price:          String	Yes	Specify price for currency
        OPT:    timeInForce:    String	No	Order timing strategy GTC, GTT, IOC, FOK (The default is GTC)
        OPT:    cancelAfter:    long	No	Cancel after n seconds，the order timing strategy is GTT
        OPT:    postOnly:       boolean	No	passive order labels, this is disabled when the order timing strategy is IOC or FOK
        OPT:    hidden:         boolean	No	Hidden or not (not shown in order book)
        OPT:    iceberg:        boolean	No	Whether or not only visible portions of orders are shown in iceberg orders
        OPT:    visibleSize:    String	No	Maximum visible quantity in iceberg orders

        Additional params for MARKET ORDERS where EITHER ONE IS REQUIRED TO BE SPECIFIED:
        REQ:    size:                   String  used when creating sell orders
        REQ:    funds:                  String  used when creating buy orders

        Return:
        orderNo:                orderid
        borrowSize:             borrow amount
        loanApplyId:            
        order_details:          order details       
        """
        size = abs(size) if size is not None else None
        funds = abs(funds) if funds is not None else None
        symbol = coin + "-" + fiat
        is_long = True if side =="buy" else False
        trade_order_id = self.generate_random_string(24)
        price = self.get_price(coin=coin, data=data, datetime_input=datetime_input)
        
        if order_type == "market" and price is None:
            price_with_slippage = self.get_price(coin=coin, data=data, datetime_input=datetime_input, activate_slippage=True, is_long=is_long)
        else:
            price_with_slippage = price 

        fee_rate = self.get_trading_fees(coin=coin, side=side)

        if side == "buy" and order_type == "market":
            maxBorrowAmount = balance_fiat * (self.leverage_factor-1)
            available_balance = balance_fiat * self.leverage_factor
            borrowSize =  min(maxBorrowAmount, abs(funds - maxBorrowAmount))
            if funds > available_balance:
                funds = funds + (available_balance - funds)

            size = (funds*(1-fee_rate)) / price_with_slippage

        else:
            balance_in_coin = balance_fiat / price_with_slippage
            maxBorrowAmount = balance_in_coin * (self.leverage_factor-1)
            available_balance = balance_in_coin * (self.leverage_factor -1)
            if size > available_balance:
                size = size + (available_balance - size)
            borrowSize =  size
            
            funds = size * price_with_slippage
        
        size_after_fees = (funds*(1-fee_rate)) / price_with_slippage

        fees = funds*fee_rate

        order_details = {   
                            "id":                   trade_order_id, 
                             "symbol":              symbol,
                             "opType":              "DEAL",
                             "type":                order_type,
                             "side":                side,
                             "price":               price_with_slippage if order_type=="market" else price,    
                             "size":                size,
                             "funds":               funds,
                             "dealFunds":           funds - fees,
                             "dealSize":            size_after_fees,
                             "fee":                 fees,
                             "feeCurrency" :        fiat,
                             "stp":                 None,
                             "stop":                None,
                             "stopTriggered":       False,
                             "stopPrice":           0,
                             "timeInForce":         "GTC",
                             "postOnly":            False,
                             "hidden":              False,
                             "iceberg":             False,
                             "visibleSize":         0,
                             "cancelAfter":         0,
                             "channel":             "BACKTEST",
                             "clientOid":           self.client_oid,
                             "remark":              None,
                             "tags":                None,
                             "isActive":            False,
                             "cancelExist":         False,
                             "createdAt":           datetime_input.strftime("%Y-%m-%d %H:%M:%S"),
                             "tradeType":           "TRADE"}

        self.logger.info(f"{'#'*1}   Auto-borrow margin order executed: Order ID {trade_order_id}, symbol: {symbol}, side: {side}, price: {order_details['price']}, dealFunds: {order_details['dealFunds']}, dealSize: {order_details['dealSize']}, Borrow Size: {borrowSize}    {'#'*1}")
        return trade_order_id, borrowSize, order_details
    

    def execute_auto_repay_margin_order_backtest(self, data=None, datetime_input=None, coin=None, fiat=None, size=None, funds=None, order_type="market", side=None, price=None):
        """
        ONLY FOR STANDARD MARGIN TRADING NOT HIGH-FREQUENCY MARGIN_V2 !!!!!!!!!

        REQ:    clientOid:      String
        REQ:    side:           "buy" or "sell"
        REQ:    symbol:         e.g. BTC-USDT
        REQ:    type:           "limit" or "market"

        OPT:    stp:            self-trade-prevention  (CN, CO, CB, DC)
        OPT:    isIsolated:     TRUE-isolated margin, FALSE -cross margin   def: false
        OPT:    autoBorrow:     True or False
        OPT:    autoRepay:      True or False
        Link:           https://www.kucoin.com/docs/rest/margin-trading/orders/place-margin-order

        Additional params for LIMIT ORDERS:
        REQ:    price:          String	Yes	Specify price for currency
        OPT:    timeInForce:    String	No	Order timing strategy GTC, GTT, IOC, FOK (The default is GTC)
        OPT:    cancelAfter:    long	No	Cancel after n seconds，the order timing strategy is GTT
        OPT:    postOnly:       boolean	No	passive order labels, this is disabled when the order timing strategy is IOC or FOK
        OPT:    hidden:         boolean	No	Hidden or not (not shown in order book)
        OPT:    iceberg:        boolean	No	Whether or not only visible portions of orders are shown in iceberg orders
        OPT:    visibleSize:    String	No	Maximum visible quantity in iceberg orders

        Additional params for MARKET ORDERS where EITHER ONE IS REQUIRED TO BE SPECIFIED:
        REQ:    size:                   String  used when creating sell orders
        REQ:    funds:                  String  used when creating buy orders

        Return:
        orderNo:                orderid
        borrowSize:             borrow amount
        loanApplyId:            
        order_details:          order details       
        """
        size = abs(size) if size is not None else None
        funds = abs(funds) if funds is not None else None
        symbol = coin + "-" + fiat
        is_long = True if side =="buy" else False
        trade_order_id = self.generate_random_string(24)
        
        # price = self.get_price(data=data, datetime_input=datetime_input, is_long=is_long)
        
        if order_type == "market" and price is None:
            price_with_slippage = self.get_price(coin=coin, data=data, datetime_input=datetime_input, activate_slippage=True, is_long=is_long)
        else:
            price_with_slippage = price 

        fee_rate = self.get_trading_fees(coin=coin, side=side)
        
        if side == "buy" and order_type == "market":
            fees = funds*fee_rate
            size = funds / price_with_slippage
            size_after_fees = (funds  - fees) / price_with_slippage
        else:
            funds = (size * price_with_slippage)
            fees = funds*fee_rate
            size_after_fees = (funds  - fees) / price_with_slippage
            

        borrowSize = 0 
        
        order_details = {   
                        "id":                  trade_order_id, 
                        "symbol":              symbol,
                        "opType":              "DEAL",
                        "type":                order_type,
                        "side":                side,
                        "price":               price_with_slippage if order_type=="market" else price,    
                        "size":                size,
                        "funds":               funds,
                        "dealFunds":           funds-fees,
                        "dealSize":            size_after_fees,
                        "fee":                 fees,
                        "feeCurrency" :        fiat,
                        "stp":                 None,
                        "stop":                None,
                        "stopTriggered":       False,
                        "stopPrice":           0,
                        "timeInForce":         "GTC",
                        "postOnly":            False,
                        "hidden":              False,
                        "iceberg":             False,
                        "visibleSize":         0,
                        "cancelAfter":         0,
                        "channel":             "BACKTEST",
                        "clientOid":           self.client_oid,
                        "remark":              None,
                        "tags":                None,
                        "isActive":            False,
                        "cancelExist":         False,
                        "createdAt":           datetime_input.strftime("%Y-%m-%d %H:%M:%S"),
                        "tradeType":           "TRADE"}

        self.logger.info(f"{'#'*1}  Auto-repay margin order executed: Order ID {trade_order_id}, symbol: {symbol}, side: {side}, price: {order_details['price']}, dealFunds: {order_details['dealFunds']}, dealSize: {order_details['dealSize']}, Borrow Size: {borrowSize}    {'#'*1}")
        return trade_order_id, borrowSize, order_details
    
    def execute_margin_order_backtest(self, data=None, datetime_input=None, coin=None, fiat=None, size=None, funds=None, order_type="market", side=None, price=None):
        """
        ONLY FOR STANDARD MARGIN TRADING NOT HIGH-FREQUENCY MARGIN_V2 !!!!!!!!!

        REQ:    clientOid:      String
        REQ:    side:           "buy" or "sell"
        REQ:    symbol:         e.g. BTC-USDT
        REQ:    type:           "limit" or "market"

        OPT:    stp:            self-trade-prevention  (CN, CO, CB, DC)
        OPT:    isIsolated:     TRUE-isolated margin, FALSE -cross margin   def: false
        OPT:    autoBorrow:     True or False
        OPT:    autoRepay:      True or False
        Link:           https://www.kucoin.com/docs/rest/margin-trading/orders/place-margin-order

        Additional params for LIMIT ORDERS:
        REQ:    price:          String	Yes	Specify price for currency
        OPT:    timeInForce:    String	No	Order timing strategy GTC, GTT, IOC, FOK (The default is GTC)
        OPT:    cancelAfter:    long	No	Cancel after n seconds，the order timing strategy is GTT
        OPT:    postOnly:       boolean	No	passive order labels, this is disabled when the order timing strategy is IOC or FOK
        OPT:    hidden:         boolean	No	Hidden or not (not shown in order book)
        OPT:    iceberg:        boolean	No	Whether or not only visible portions of orders are shown in iceberg orders
        OPT:    visibleSize:    String	No	Maximum visible quantity in iceberg orders

        Additional params for MARKET ORDERS where EITHER ONE IS REQUIRED TO BE SPECIFIED:
        REQ:    size:                   String  used when creating sell orders
        REQ:    funds:                  String  used when creating buy orders

        Return:
        orderNo:                orderid
        borrowSize:             borrow amount
        loanApplyId:            
        order_details:          order details       
        """
        size = abs(size) if size is not None else None
        funds = abs(funds) if funds is not None else None
        symbol = coin + "-" + fiat
        is_long = True if side =="buy" else False
        trade_order_id = self.generate_random_string(24)
        # price = self.get_price(data=data, datetime_input=datetime_input, activate_slippage=True)

        if order_type == "market" and price is None:
            price_with_slippage = self.get_price(coin=coin, data=data, datetime_input=datetime_input, activate_slippage=True, is_long=is_long)
        else:
            price_with_slippage = price 

        fee_rate = self.get_trading_fees(coin=coin, side=side)
        
        if side == "buy" and order_type == "market":
            fees = funds*fee_rate
            size = funds / price_with_slippage
            size_after_fees = (funds  - fees) / price_with_slippage
        else:
            funds = (size * price_with_slippage)
            fees = funds*fee_rate
            size_after_fees = (funds  - fees) / price_with_slippage

        borrowSize = 0 

        order_details = {   
                        "id":                  trade_order_id, 
                        "symbol":              symbol,
                        "opType":              "DEAL",
                        "type":                order_type,
                        "side":                side,
                        "price":               price_with_slippage if order_type=="market" else price,    
                        "size":                size,
                        "funds":               funds,
                        "dealFunds":           funds-fees,
                        "dealSize":            size_after_fees,
                        "fee":                 fees,
                        "feeCurrency" :        fiat,
                        "stp":                 None,
                        "stop":                None,
                        "stopTriggered":       False,
                        "stopPrice":           0,
                        "timeInForce":         "GTC",
                        "postOnly":            False,
                        "hidden":              False,
                        "iceberg":             False,
                        "visibleSize":         0,
                        "cancelAfter":         0,
                        "channel":             "BACKTEST",
                        "clientOid":           self.client_oid,
                        "remark":              None,
                        "tags":                None,
                        "isActive":            False,
                        "cancelExist":         False,
                        "createdAt":           datetime_input.strftime("%Y-%m-%d %H:%M:%S"),
                        "tradeType":           "TRADE"}

        self.logger.info(f"{'#'*1}   Standard Margin order executed: Order ID {trade_order_id}, symbol: {symbol}, side: {side}, price: {order_details['price']}, dealFunds: {order_details['dealFunds']}, dealSize: {order_details['dealSize']}.   {'#'*1}")
        return trade_order_id, order_details

    
    def execute_trade(self, decision):
        """
        Führt einen Handel basierend auf dem aktuellen Modus und der Entscheidung aus.
        """
        if self.trade_mode == 'spot':
            self.execute_spot_trade(decision)
        elif self.trade_mode == 'margin':
            self.execute_margin_trade(decision)
        else:
            print("Ungültiger Handelsmodus.")

    def execute_spot_trade(self, decision):
        """
        Simuliert die Ausführung eines Spot-Handels.
        """
        order_id = random.randint(100000, 999999)
        timestamp = datetime.now()
        order_details = {                    #Am besten an genaue parameter namen von position_info von enter_margin_trade verwenden
            'OrderID': order_id,
            'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Symbol': decision['symbol'],
            'Type': decision['action'],
            'Price': decision['price'],
            'Quantity': decision['quantity'],
            'Status': 'Executed',
            'Mode': 'Spot'
        }
        self.order_history = pd.concat([self.order_history, pd.DataFrame([order_details])], ignore_index=True)
        print(f"Spot trade executed: {order_details}")

    def check_order_triggered(self, coin=None, fiat=None, data=None, datetime_input = None, order_details=None, order_type=None):
        """
        REQ:        order_id.       individual orderId by trade
        REQ:        symbol:         e.g. BTC-USDT
        
        Returns:    True if order still active, False otherwise
        """
        try:
            # Retrieve the order details from KuCoin
            price = self.get_price(coin=coin, data=data, datetime_input=datetime_input)
            order_price = order_details["price"]
            side = order_details["side"]
            
            # Check if the order has been executed or is still active
            
            if order_type == "tp":
                if side == "buy":
                    if price<= order_price:
                        return True  
                    else:
                        return False # Order is still active, not triggered
                elif side == "sell":
                    if price>= order_price:
                        return True  
                    else:
                        return False # Order is still active, not triggered
            elif order_type =="sl":
                if side == "buy":
                    if price>= order_price:
                        return True  
                    else:
                        return  False # Order is still active, not triggered
                elif side == "sell":
                    if price<= order_price:
                        return True  
                    else:
                        return False # Order is still active, not triggered
            elif order_type == "buy":
                if price<= order_price:
                    return True  
                else:
                    return  False # Order is still active, not triggered
            elif order_type == "sell":
                if price >= order_price:
                    return True  
                else:
                    return False # Order is still active, not triggered

        except Exception as e:
            print(f"An error occurred while checking order status: {e}")
            return


    def check_sl_or_tp_triggered_backtest(self, data=None, datetime_input=None, coin=None, fiat=None, position=None):
        """
        REQ:            Order id:           Order Id to check, use initial buy / sell order from positions tracker
        REG:            symbol:             Symbol to query order details

        Returns:

        Order that has been canceled:       Either TP, SL or not triggered if no stop loss or take profit has been reached yet

        Functionality:                      

        Function queries if any of SL or TP has been triggered. If yes it will get the details and use that to update the positions tracker. It also calculates the correct tax amount. 
        """
        
        order_id = position["order_id"]
        symbol = position["pair"]
        side = position["side"]
        size = position['size']
        funds = position['funds']

        sl_order_details = self.ledger_data["order_details"][order_id]["sl_order_details"]
        tp_order_details = self.ledger_data["order_details"][order_id]["tp_order_details"]

        if self.check_order_triggered(coin=coin, fiat=fiat, data=data, datetime_input=datetime_input, order_details=sl_order_details, order_type="sl"):
            try:
                sl_order_details = sl_order_details
                #timestamp_now = datetime.now().timestamp()*1000    check later in backtest if createdAt is timestamp when order was filled else use timestamp now
                self.logger.info("#################### Stop loss order triggered ####################")
                self.logger.info(f"{'#'*1}  Stop loss order triggered: Order ID {sl_order_details['id']}  {'#'*1}")
                self.logger.info("#################################################################")
                #update position respectively
                side = sl_order_details["side"]
                
                if side == "sell":
                    borrowed_asset = fiat
                    borrowed_amount = funds
                    closing_amount = sl_order_details["dealFunds"] 
                    pnl_trade = closing_amount - position['invested_usdt']

                    self.balances["trailing_balances"][fiat]  += sl_order_details["dealFunds"]
                    self.balances["trailing_balances"][coin]  = self.balances["trailing_balances"][coin] - sl_order_details["size"]

                elif side =="buy":
                    borrowed_asset = coin
                    borrowed_amount = size
                    closing_amount = sl_order_details["size"]
                    pnl_trade = closing_amount - size
        
                    if np.round(pnl_trade,8) < 0:
                        rebuy_price = self.get_price(coin=coin, data=data, datetime_input=datetime_input) * (1+self.slippage)
                        rebuy_order_id, rebuy_order_details = self.execute_margin_order_backtest(data=data, datetime_input=datetime_input, coin=coin, fiat=fiat, funds=None, size=abs(pnl_trade), order_type="limit", side=side, price=rebuy_price)
                        position["rebuy_order_id"] = rebuy_order_id

                        self.balances["trailing_balances"][coin] = pnl_trade + rebuy_order_details["size"]
                        self.balances["trailing_balances"][fiat] = self.balances["trailing_balances"][fiat] + (self.balances["trailing_balances"][fiat] - sl_order_details["funds"]) - rebuy_order_details["dealFunds"]  - sl_order_details["fee"] #- abs((sl_order_details["funds"] - (position["funds"]+position["fees_opening_trade"])) + rebuy_order_details["dealFunds"] + sl_order_details["fee"] + rebuy_order_details["fee"])
                    
                        usdt_convert_order_details = {"fee":0}
                        usdt_convert_order_details["dealFunds"] = 0
                        usdt_convert_order_details["funds"] = 0
                    
                    elif np.round(pnl_trade,8) == 0:
                        rebuy_order_details = {"fee":0}
                        rebuy_order_details["dealFunds"] = 0
                        rebuy_order_details["funds"] = 0
                        
                        usdt_convert_order_details = {"fee":0}
                        usdt_convert_order_details["dealFunds"] = 0
                        usdt_convert_order_details["funds"] = 0
                        
                        self.balances["trailing_balances"][coin] = self.balances["trailing_balances"][coin]  
                        self.balances["trailing_balances"][fiat] = self.balances["trailing_balances"][fiat] + (self.balances["trailing_balances"][fiat] - sl_order_details["funds"]) - sl_order_details["fee"] #- (position["funds"]+position["fees_opening_trade"])) + sl_order_details["fee"])
                    else:
                        # initial balance in coin - close_order_details["dealSize"]
                        rebuy_order_details = {"fee":0}
                        rebuy_order_details["dealFunds"] = 0
                        rebuy_order_details["funds"] = 0

                        self.balances["trailing_balances"][coin] = pnl_trade  

                        #when short convert coin into usdt
                        coin_balance_after_trade = self.balances["trailing_balances"][coin]
                        
                        usdt_convert_order_id, usdt_convert_order_details = self.execute_margin_order_backtest(data=data, datetime_input=datetime_input, coin=coin, fiat=fiat, size=coin_balance_after_trade, order_type="market", side="sell")
                        position["convert_order_id"] = usdt_convert_order_id
                        self.balances["trailing_balances"][fiat] =  self.balances["trailing_balances"][fiat] + (self.balances["trailing_balances"][fiat] - sl_order_details["funds"]) + usdt_convert_order_details["dealFunds"] - sl_order_details["fee"] 
                        self.balances["trailing_balances"][coin] = pnl_trade - usdt_convert_order_details["size"]
                
                #update position
                position["close_order_id"] = position.get("stop_loss_order_id")
                position['status'] = 'closed'
                position['timestamp_closed'] = datetime_input.strftime("%Y-%m-%d %H:%M:%S")
                position['time_closed'] =    datetime_input.strftime("%Y-%m-%d %H:%M:%S")
                position["repaid"] = "repaid"
                
                if side =="buy" and pnl_trade <0:
                    fees = { "sum":     sum([sl_order_details["fee"], rebuy_order_details["fee"], usdt_convert_order_details["fee"]]), 
                            "single":   [sl_order_details["fee"], rebuy_order_details["fee"], usdt_convert_order_details["fee"]]}
                elif side =="buy" and pnl_trade >=0:
                    fees = { "sum":     sum([sl_order_details["fee"], usdt_convert_order_details["fee"]]), 
                            "single":   [sl_order_details["fee"], usdt_convert_order_details["fee"]]}                
                else:
                    fees = sl_order_details["fee"]

                position["fees_closing_trade"] = fees["sum"] if isinstance(fees, dict) else fees
                position["closing_balance"] = self.balances["trailing_balances"][fiat]

                if side == "sell":
                    closing_amount_invested = sl_order_details["dealFunds"]
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = (closing_amount_invested - position['invested_usdt'])    #position['closing_balance']
                else:
                    # time.sleep(1)
                    closing_amount_invested = sl_order_details["funds"] - usdt_convert_order_details["dealFunds"] + rebuy_order_details["funds"]
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = (position['funds'] - (closing_amount_invested+fees["sum"]))

                if position["pnl"] > 0:
                    self.logger.info(f"{'#'*1} Position closed with profit of {position['pnl']} {fiat}  {'#'*1}")
                    position["owed_tax"] = position["pnl"] * 0.25
                    self.trading_metrics_data["winning_trades"] += 1

                    if position["pnl"] > self.trading_metrics_data["largest_gain"]:
                        self.trading_metrics_data["largest_gain"] = position["pnl"]
                elif position["pnl"] < 0:
                    self.logger.info(f"{'#'*1}   Position closed with loss of {position['pnl']} {fiat}    {'#'*1}")
                    position["owed_tax"] = position["pnl"] * 0.25
                    self.trading_metrics_data["losing_trades"] += 1

                    if position["pnl"] < self.trading_metrics_data["largest_loss"]:
                        self.trading_metrics_data["largest_loss"] = position["pnl"]

                self.save_trading_metrics(self.trading_metrics_path, self.trading_metrics_data)  

                #set open position to None
                self.ledger_data["current_trades"][symbol] = None
            
                #update balances
                self.ledger_data["balances"]["trailing_balances"]["total_balance"] = self.calculate_balance_in_fiat(data=data, datetime_input=datetime_input, fiat=fiat, coin=coin, getSum=True, get_only_trade_pair_bal=True)
                self.logger.info(f"{'#'*1} Current total balance in USDT is {self.ledger_data['balances']['trailing_balances']['total_balance']} {'#'*1}")

                if side == "buy":
                    self.ledger_data["order_details"][order_id]["rebuy_order_details"] = rebuy_order_details
                    self.ledger_data["order_details"][order_id]["usdt_convert_order_details"] = usdt_convert_order_details

                self.update_position(position)
                time.sleep(self.time_sleep_factor_long)

                return str("tp_cancel")
            except:
                print("take profit order could not be canceled")
        
        elif self.check_order_triggered(coin=coin, fiat=fiat, data=data, datetime_input=datetime_input, order_details=tp_order_details, order_type="tp"):
            try:
                #cancel sl order
                # self.cancel_order_by_order_id(position.get('stop_loss_order_id'))
                tp_order_details = tp_order_details
                self.logger.info("#################### Take profit order triggered ####################")
                self.logger.info(f"{'#'*1}  Take profit order triggered: Order ID {tp_order_details['id']}  {'#'*1}")
                self.logger.info("#################################################################")
                #update position respectively
                side = tp_order_details["side"]
            
                if side == "sell":
                    borrowed_asset = fiat
                    borrowed_amount = funds 
                    closing_amount = tp_order_details["dealFunds"] 
                    pnl_trade = closing_amount - position['invested_usdt']

                    self.balances["trailing_balances"][fiat]  += tp_order_details["dealFunds"]
                    self.balances["trailing_balances"][coin]  = self.balances["trailing_balances"][coin] - tp_order_details["size"]

                elif side =="buy":
                    borrowed_asset = coin
                    borrowed_amount = size
                    closing_amount = tp_order_details["size"]
                    pnl_trade = closing_amount - size

                    if np.round(pnl_trade,8) < 0:
                        rebuy_price = self.get_price(coin=coin, data=data, datetime_input=datetime_input) * (1+self.slippage)
                        rebuy_order_id, rebuy_order_details = self.execute_margin_order_backtest(data=data, datetime_input=datetime_input, coin=coin, fiat=fiat, funds=None, size=abs(pnl_trade), order_type="limit", side=side, price=rebuy_price)
                        position["rebuy_order_id"] = rebuy_order_id

                        self.balances["trailing_balances"][coin] = pnl_trade + rebuy_order_details["size"]
                        self.balances["trailing_balances"][fiat] = self.balances["trailing_balances"][fiat] + (self.balances["trailing_balances"][fiat] - tp_order_details["funds"]) - rebuy_order_details["dealFunds"]  - tp_order_details["fee"] #- abs((tp_order_details["funds"] - (position["funds"]+position["fees_opening_trade"])) + rebuy_order_details["dealFunds"] + tp_order_details["fee"] + rebuy_order_details["fee"])

                        usdt_convert_order_details = {"fee":0}
                        usdt_convert_order_details["dealFunds"] = 0
                        usdt_convert_order_details["funds"] = 0
                    
                    elif np.round(pnl_trade,8) == 0:
                        rebuy_order_details = {"fee":0}
                        rebuy_order_details["dealFunds"] = 0
                        rebuy_order_details["funds"] = 0

                        usdt_convert_order_details = {"fee":0}
                        usdt_convert_order_details["dealFunds"] = 0
                        usdt_convert_order_details["funds"] = 0

                        self.balances["trailing_balances"][coin] = self.balances["trailing_balances"][coin]       #check that    funds is    999 * 0.99 and fees is 999*0.001 so that 999 - 989.01 - 999*0.001 = 9.01
                        self.balances["trailing_balances"][fiat] = self.balances["trailing_balances"][fiat] + (self.balances["trailing_balances"][fiat] - tp_order_details["funds"] - tp_order_details["fee"])  #(((position["funds"]+position["fees_opening_trade"]) - tp_order_details["funds"])) - tp_order_details["fee"]
                    else:
                        # initial balance in coin - close_order_details["dealSize"]
                        rebuy_order_details = {"fee":0}
                        rebuy_order_details["dealFunds"] = 0
                        rebuy_order_details["funds"] = 0

                        self.balances["trailing_balances"][coin] = pnl_trade  

                        #when short convert coin into usdt
                        coin_balance_after_trade = self.balances["trailing_balances"][coin]
                        
                        usdt_convert_order_id, usdt_convert_order_details = self.execute_margin_order_backtest(data=data, datetime_input=datetime_input, coin=coin, fiat=fiat, size=coin_balance_after_trade, order_type="market", side="sell")
                        position["convert_order_id"] = usdt_convert_order_id
                        self.balances["trailing_balances"][fiat] =  self.balances["trailing_balances"][fiat] + (self.balances["trailing_balances"][fiat] - tp_order_details["funds"]) + usdt_convert_order_details["dealFunds" ] - tp_order_details["fee"]   #+ usdt_convert_order_details["dealFunds"] -  (tp_order_details["fee"])
                        self.balances["trailing_balances"][coin] = pnl_trade - usdt_convert_order_details["size"]
                    
                #update position
                position["close_order_id"] = position.get("stop_loss_order_id")
                position['status'] = 'closed'
                position['timestamp_closed'] = datetime_input.strftime("%Y-%m-%d %H:%M:%S")
                position['time_closed'] =     datetime_input.strftime("%Y-%m-%d %H:%M:%S")
                position["repaid"] = "repaid"
                
                if side =="buy" and pnl_trade <0:
                    fees = { "sum":     sum([tp_order_details["fee"], rebuy_order_details["fee"], usdt_convert_order_details["fee"]]), 
                            "single":   [tp_order_details["fee"], rebuy_order_details["fee"], usdt_convert_order_details["fee"]]}
                elif side =="buy" and pnl_trade >=0:
                    fees = { "sum":     sum([tp_order_details["fee"], usdt_convert_order_details["fee"]]), 
                            "single":   [tp_order_details["fee"], usdt_convert_order_details["fee"]]}                
                else:
                    fees = tp_order_details["fee"]

                position["fees_closing_trade"] = fees["sum"] if isinstance(fees, dict) else fees
                position["closing_balance"] = self.balances["trailing_balances"][fiat]

                if side == "sell":
                    closing_amount_invested = tp_order_details["dealFunds"]
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = (closing_amount_invested - position['invested_usdt'])   #position['closing_balance']
                else:
                    closing_amount_invested = tp_order_details["funds"] - usdt_convert_order_details["dealFunds"] + rebuy_order_details["funds"]    #using funds because we need to invest/buy the coin again
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = (position['funds'] - closing_amount_invested)
                
                if position["pnl"] > 0:
                    self.logger.info(f"{'#'*1} Position closed with profit of {position['pnl']} {fiat}  {'#'*1}")
                    position["owed_tax"] = position["pnl"] * 0.25
                    self.trading_metrics_data["winning_trades"] += 1

                    if position["pnl"] > self.trading_metrics_data["largest_gain"]:
                        self.trading_metrics_data["largest_gain"] = position["pnl"]
                        
                elif position["pnl"] < 0:
                    self.logger.info(f"{'#'*1}   Position closed with loss of {position['pnl']} {fiat}    {'#'*1}")
                    position["owed_tax"] = position["pnl"] * 0.25
                    self.trading_metrics_data["losing_trades"] += 1

                    if position["pnl"] < self.trading_metrics_data["largest_loss"]:
                        self.trading_metrics_data["largest_loss"] = position["pnl"]
                
                self.save_trading_metrics(self.trading_metrics_path, self.trading_metrics_data)  

                #set open position to None
                self.ledger_data["current_trades"][symbol] = None
                
                #update balances
                self.ledger_data["balances"]["trailing_balances"]["total_balance"] = self.calculate_balance_in_fiat(data=data, datetime_input=datetime_input, fiat=fiat, coin=coin, getSum=True, get_only_trade_pair_bal=True)
                self.logger.info(f"{'#'*1} Current total balance in USDT is {self.ledger_data['balances']['trailing_balances']['total_balance']} {'#'*1}")
                
                if side == "buy":
                    self.ledger_data["order_details"][order_id]["rebuy_order_details"] = rebuy_order_details
                    self.ledger_data["order_details"][order_id]["usdt_convert_order_details"] = usdt_convert_order_details
                
                self.update_position(position)
                time.sleep(self.time_sleep_factor_long)

                return str("sl_cancel")
            except:
                print("Stop loss order could not be canceled")
    
        elif not self.check_order_triggered(coin=coin, fiat=fiat, data=data, datetime_input=datetime_input, order_details=sl_order_details, order_type="sl") and not self.check_order_triggered(coin=coin, fiat=fiat, data=data, datetime_input=datetime_input, order_details=tp_order_details, order_type="tp"):
            return str("not triggered")

#################################################################################################################################################################
#
#
#                                                                       POSITION AND ACCOUNT FUNCTIONS
#
#################################################################################################################################################################
  
    # def get_account_list(self, curr=None):
    #     df = pd.DataFrame(columns=["currency","balance"])
        
    #     for coin in self.ledger_data["balances"]["trailing_balances"]:
    #         array = np.zeros((1,2))
    #         tmp_df = pd.DataFrame(data = array,columns=["currency","balance"] )
    #         tmp_df["currency"] = coin
    #         tmp_df["balance"] = self.ledger_data["balances"]["trailing_balances"][coin]
            
    #         df = pd.concat([df, tmp_df], axis=0)
        
    #     if curr:
    #         df = df[df.currency==curr]

    #     df = df.reset_index(drop=True)
    #     return df
    

    def get_account_list(self, curr=None):
        df = pd.DataFrame(columns=["currency", "balance"])
        
        for coin in self.ledger_data["balances"]["trailing_balances"]:
            balance = self.ledger_data["balances"]["trailing_balances"][coin]
            if pd.isna(balance):  # Check if balance is NaN
                balance = 0  # Set balance to 0 if it's NaN
            tmp_df = pd.DataFrame(data={"currency": [coin], "balance": [balance]})
            
            # Additional explicit checks for empty and all-NA cases
            if not tmp_df.empty and not tmp_df.isna().all(axis=None):
                if df.empty:
                    df = tmp_df.copy()  # Direct assignment if df is initially empty
                else:
                    df = pd.concat([df, tmp_df], ignore_index=True)  # Only concatenate if tmp_df is valid

        if curr:
            df = df[df.currency == curr]

        df = df.reset_index(drop=True)
        return df


    def calculate_balance_in_fiat(self,data=None, datetime_input=None, coin=None, fiat=None, getSum=False, get_only_trade_pair_bal = False):
        """
        REQ:        fiat:                   e.g. fiat currency to calculate value for e.g. EUR or USD, or USDT
        OPT:        coin:                   specify if you want to calc the value for a specific coin list or single string
        OPT:        use_only_available:     use only the available balance in case you should be in a tradem, def is False
        OPT:        getSum:                 Calculates the sum of all queried balances
        OPT:        hf:                     Default set to True, if true returns only balance in HF account, else all other balances
        """
        account_balances = self.get_account_list()
        # account_balances
        
        if "total_balance" in account_balances["currency"].values:
            account_balances = account_balances.copy(deep=True).drop(account_balances[account_balances["currency"]=="total_balance"].index)
        # account_balances = account_balances[account_balances.values!=0]
        account_balances = pd.DataFrame(account_balances.groupby(["currency"])["balance"].sum())
        
        if coin is not None:
            if get_only_trade_pair_bal:
               account_balances = account_balances.loc[account_balances.index.isin([coin, fiat])]

            else:
                account_balances = account_balances.loc[account_balances.index==coin]
        
        fiat_balance_df = pd.DataFrame(columns=account_balances.index, index=["value"])
        fiat_balance_df.index.name = None
        total_balance = 0

        for coin in account_balances.index:
            asset_amount = account_balances.loc[coin, "balance"]
            asset_price = 0
            # if fiat == "EUR" or fiat == "USD":
            #     asset_price = self.get_fiat_price(fiat, coin).loc["data",coin]
            #else:
            if coin != fiat:
                symbol = coin + "-" + fiat
                asset_price = self.get_price(coin=coin, data=data, datetime_input=datetime_input)
            else:
                asset_price = 1
            balance_value = asset_amount * asset_price
            if getSum:
                total_balance += balance_value 
            else:
                fiat_balance_df.loc["value",coin] = balance_value

        if getSum:
            return total_balance
        else:
            return fiat_balance_df

################################################################################################################################################################
#
#
#                                                                  BACKTEST FUNCTIONS
#
#################################################################################################################################################################

    def get_price(self, coin=None, data=None, datetime_input=None,  activate_slippage=False, is_long=True):
        """
        REQ:        data                    data to be used, must cointain column with price (<- preliminary, can also be close)
        REQ:        timestamp               unix timestamp
        OPT:        activate_slippage       True if slippage shall be applied, False only price is returned, default=False
        OPT:        is_long                 if activate_slippage is enabled, specify the slippage side
        
        Returns:

        price       price of currency
        """
    
        if isinstance(datetime_input, str):
        # Convert datetime string to pandas datetime object
            target_time = pd.to_datetime(datetime_input, format="%Y-%m-%d %H:%M:%S")
        elif isinstance(datetime_input, pd.Timestamp):
            # Use datetime object directly
            target_time = datetime_input

        elif isinstance(datetime_input, pd.DatetimeIndex):
            pass
        else:
            raise ValueError("datetime_input must be a string or pandas Timestamp object")
        
        # Find the index of the closest datetime to the target_time
        # Find the index of the closest datetime to the target_time
        # closest_index = (data.index - target_time).abs().argmin()
        
        coin = self.coin if coin is None else coin
        
        slug = self.ticker_to_slug_mapping[coin]

        price_metric = "price_" + self.fiat.lower() + "_" + slug 
        
        price = data.loc[target_time, price_metric]
        # Get the price at the closest index
        
        if activate_slippage:
            # Assuming apply_slippage is a method that modifies the price based on some logic
            price = self.apply_slippage(price, is_buy_order=is_long)
            
        return price

    def apply_slippage(self, price, is_buy_order, slippage_percent=None):
        """
        Berechnet den Ausführungspreis unter Berücksichtigung von Slippage.

        :param price: Der ursprünglich erwartete Ausführungspreis der Order.
        :param is_buy_order: True, wenn es sich um eine Kauforder handelt, sonst False.
        :param slippage_percent: Der Slippage-Prozentsatz (z.B. 0.05 für 5% Slippage).
        :return: Der angepasste Ausführungspreis unter Berücksichtigung von Slippage.
        """
        slippage_percent = self.slippage if slippage_percent is None else slippage_percent
        slippage_amount = price * slippage_percent
        if is_buy_order:
            return price + slippage_amount
        else:
            return price - slippage_amount

    def generate_random_string(self, length):
        # Define characters to choose from
        characters = string.ascii_uppercase + string.digits
        
        length = length

        # Generate random string
        random_string = ''.join(random.choice(characters) for _ in range(length))
        
        return random_string

    def convert_timestamp_to_datetime(self, timestamp):
        """"
        REQ:        timestamp       timestamp in unix ms format

        Returns:    datetime:       datetime object of format "Y-m-d H:M" 
        """
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


    def get_trading_fees(self, coin=None, side=None):
        """
        Gibt simulierte Handelsgebühren zurück.
        """
        trading_fees = {coin:{"takerFeeRate":0.001, "makerFeeRate":0.001}}


        if coin in trading_fees:
            fee_rate = trading_fees[coin]
            fee_rate = float(fee_rate["takerFeeRate"]) if side == "buy" else float(fee_rate["makerFeeRate"])

            return fee_rate
        else:
            self.logger.error("Coin not found in trading fees")
            return None
        
    #################################################################################################################################################################
#
#
#                                                                  POSITION Functions Internal
#
#################################################################################################################################################################

    
    def load_ledger_data(self, filepath):
        if not os.path.exists(filepath):
            ledger = {"positions": [], "balances": {"initial_balance":{self.coin:0, self.fiat:1000}, "trailing_balances":{self.coin:0, self.fiat:1000}}, "current_trades":{self.symbol:None}, "order_details":{}}
            return ledger 
        with open(filepath, "r") as file:
            return json.load(file)
        
    def save_data(self, filepath):
        # Convert Timestamp objects to strings
        for position in self.ledger_data["positions"]:
            if 'timestamp_opened' in position and isinstance(position['timestamp_opened'], pd.Timestamp):
                position['timestamp_opened'] = position['timestamp_opened'].isoformat()
            if 'timestamp_closed' in position and isinstance(position['timestamp_closed'], pd.Timestamp):
                position['timestamp_closed'] = position['timestamp_closed'].isoformat()

        with open(filepath, "w") as file:
            json.dump(self.ledger_data, file, indent=4)

    def load_trading_metrics(self, metrics_file_path=None):
        if not os.path.exists(metrics_file_path):
            trading_metrics_data = {"losing_trades": 0 , "winning_trades": 0, "largest_loss":0, "largest_gain":0, "total_trades":0}
            
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
        'current_price':                current_price
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
        "owed tax:                      None
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
        log_directory = "Kucoin Trader Backtest"
        log_file_name = f"Kucoin_trader_backtest_log_{timestamp}.txt"
        log_file_path = os.path.join(logger_path, log_directory, log_file_name)
        if not os.path.exists(os.path.join(logger_path, log_directory)):
            os.makedirs(os.path.join(logger_path, log_directory))

        self.logger.add(log_file_path, rotation="500 MB", level="INFO")




#################################################################################################################################################################
#
#
#                                                                       DATA FUNCTIONS  ##  WILL BE REMOVED  ##  AS SEPERATE CLASS
#
#################################################################################################################################################################


    # 
    #               We will code the historical data loader as a seperate class with more functionality, below is just a temporary solution
    # 
        

    # def load_historical_data(self, filepath, format='csv', datetime_col='timestamp', price_col='close', frequency=None):
    #     """
    #     Lädt und verarbeitet Handelsdaten aus einer Datei.

    #     :param filepath: Der Pfad zur Datenquelle.
    #     :param format: Das Format der Datenquelle ('csv', 'json', 'excel').
    #     :param datetime_col: Name der Spalte mit Zeitstempeln.
    #     :param price_col: Name der Spalte mit Preisen.
    #     :param frequency: Frequenz zur Umwandlung der Daten ('T' für Minuten, 'H' für Stunden).
    #     """
    #     if format == 'csv':
    #         data = pd.read_csv(filepath)
    #     elif format == 'json':
    #         data = pd.read_json(filepath)
    #     elif format == 'excel':
    #         data = pd.read_excel(filepath)
    #     else:
    #         raise ValueError(f"Unsupported format: {format}")

    #     data[datetime_col] = pd.to_datetime(data[datetime_col])
    #     if frequency:
    #         data.set_index(datetime_col, inplace=True)
    #         data = data.resample(frequency).ffill()  # Forward fill zur Imputation von fehlenden Werten
    #         data.reset_index(inplace=True)

    #     self.historical_data = data[[datetime_col, price_col]]
    #     print("Daten geladen und verarbeitet.")


    # def connect_websocket(self):
    #     """
    #     Stellt eine Verbindung zum KuCoin WebSocket Server her und empfängt live Marktdaten.
    #     """
    #     # Pfad zur Konfigurationsdatei
    #     config_path = os.path.join(os.path.dirname(__file__), 'Config', 'JansConfig.ini')

    #     # Instanz des KucoinWebSocketClients erstellen
    #     self.ws_client = KucoinWebSocketClient(config_path)

    #     # Thread zum Starten des WebSocket-Clients
    #     ws_thread = threading.Thread(target=lambda: self.ws_client.start(private=False))
    #     ws_thread.start()

    #     print("WebSocket-Verbindung hergestellt.")


    # def on_new_market_data(self, data):
    #     """
    #     Wird aufgerufen, wenn neue Marktdaten über WebSocket empfangen werden.

    #     :param data: Die empfangenen Marktdaten.
    #     """
    #     price_data = data['price']
    #     print("Neue Marktdaten empfangen:", price_data)


    # def receive_data(self, data, strategy):
    #     """
    #     Empfängt Markt- oder historische Daten und entscheidet über Handelsaktionen basierend auf einer Strategie.
    #     """
    #     decision = strategy(data)
    #     if decision['action'] != 'Hold':
    #         self.execute_trade(decision)

    # def process_data_parallel(self, data, chunksize=100):
    #     """
    #     Verarbeitet Daten in parallelen Prozessen.

    #     :param data: Die zu verarbeitenden Daten als DataFrame.
    #     :param chunksize: Die Größe eines Datenchunks für jeden Prozess.
    #     """
    #     chunks = [data[i:i + chunksize] for i in range(0, data.shape[0], chunksize)]

    #     with ProcessPoolExecutor() as executor:
    #         results = executor.map(process_data_chunk, chunks)

    #     combined_results = pd.concat(results)
    #     return combined_results



# trader = Backtester()

# trader.ledger_update(order_id='123456', order_type='Trade', symbol='BTC-USD', amount=0.5, fee=0.0005)

# print("Backtest class imported successfully.")
