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

base_path = os.path.dirname(os.path.realpath(__file__))
crypto_bot_path = os.path.dirname(base_path)
crypto_bot_path = r"C:\Users\mauri\Documents\Trading Bot\Python\AlgoTrader"
Python_path = os.path.dirname(crypto_bot_path)
Trading_bot_path = os.path.dirname(Python_path)
Trading_path = os.path.join(Trading_bot_path,"Trading")
data_path_crypto = os.path.join(Trading_bot_path, "Data", "Cryptocurrencies")
datasets_path = os.path.join(data_path_crypto, "Datasets")
csv_dataset_path = os.path.join(datasets_path, "crypto datasets", "csv")
hdf_dataset_path = os.path.join(datasets_path, "crypto datasets", "hdf5")
DataLoader_path = os.path.join(crypto_bot_path, "Data Loader")


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
sys.path.append(DataLoader_path)

import mo_utils as utils


#################################################################################################################################################################
#
#
#                                                                  KUCOIN TRADER CLASS
#
#################################################################################################################################################################


class MainApiTrader:
    def __init__(self, exchange, coin, fiat, slippage, leverage, logger_input=None):
        
        if logger is None:
            self.logger = logger
            self.configure_logger()
        else:
            self.logger = logger_input

        config_path = utils.find_config_path()
        self.api_config = utils.read_config_file(os.path.join(config_path,"bitget_config.ini"))

        self.api_key = utils.get_config_value(self.api_config, "overview", "eins")
        self.api_secret = utils.get_config_value(self.api_config, "overview", "zwei")
        self.api_passphrase = utils.get_config_value(self.api_config, "overview", "pass")
        self.client_user_id = utils.get_config_value(self.api_config, "overview", "client_user_id")

        self.exchange = getattr(ccxt, exchange)({
             "apiKey": self.api_key, 
             "secret": self.api_secret, 
             "password": self.api_passphrase})
        #      "options":{"defaultType":"margin"}})
        # self.exchange.options["createMarketBuyOrderRequiresPrice"] = "False"

        self.coin = coin
        self.fiat = fiat

        self.ccxt_symbol = f"{coin}/{fiat}"

        self.slippage = slippage 
        self.leverage_factor = leverage

        # Ledger data 
        trading_path = utils.find_trading_path()
        self.ledger_path = os.path.join(trading_path, "trading_ledger.json")
        # self.ledger_data = self.load_ledger_data(self.ledger_path)
        # time.sleep(0.5)
        
#################################################################################################################################################################
#
#
#                                                                  TRADING FUNCTIONS
#
#################################################################################################################################################################
    
    def enter_margin_trade(self, coin=None, fiat=None, amount=None, balance_fiat = None, is_long=None, order_type=None, limit_price=None, stop_price=None, take_profit_price=None):
        try:
            #symbol
            symbol = coin + "-" + fiat

            # Create the initial margin order
            if is_long is not None:
                side = 'buy' if is_long else 'sell'
            else:
                self.logger("No side specificed aborting trade")
                return
            #function for HF:   initial_balance = self.get_margin_account_details(curr=fiat)["available"].values[0]
            initial_balance = self.get_account_list(curr=fiat, account_type="crossed_margin")["available"].values[0]
            price = self.get_price(symbol=self.ccxt_symbol)

            if side == "buy":
                available_balance = balance_fiat * self.leverage_factor
    
            else:
                balance_in_coin = balance_fiat / price

                available_balance = balance_in_coin * (self.leverage_factor -1)
            
            if funds > available_balance:
                funds = funds + (available_balance - funds)
                
           
            initial_order_id = self.execute_auto_borrow_margin_order(amount=amount, side=side, order_type=order_type) #removed price as no limit order strategy

            # Calculate the invested amount in USDT
            # if funds is None:
            #     current_price = self.get_price(symbol)
            #     invested_amount = size * ([limit_price if not np.isnull(limit_price) else current_price])
            # else:
            #     invested_amount = funds
    
            order_details = self.get_order_details_by_id(orderId=initial_order_id)
            execution_price = self.calc_price_from_details(dealFunds=order_details["dealFunds"],dealSize=order_details["dealSize"])
            fees = order_details["fee"]

            size = order_details["dealSize"]
            funds = order_details["dealFunds"]

            if side == "buy":    
                sl_size = size 
                tp_size = size 

                if size == None:
                    raise ("size is none please check and adjust invested amount calculation")
                    
            else:
                sl_size = size 
                tp_size = size

                sl_funds = funds
                tp_funds = funds 

                if funds == None:
                    raise ("funds is none please check and adjust invested amount calculation")
                                                                                                           
            # Create a stop loss order if stop_price is specified
            stop_loss_order = None
            take_profit_order = None
            
            if stop_price:
                stop_loss_side = 'sell' if is_long else 'buy'
                if side == "buy":
                    stop_loss_order_id = self.execute_stop_order(size=sl_size, order_type="market", side=stop_loss_side, StopPrice=stop_price, stop="loss")
                else:
                    stop_loss_order_id = self.execute_stop_order(funds=sl_funds, order_type="market", side=stop_loss_side, StopPrice=stop_price, stop="entry")
                    
                stop_loss_order = True

            # Create a take-profit order if specified
            if take_profit_price:
                take_profit_side = 'sell' if is_long else 'buy'
                if side =="buy":
                    take_profit_order_id = self.execute_stop_order(size=tp_size, order_type="market", side=take_profit_side, StopPrice=take_profit_price, stop="entry")
                else:
                    take_profit_order_id = self.execute_stop_order(funds=tp_funds, order_type="market", side=take_profit_side, StopPrice=take_profit_price, stop="loss")
                take_profit_order = True

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
                'timestamp_opened':             order_details["createdAt"],
                'timestamp_closed':             None,
                'time_opened':                  self.convert_timestamp_to_datetime(order_details["createdAt"]),
                'time_closed':                  None,
                'invested_usdt':                order_details["dealFunds"],
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
            
            self.update_position(position_info)
            
            
        except Exception as e:
            return f"An error occurred: {e}"
    
    def close_margin_position(self,coin=None,fiat="USDT",order_id=None, current_position=None):
        """
        closes margin orders by order id
        """
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

            risk_prevention_status = self.check_sl_or_tp_triggered(coin=coin, fiat=fiat, position=position)

            if risk_prevention_status =="not triggered":
                # Place an order to close the position, the take profit order and the stop loss order
                close_order_id, borrowSize = self.execute_auto_repay_margin_order(symbol=symbol, size=closing_size, funds=closing_funds, order_type="market", side=side)
                
                self.cancel_all_orders()

                #get close order details
                close_order_details = self.get_order_details_by_id(orderId=close_order_id)
                
                if side == "sell":
                    closing_amount_invested = close_order_details["dealFunds"] 
                    pnl_trade = closing_amount - position['invested_usdt']
                elif side =="buy":
                    closing_amount = close_order_details["dealSize"]
                    pnl_trade = closing_amount - size

                    if pnl_trade < 0:
                        rebuy_price = self.get_price(symbol=symbol) * (1 + self.slippage)
                        rebuy_order_id= self.execute_margin_order(symbol=symbol, size=pnl_trade, order_type="limit", side=side,price=rebuy_price)
                        position["rebuy_order_id"] = rebuy_order_id

                # Check if there are borrowed funds associated with this position
                # if position and borrowSize > 0:
                #     # Implement repaying logic
                #     repaid_order_id, repaid_amount = self.repay_funds(curr=borrowed_asset, symbol=position['symbol'], size=borrowSize)      #might not be borrowSize but  actual initial size
                #     self.logger.info(f"Repaid order of amount {repaid_amount}")

                #only applicable if HF trading
                #double check if all borrowed funds have been repaid
                borrowed_asset_details = self.get_margin_account_details(curr=borrowed_asset) 
                liabilities = borrowed_asset_details["liability"].values[0]
                available = borrowed_asset_details["available"].values[0]
                
                if liabilities >  0:
                    size_repay = liabilities
                    
                    if liabilities > available and side =="buy":
                        owed_amount = liabilities - available
                        rebuy_price_debt = self.get_price(symbol=symbol) * (1 + self.slippage)

                        rebuy_order_debt_id = self.execute_margin_order(symbol=symbol, size=owed_amount, order_type="limit", side=side,price=rebuy_price_debt)

                    repaid_order_id, repaid_amount = self.repay_funds(curr=borrowed_asset, size=size_repay)
                    self.logger.info(f"Repaid order of amount {repaid_amount}")
                    
                #when short convert coin into usdt
                if side == "buy":
                    coin_balance_after_trade = self.get_account_list(curr=coin, account_type="margin")["available"].values[0]
                    convert_order_id = self.execute_margin_order(symbol=symbol, size=coin_balance_after_trade, order_type="market", side="sell")#
                    position["convert_order_id"] = convert_order_id

                #get convert order details and rebuy_order_details
                convert_order_id_details = self.get_order_details_by_id(orderId=convert_order_id)
                rebuy_order_details = self.get_order_details_by_id(orderId=rebuy_order_id)

                # Update the position as closed in the ledger
                position["close_order_id"] = close_order_id
                position['status'] = 'closed'
                position['timestamp_closed'] = close_order_details["createdAt"]
                position['time_closed'] =      self.convert_timestamp_to_datetime(close_order_details["createdAt"])
                position["repaid"] = "repaid"
                
                
                if side =="buy" and pnl_trade <0:
                    fees = { "sum":     sum(close_order_details["fee"], rebuy_order_details["fee"], convert_order_id_details["fee"]), 
                            "single":   [close_order_details["fee"], rebuy_order_details["fee"], convert_order_id_details["fee"]]}
                elif side =="buy" and pnl_trade >=0:
                    fees = { "sum":     sum(close_order_details["fee"], convert_order_id_details["fee"]), 
                            "single":   [close_order_details["fee"], convert_order_id_details["fee"]]}                
                else:
                    fees = close_order_details["fee"]
                
                position["fees_closing_trade"] = fees

                closing_balances = self.get_account_list(account_type="margin")
                closing_balance_fiat = closing_balances[closing_balances["currency"]==fiat]["available"].values[0] #self.get_margin_account_details(curr=fiat)["available"].values[0]
                closing_balance_coin = closing_balances[closing_balances["currency"]==coin]["available"].values[0]
                position["closing_balance"] = closing_balance_fiat
                
                if side == "sell":
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = pnl_trade   
                else:
                    closing_amount_invested = close_order_details["dealFunds"] + convert_order_id_details["dealFunds"]
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = (closing_amount_invested - position['invested_usdt'])

                if position["pnl"] >0:
                    position["owed tax"] = position["pnl"] * 0.25
                else:
                    position["owed tax"] = {"loss":position["pnl"]}

                #set open position to None
                self.ledger_data["current_trades"][symbol] = None

                #update balances
                self.ledger_data["balances"]["trailing_balances"][fiat] = closing_balance_fiat
                self.ledger_data["balances"]["trailing_balances"][coin] = closing_balance_coin
                self.ledger_data["balances"]["trailing_balances"]["total_balance"] = self.calculate_balance_in_fiat(fiat=fiat, coin=coin, getSum=True, get_only_trade_pair_bal=True, account_type="margin")
                self.update_position(position)
                return repaid_order_id
            else:
                if risk_prevention_status == "tp_cancel":
                    closed_by = "stop loss"
                else:
                    closed_by = "take profit"
                logger.info(f"Position already closed by {closed_by}")
                
                liabilities = self.get_margin_account_details(curr=borrowed_asset)["liability"].values[0]
                
                if liabilities >  0:
                    size_repay = liabilities
                    
                    repaid_order_id, repaid_amount = self.repay_funds(curr=borrowed_asset, size=size_repay)
                    self.logger.info(f"Repaid order of amount {repaid_amount}")
                return
        
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None
        

    def get_trading_fees(self, currency_type=0):
        """
        REQ:            currencyType:       string    0: crypto-currency, 1: fiat currency   default is 0
        link:           https://www.kucoin.com/docs/rest/funding/trade-fee/basic-user-fee-spot-margin-trade_hf

        Returns:
        takerFeeRate:   base taker fee
        makerFeeRate:   base maker fee   	
        """
        try:
            symbol = self.ccxt_symbol
            self.exchange.fetch_trading_fees(symbol)
        
        except Exception as e:
            print(f"An error occurred when retrieving trading fees: {e}")
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
            print(f"An error occurred when retrieving actual trading pair fees: {e}")
            return {"error": f"An exception occurred: {str(e)}"}

    


#################################################################################################################################################################
#
#
#                                                                       Margin Order Functions (V1)
#
#################################################################################################################################################################
    

    def execute_auto_borrow_margin_order(self, amount=None, side=None, price=None, order_type="market"):
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
            amount = float(str(amount)[:7])
            differentiating_exchanges = ["KuCoin", "Bitget"]

            exchange_name = self.exchange.name

            if exchange_name in differentiating_exchanges:
                if exchange_name == "KuCoin":
                    
                    if size:
                        params["size"] =  size
                
                    if funds and order_type=="market": 
                        params["funds"] = funds

                    amount = 0
                if exchange_name == "Bitget":
                    params = {"marginMode": "cross", 
                      "loanType": "autoLoan"}
                    
                    if side == "buy" and order_type == "market":
                        params["quoteSize"] = amount
                    elif side == "sell" or order_type == "limit":
                        params["baseSize"] = amount

                    if price and order_type == "limit":
                        params["price"] = price
                    
                    amount = 0
         
            if order_type == "limit" and price:
                order  = self.exchange.create_order(symbol=self.ccxt_symbol, amount=amount,  side=side, type=order_type, price=price, params=params)
            else:
                order = self.exchange.create_order(symbol=self.ccxt_symbol, amount=amount,  price=0, side=side, type=order_type, params=params)
            
            order_id  = order["info"]["orderId"]
            # borrowSize = float(order["info"]["borrowSize"])

            return order_id 
        except Exception as e:
            print(f"error: ", f"Failed to create auto-borrow order: {e}")
            return 
        
    def execute_auto_repay_margin_order(self, amount=None, side=None, price=None, order_type="market"):
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
            amount = float(str(amount)[:7])
            differentiating_exchanges = ["KuCoin", "Bitget"]

            exchange_name = self.exchange.name

            if exchange_name in differentiating_exchanges:
                if exchange_name == "KuCoin":
                    
                    if size:
                        params["size"] =  size
                
                    if funds and order_type=="market": 
                        params["funds"] = funds

                    amount = 0
                if exchange_name == "Bitget":
                    params = {"marginMode": "cross", 
                      "loanType": "autoRepay"}
                    
                    if side == "buy" and order_type == "market":
                        params["quoteSize"] = amount
                        amount = 0
                    elif side == "sell" or order_type == "limit":
                        params["baseSize"] = amount
                        
                    if price and order_type == "limit":
                        params["price"] = price
                    
                    amount = amount  
         
            if order_type == "limit" and price:
                order  = self.exchange.create_order(symbol=self.ccxt_symbol, amount=amount,  side=side, type=order_type, price=price, params=params)
            else:
                order = self.exchange.create_order(symbol=self.ccxt_symbol, amount=amount,  price=0, side=side, type=order_type, params=params)
            
            order_id  = order["info"]["orderId"]
            # borrowSize = float(order["info"]["borrowSize"])

            return order_id
        except Exception as e:
            print(f"error: ", f"Failed to create auto-repay order: {e}")
            return 
        

    def execute_margin_order(self, symbol=None, size=None, funds=None, order_type="market", side=None, price=None):
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
            amount = float(str(amount)[:7])
            differentiating_exchanges = ["KuCoin", "Bitget"]

            exchange_name = self.exchange.name

            if exchange_name in differentiating_exchanges:
                if exchange_name == "KuCoin":
                    
                    if size:
                        params["size"] =  size
                
                    if funds and order_type=="market": 
                        params["funds"] = funds

                    amount = 0
                if exchange_name == "Bitget":
                    params = {"marginMode": "cross", 
                      "loanType": "normal"}
                    
                    if side == "buy" and order_type == "market":
                        params["quoteSize"] = amount
                    elif side == "sell" or order_type == "limit":
                        params["baseSize"] = amount

                    if price and order_type == "limit":
                        params["price"] = price
                    
                    amount = 0
         
            if order_type == "limit" and price:
                order  = self.exchange.create_order(symbol=self.ccxt_symbol, amount=amount,  side=side, type=order_type, price=price, params=params)
            else:
                order = self.exchange.create_order(symbol=self.ccxt_symbol, amount=amount,  price=0, side=side, type=order_type, params=params)
            
            order_id  = order["info"]["orderId"], 
            # borrowSize = float(order["info"]["borrowSize"])

            return order_id
        except Exception as e:
            print(f"error: ", f"Failed to create auto-repay order: {e}")
            return 
        
    
    def execute_stop_order(self, amount=None, order_type="market", side=None, price=None, sl_price=None,tp_price=None, stop=None, auto_repay=False):
        """
    REQ:    side:                      "buy" or "sell"
        REQ:    order_type:            "limit" or "market"
        REQ:   stopPrice:              String  stop price
        REQ:   is_sl:                  Price value for stop loss^
        REQ:   is_tp:                  Price value for take profit

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
            differentiating_exchanges = ["KuCoin", "Bitget"]
            params = {}      
            exchange_name = self.exchange.name

            if exchange_name in differentiating_exchanges:
                if exchange_name == "KuCoin":
                    trade_type = "MARKET_TRADE"
                    params["tradeType"] = trade_type
                    params["stopPrice"] = stopPrice
                    params["stop"] = stop
                    amount = 0

                    if size:
                        params["size"] =  size
                
                    if funds and order_type=="market": 
                        params["funds"] = funds
                        params["autoRepay"] = True

                if exchange_name == "Bitget":
                    params = {"marginMode": "cross"}

                    if sl_price:
                        params["stopLossPrice"] =  sl_price
                    elif tp_price:
                        params["takeProfitPrice"] = tp_price

                    if auto_repay: 
                        params["loanType"] = "autoRepay"

                    if side == "buy" and order_type == "market":
                        params["quoteSize"] = amount
                        amount = 0
                    elif side == "sell" or order_type == "limit":
                        params["baseSize"] = amount
                        
                    if price and order_type == "limit":
                        params["price"] = price
                    
                    amount = amount  
            else:
                if side=="buy":
                    amount = funds/self.get_price(symbol=self.ccxt_symbol)
                else: 
                    amount = size
                params = params


            if order_type == "limit" and price:
                order  = self.exchange.create_order(symbol=self.ccxt_symbol, amount=amount,  side=side, type=order_type, price=price, params=params)
            else:
                order = self.exchange.create_order(symbol=self.ccxt_symbol, amount=amount,  price=0, side=side, type=order_type, params=params)
            
            order_id  = order["info"]["orderId"]
            
            
                # amount = 0 

            
            order = self.exchange.create_order(symbol=self.ccxt_symbol, amount=amount, side=side, type=order_type, params=params)
            
            order_id  = order["info"]["orderId"]
        

            return order_id
        except Exception as e:
            print(f"error: ", f"Failed to create margin order: {e}")
            return
        

    def cancel_order_by_order_id(self, order_id=None,is_stop=False, tradeType ="MARGIN_TRADE"):
        """
        REQ:        symbol:         e.g. BTC-USDT
        REQ:        orderId         String

        Returns:
        orderId:                orderId of the cancelled order
        """
        params = {"tradeType": tradeType} 
        symbol = self.ccxt_symbol

        try:
            if is_stop:
                params["stop"] = True
                
            cancelled_order = self.exchange.cancelOrder(order_id, params=params)

            return cancelled_order
        except Exception as e:
            print(f"An error occurred while cancelling order {order_id}: {e}")

    
    def cancel_all_orders(self, order_ids=None, is_stop=None, tradeType="MARGIN_TRADE"):
        """
        REQ:        symbol      e.g. BTC-USDT
        
        Kucoint API:
        Link:       https://www.kucoin.com/docs/rest/spot-trading/orders/cancel-order-by-orderid

        Returns:
        orderId:                orderId of the cancelled order
        """
        try:            
            symbol = self.ccxt_symbol
            params = {"tradeType": tradeType}

            if is_stop:
                params["stop"] = True

            if order_ids:
                params["orderIds"] = order_ids

            cancelled_orders = self.exchange.cancel_all_orders(symbol=symbol, params=params)
            
            return cancelled_orders            
        except Exception as e:
            print(f"An error occurred while cancelling all orders: {e}")


    def check_order_triggered(self, order_id):
        """
        REQ:        order_id.       individual orderId by trade
        REQ:        symbol:         e.g. BTC-USDT
        
        Returns:    True if order still active, False otherwise
        """
        try:
            ordder_details = self.get_order_details_by_id(orderId=order_id)

            if ordder_details['isActive']:
                return False  # Order is still active, not triggered
            else:
                return True  # Order has been executed or cancelled, assuming triggered

        except Exception as e:
            print(f"An error occurred while checking order status: {e}")
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
        
        order_id = position["order_id"]
        symbol = position["pair"]
        side_initial_position = position["side"]
        size = position['size']
        funds = position['funds']

        stop_loss_order_id = position.get('stop_loss_order_id')
        take_profit_order_id = position.get('take_profit_order_id')

        check_initial_order_triggered = self.check_order_triggered(order_id=order_id)
        
        
        check_stop_loss_triggered   = self.check_order_triggered(stop_loss_order_id)
        check_take_profit_triggered = self.check_order_triggered(take_profit_order_id)
        
        if check_initial_order_triggered and check_stop_loss_triggered:
            try:
                #cancel tp order
                try:
                    self.cancel_order_by_order_id(order_id=take_profit_order_id, is_stop=True)
                except Exception as e:
                    print(f"An error occurred while cancelling take profit order: {e}")
                    print("Cancelling all open orders")
                    self.cancel_all_orders(is_stop=True)

                sl_order_details =self.get_order_details_by_id(stop_loss_order_id)
            
                #timestamp_now = datetime.now().timestamp()*1000    check later in backtest if createdAt is timestamp when order was filled else use timestamp now
                
                #update position respectively
                side = sl_order_details["side"]
                
                if side == "sell":
                    borrowed_asset = fiat
                    borrowed_amount = funds
                    closing_amount_invested = sl_order_details["dealFunds"] 
                    pnl_trade = closing_amount - position['invested_usdt']
                elif side =="buy":
                    borrowed_asset = coin
                    borrowed_amount = size
                    closing_amount = sl_order_details["dealSize"]
                    pnl_trade = closing_amount - size

                    if pnl_trade < 0:
                        rebuy_price = self.get_price(symbol=symbol) * (1 + self.slippage)
                        rebuy_order_id = self.execute_margin_order(symbol=symbol, size=pnl_trade, order_type="limit", side=side,price=rebuy_price)
                        position["rebuy_order_id"] = rebuy_order_id

                borrowed_asset_details = self.get_margin_account_details(curr=borrowed_asset) 
                liabilities = borrowed_asset_details["liability"].values[0]
                available = borrowed_asset_details["available"].values[0]
                
                if liabilities >  0:
                    size_repay = liabilities
                    
                    if liabilities > available and side =="buy":
                        owed_amount = liabilities - available
                        rebuy_price_debt = self.get_price(symbol=symbol) * (1 + self.slippage)

                        rebuy_order_debt_id = self.execute_margin_order(symbol=symbol, size=owed_amount, order_type="limit", side=side,price=rebuy_price_debt)

                    repaid_order_id, repaid_amount = self.repay_funds(curr=borrowed_asset, size=size_repay)
                    self.logger.info(f"Repaid order of amount {repaid_amount}")
                    
                #when short convert coin into usdt
                if side == "buy":
                    coin_balance_after_trade = self.get_account_list(curr=coin, account_type="margin")["available"].values[0]
                    convert_order_id = self.execute_margin_order(symbol=symbol, size=coin_balance_after_trade, order_type="market", side="sell")#
                    position["convert_order_id"] = convert_order_id

                #get convert order details and rebuy_order_details
                convert_order_id_details = self.get_order_details_by_id(orderId=convert_order_id)
                rebuy_order_details = self.get_order_details_by_id(orderId=rebuy_order_id)

                #update position
                position["close_order_id"] = stop_loss_order_id
                position['status'] = 'closed'
                position['timestamp_closed'] = sl_order_details["createdAt"]
                position['time_closed'] =      self.convert_timestamp_to_datetime(sl_order_details["createdAt"])
                position["repaid"] = "repaid"
                
                if side =="buy" and pnl_trade <0:
                    fees = { "sum":     sum(sl_order_details["fee"], rebuy_order_details["fee"], convert_order_id_details["fee"]), 
                            "single":   [sl_order_details["fee"], rebuy_order_details["fee"], convert_order_id_details["fee"]]}
                elif side =="buy" and pnl_trade >=0:
                    fees = { "sum":     sum(sl_order_details["fee"], convert_order_id_details["fee"]), 
                            "single":   [sl_order_details["fee"], convert_order_id_details["fee"]]}                
                else:
                    fees = sl_order_details["fee"]
                
                position["fees_closing_trade"] = fees

                closing_balances = self.get_account_list(account_type="margin")
                closing_balance_fiat = closing_balances[closing_balances["currency"]==fiat]["available"].values[0] #self.get_margin_account_details(curr=fiat)["available"].values[0]
                closing_balance_coin = closing_balances[closing_balances["currency"]==coin]["available"].values[0]
                position["closing_balance"] = closing_balance_fiat

                if side == "sell":
                    closing_amount_invested = sl_order_details["dealFunds"]
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = (closing_amount_invested - position['invested_usdt'])    #position['closing_balance']
                else:
                    closing_amount_invested = sl_order_details["dealFunds"] + convert_order_id_details["dealFunds"]
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = (closing_amount_invested - position['invested_usdt'])
                
                if position["pnl"] >0:
                    position["owed tax"] = position["pnl"] * 0.25
                else:
                    position["owed tax"] = {"loss":position["pnl"]}

                #set open position to None
                self.ledger_data["current_trades"][symbol] = None
                
                #update balances
                self.ledger_data["balances"]["trailing_balances"][self.fiat] = closing_balance_fiat
                self.ledger_data["balances"]["trailing_balances"][self.coin] = closing_balance_coin
                self.ledger_data["balances"]["trailing_balances"]["total_balance"] = self.calculate_balance_in_fiat(fiat=fiat, coin=coin, getSum=True, get_only_trade_pair_bal=True, account_type="margin")
                
                self.update_position(position)
                return str("tp_cancel")
            except:
                print("take profit order could not be canceled")
        
        elif check_initial_order_triggered and check_take_profit_triggered:
            try:
                #cancel sl order
                try:
                    self.cancel_order_by_order_id(order_id=take_profit_order_id, is_stop=True)
                except Exception as e:
                    print(f"An error occurred while cancelling take profit order: {e}")
                    print("Cancelling all open orders")
                    self.cancel_all_orders(is_stop=True)

                tp_order_details =self.get_order_details_by_id(take_profit_order_id)

                #update position respectively
                side = tp_order_details["side"]
                
                if side == "sell":
                    borrowed_asset = fiat
                    borrowed_amount = funds
                    closing_amount_invested = tp_order_details["dealFunds"] 
                    pnl_trade = closing_amount - position['invested_usdt']
                elif side =="buy":
                    borrowed_asset = coin
                    borrowed_amount = size
                    closing_amount = tp_order_details["dealSize"]
                    pnl_trade = closing_amount - size

                    if pnl_trade < 0:
                        rebuy_price = self.get_price(symbol=symbol) * (1 + self.slippage)
                        rebuy_order_id = self.execute_margin_order(symbol=symbol, size=pnl_trade, order_type="limit", side=side,price=rebuy_price)
                        position["rebuy_order_id"] = rebuy_order_id

                borrowed_asset_details = self.get_margin_account_details(curr=borrowed_asset) 
                liabilities = borrowed_asset_details["liability"].values[0]
                available = borrowed_asset_details["available"].values[0]
                
                if liabilities >  0:
                    size_repay = liabilities
                    
                    if liabilities > available and side =="buy":
                        owed_amount = liabilities - available
                        rebuy_price_debt = self.get_price(symbol=symbol) * (1 + self.slippage)

                        rebuy_order_debt_id = self.execute_margin_order(symbol=symbol, size=owed_amount, order_type="limit", side=side,price=rebuy_price_debt)

                    repaid_order_id, repaid_amount = self.repay_funds(curr=borrowed_asset, size=size_repay)
                    self.logger.info(f"Repaid order of amount {repaid_amount}")
                    
                #when short convert coin into usdt
                if side == "buy":
                    coin_balance_after_trade = self.get_account_list(curr=coin, account_type="margin")["available"].values[0]
                    convert_order_id = self.execute_margin_order(symbol=symbol, size=coin_balance_after_trade, order_type="market", side="sell")#
                    position["convert_order_id"] = convert_order_id

                #get convert order details and rebuy_order_details
                convert_order_id_details = self.get_order_details_by_id(orderId=convert_order_id)
                rebuy_order_details = self.get_order_details_by_id(orderId=rebuy_order_id)

                position["close_order_id"] = position.get("take_profit_order_id")
                position['status'] = 'closed'
                position['timestamp_closed'] = tp_order_details["createdAt"]
                position['time_closed'] =      self.convert_timestamp_to_datetime(tp_order_details["createdAt"])
                position["repaid"] = "repaid"
                
                if side =="buy" and pnl_trade <0:
                    fees = { "sum":     sum(sl_order_details["fee"], rebuy_order_details["fee"], convert_order_id_details["fee"]), 
                            "single":   [sl_order_details["fee"], rebuy_order_details["fee"], convert_order_id_details["fee"]]}
                elif side =="buy" and pnl_trade >=0:
                    fees = { "sum":     sum(sl_order_details["fee"], convert_order_id_details["fee"]), 
                            "single":   [sl_order_details["fee"], convert_order_id_details["fee"]]}                
                else:
                    fees = sl_order_details["fee"]
                
                position["fees_closing_trade"] = fees

                closing_balances = self.get_account_list(account_type="margin")
                closing_balance_fiat = closing_balances[closing_balances["currency"]==fiat]["available"].values[0] #self.get_margin_account_details(curr=fiat)["available"].values[0]
                closing_balance_coin = closing_balances[closing_balances["currency"]==coin]["available"].values[0]
                position["closing_balance"] = closing_balance_fiat

                if side == "sell":
                    closing_amount_invested = tp_order_details["dealFunds"]
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = (closing_amount_invested - position['invested_usdt'])    #position['closing_balance']
                else:
                    closing_amount_invested = tp_order_details["dealFunds"] + convert_order_id_details["dealFunds"]
                    position["closing_amount_invested"] = closing_amount_invested
                    position["pnl"] = (closing_amount_invested - position['invested_usdt'])
                
                if position["pnl"] >0:
                    position["owed tax"] = position["pnl"] * 0.25
                else:
                    position["owed tax"] = {"loss":position["pnl"]}

                #set open position to None
                self.ledger_data["current_trades"][symbol] = None
                
                #update balances
                self.ledger_data["balances"]["trailing_balances"][self.fiat] = closing_balance_fiat
                self.ledger_data["balances"]["trailing_balances"][self.coin] = closing_balance_coin
                self.ledger_data["balances"]["trailing_balances"]["total_balance"] = self.calculate_balance_in_fiat(fiat=fiat, coin=coin, getSum=True, get_only_trade_pair_bal=True, account_type="margin")
                
                self.update_position(position)
                return str("sl_cancel")
            except:
                print("Stop loss order could not be canceled")
    
        elif check_initial_order_triggered and not check_stop_loss_triggered and not check_take_profit_triggered:
            return str("not triggered")


    def get_order_details_list(self, orderId=None, get_open_orders=False, side=None, is_stop=False, tradeType="MARGIN_TRADE", type=None, startAt=None,endAt=None):
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
            symbol = self.ccxt_symbol
            
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
            return df 
            
        except Exception as e:
            print(f"An error occurred when retrieving order history: {e}")
            return {"error": f"An exception occurred: {str(e)}"}
        
    def get_order_details_by_id(self, orderId=None, is_stop=False, tradeType="MARGIN_TRADE"):
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
            symbol = self.ccxt_symbol
            params={"tradeType":tradeType}
            
            if tradeType and tradeType != "MARGIN_TRADE":
                params = {"tradeType":tradeType}

            if is_stop:
                params["stop"] = is_stop

            while order_details is None and retry_count< max_retries:
                order_details = self.exchange.fetchOrder(orderId, symbol, params)

            df = pd.DataFrame(order_details["info"], index=[0])
            
            if is_stop:
                if df["side"].values[0] == "buy":
                        df = df.rename(columns={"takerFeeRate":"fee"})
                else:
                    df = df.rename(columns={"makerFeeRate":"fee"})
                df = df.rename(columns={"funds":"dealFunds", "size":"dealSize"})

            return df
           
        except Exception as e:
            print(f"An error occurred when retrieving order details: {e}")
            return {"error": f"An exception occurred: {str(e)}"}
  

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
                response_df = pd.DataFrame(response_json["data"])

                return response_df
            else:
                return {"error": f"Failed to fetch data: {response.status_code}", "details": response_json}
        
        except Exception as e:
            print(f"An error occurred for getting the filled orders: {e}")
            return {"error": f"An exception occurred: {str(e)}"}
        
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


    def calc_price_from_details(self, dealSize=None, dealFunds=None):

        price = dealFunds / dealSize  
        price = round(price, 5)
        return price

    

#################################################################################################################################################################
#
#
#                                                                  API FUNCTIONS
#
#################################################################################################################################################################


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


     
    # def create_sign_in(self, api_key, api_secret, api_passphrase, method, url, endpoint, data='', params=None):
    #     """
    #     Generates the headers required for Kucoin API requests.

    #     :param api_key: Your Kucoin API key.
    #     :param api_secret: Your Kucoin API secret.
    #     :param api_passphrase: Your Kucoin API passphrase.
    #     :param method: The HTTP method (e.g., 'GET', 'POST').
    #     :param endpoint: The API endpoint (e.g., '/api/v1/accounts').
    #     :param data: The request body, if any (for POST requests).
    #     :return: A dictionary containing the headers.
    #     """
    #     now = int(time.time() * 1000)
    #     # now = utils.get_ntp_time(in_ms=True)
    #     str_to_sign = str(now) + method + endpoint

    #     if method in ['GET', 'DELETE'] and params:
    #         # Sort and encode parameters
    #         sorted_params = sorted(params.items())
    #         #query_string = '&'.join([f"{key}={urllib.parse.quote_plus(str(value))}" for key, value in sorted_params])
    #         encoded_query_string = urllib.parse.urlencode(sorted_params)
    #         str_to_sign += '?' + encoded_query_string  # Add '?' before query string
    #     elif data:
    #         str_to_sign += data

    #     signature = base64.b64encode(hmac.new(api_secret.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest()) #.decode()

    #     passphrase = base64.b64encode(hmac.new(api_secret.encode('utf-8'), api_passphrase.encode('utf-8'), hashlib.sha256).digest()) #.decode()

    #     headers = {
    #         "KC-API-SIGN": signature,
    #         "KC-API-TIMESTAMP": str(now),
    #         "KC-API-KEY": api_key,
    #         "KC-API-PASSPHRASE": passphrase,
    #         "KC-API-KEY-VERSION": "2"
    #     }

    #     if method == 'POST':
    #         headers["Content-Type"] = "application/json"

    #     return headers

    def make_request(self, method, endpoint, params=None, body=None):
        timestamp = str(int(time.time() * 1000))  # Milliseconds
        headers = self._generate_headers(method, endpoint, timestamp, body)
        base_url = "https://api.bitget.com"
        url = base_url + endpoint

        method = method.upper()  # Ensure method is uppercase
        
        # Using requests.request to simplify handling of different methods
        
        if method == "GET":
            url += '?' + urllib.parse.urlencode(params)
            response = requests.get(url, headers=headers, params=params)
        else:
            data = json.dumps(body) if body and method == "POST" else None
            response = requests.request(method, url, headers=headers, params="", data=data)
        
        return response.json()


    def _generate_headers(self, method, endpoint, timestamp, body, params=None):
        # For GET requests, body should be None and not used in the signature
        body_str = json.dumps(body) if body and method == "POST" else ''
        
        # For GET requests with parameters, ensure they are part of the endpoint in the pre-sign string
        if method == 'GET' and params:
            encoded_params = urllib.parse.urlencode(sorted(params.items()))
            pre_sign_string = f"{timestamp}{method}{endpoint}?{encoded_params}"
        else:
            pre_sign_string = f"{timestamp}{method}{endpoint}{body_str}"
        
        
        signature = hmac.new(self.api_secret.encode('utf-8'), pre_sign_string.encode('utf-8'), hashlib.sha256).digest()
        signature_b64 = base64.b64encode(signature).decode()

        headers = {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature_b64,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.api_passphrase,
            "Content-Type": "application/json"}
        
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
        elif isinstance(datetime_input, dt.datetime):
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

    def inner_transfer(self, currency=None, amount=None, from_acc=None, to_acc=None):
        """
        REQ:    currency:   Currency e.g. BTC
        REQ:    from:       Payment Account Type: spot, p2p, coin_futures, usdt_futures, crossed_margin, isolated_margin
        REQ:    to:         Receiving Account Type: spot, p2p, coin_futures, usdt_futures, crossed_margin, isolated_margin
        REQ:    amount:     Transfer amount, the precision being a positive integer multiple of the Currency Precision

        OPT:    from-tag:   Trading pair, required when the payment account type is isolated, e.g.: BTC-USDT
        OPT:    to_tag:     Trading pair, required when the payment account type is isolated, e.g.: BTC-USDT
        
        """
        try:
            endpoint = "/api/v2/spot/wallet/transfer"

            params = {"fromType": from_acc, "toType": to_acc, "amount": amount,"coin": currency}

            transfer = self.exchange.transfer(code=currency, amount=amount, fromAccount=from_acc, toAccount=to_acc, params=params)   #
            
            # if from_acc == "isolated_margin" or to_acc == "isolated_margin" or from_acc == "cross_margin" or to_acc == "cross_margin":
            #     if coin != "USDT": 
            #         params["symbol"] = currency+"USDT"
            #     else:
            #         params["symbol"] = "BTCUSDT"

            # response_json = self.make_request('POST', endpoint, body=params)

            # # Check if response is successful
            # if response_json["msg"] == "success":
            #     self.logger.info(f"Successfully transferred {amount} {currency} from {from_acc} to {to_acc}")

        except Exception as e:
            print(f"An error occurred when transferring funds: {e}")
            return {"error": f"An exception occurred: {str(e)}"}

        


    def get_account_list(self, curr=None, account_type=None):
        """
        OPT:            currency:       e.g. KCS
        OPT:            type:           spot, cross_margin, isolated_margin

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
            if account_type is None or account_type == "spot":
                endpoint = "/api/v2/spot/account/assets"
                params = {}

                if curr is not None:
                    params["coin"] = coin

                response_json = self.make_request("GET", endpoint, params=params)

                # Check if response is successful
                if response_json["msg"] == "success":
                    df = pd.DataFrame(response_json["data"])
                    if len(df) == 0:
                        return 0
                    else:
                        df[["available","frozen", "locked"]] = df[["available","frozen", "locked"]].astype(float)
                    
                        
            elif account_type == "crossed_margin":
                
                endpoint = "/api/v2/margin/crossed/account/assets"
                params = {}

                # if curr is not None:
                #     params["coin"] = coin

                response_json = self.make_request("GET", endpoint, params=params)

                # Check if response is successful
                if response_json["msg"] == "success":
                    df = pd.DataFrame(response_json["data"])
                    if len(df) == 0:
                        return 0
                    else:
                        df[["totalAmount","available","frozen", "borrow", "interest", "net", "coupon"]] = df[["totalAmount","available","frozen", "borrow", "interest", "net", "coupon"]].astype(float)
                    

            elif account_type == "isolated_margin":
                endpoint = "/api/v2/margin/isolated/account/assets"
                params = {}

                if curr is not None:
                    params["coin"] = coin

                response_json = self.make_request("GET", endpoint, params=params)

                # Check if response is successful
                if response_json["msg"] == "success":
                    df = pd.DataFrame(response_json["data"])
                    if len(df) == 0:
                        return 0
                    else:
                        df[["totalAmount","available","frozen", "borrow", "interest", "net", "coupon"]] = df[["totalAmount","available","frozen", "borrow", "interest", "net", "coupon"]].astype(float)
                
            elif account_type == "swap":
                marketType = "swap"
            elif account_type == "futures":
                marketType = "futures"

            if curr is not None:
                df = df[df.coin == curr]
            
            return df
                    
        except Exception as e:
            print(f"An error occurred: {e}")
            return 0

    ########    ONLY WORKS FOR SPOT    ########
    # def get_account_balance(self, coin =None, margin_mode=None):
    #     """
    #     REQ:            accountid:       string
        

    #     Returns:
    #     currency:   	The currency of the account
    #     balance:    	Total funds in the account
    #     holds:      	Funds on hold (not available for use)
    #     available:  	Funds available to withdraw or trade
    #     """
    #     try:
    #         params = {}
    #         if coin is not None:
                
    #             params["coin"] = coin
    #         # if margin_mode:
    #         #     params["marginMode"] = margin_mode
            
    #         balance = self.exchange.fetchBalance()
    #         balance = pd.DataFrame(balance["info"])
    #         return balance
    #     except Exception as e:
    #         print(f"An error occurred when retrieving account details: {e}")
    #         return {"error": f"An exception occurred: {str(e)}"}
        
        
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
                if coin != fiat:
                    symbol = coin + "-" + fiat
                    asset_price = self.get_price(symbol)
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


    def borrow_funds(self, currency=None, size =None, symbol=None, isisolated=None):
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
            symbol = self.ccxt_symbol

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



    def repay_funds(self, curr = None, size =None, symbol=None, isisolated=None):
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
            symbol = self.ccxt_symbol
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

            borrow_order = self.exchange.repayCrossMargin(curr, amount, params)
            borrow_order_id = borrow_order["id"]
            borrow_amount = borrow_order["amount"]

            return borrow_order_id,borrow_amount
        
        except Exception as e:
            print(f"An error occurred when repaying funds of size {size}.: {e}")
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

    def check_if_is_still_borrowing(self, coin, fiat, symbol=None):
        """
        checks borrowing and repayment history for order id of trading pair and identifies if it still borrowing.
        
        REQ:                currency
        OPT:                if isosolated: provide symbol

        Returns: 
        True, False:        True if still borrowing, False if not borrowing
        Amount:             Borrowed amount if still borrowing, else 0
        
        """

        borrow_histo_fiat = self.get_margin_borrowing_history(coin=fiat)
        borrow_histo_coin = self.get_margin_borrowing_history(coin=coin)
        
        repay_histo_fiat = self.get_margin_repayment_history(coin=fiat)
        repay_histo_coin = self.get_margin_repayment_history(coin= coin)

        if borrow_histo_fiat["totalNum"] ==0 and borrow_histo_coin["totalNum"] ==0:
            return False
        else:
            
            borrowed_amount_fiat = 0
            repaid_amount_fiat = 0

            borrowed_amount_coin = 0 
            repaid_amount_coin = 0 

            for item in borrow_histo_fiat['items']:
                if item["currency"] == fiat and item["status"]=="SUCCESS":
                    borrowed_amount_fiat += float(item["actualSize"])
            for item in borrow_histo_coin["items"]:   
                if item["currency"] == coin and item["status"]=="SUCCESS":
                    borrowed_amount_coin += float(item["actualSize"])

            for item in repay_histo_fiat['items']:
                if item["currency"] == fiat and item["status"]=="SUCCESS":
                    repaid_amount_fiat += float(item["principal"])
            
            for item in repay_histo_coin["items"]:
                if item["currency"] == coin and item["status"]=="SUCCESS":
                    repaid_amount_coin += float(item["principal"])

            outstanding_borrowed_amount_fiat = borrowed_amount_fiat - repaid_amount_fiat
            outstanding_borrowed_amount_coin = borrowed_amount_coin - repaid_amount_coin

            if outstanding_borrowed_amount_fiat >0 and outstanding_borrowed_amount_coin<=0:
                self.logger(f"Borrowed fiat amount is {outstanding_borrowed_amount_fiat}")
                return outstanding_borrowed_amount_fiat
            elif outstanding_borrowed_amount_coin >0 and outstanding_borrowed_amount_fiat<=0:
                self.logger(f"Borrowed coin amount is {outstanding_borrowed_amount_coin}")
                return outstanding_borrowed_amount_coin
            elif outstanding_borrowed_amount_fiat>0 and outstanding_borrowed_amount_coin>0:
                self.logger(f"Borrowed fiat amount is {outstanding_borrowed_amount_fiat}, borrowed coin amount is {outstanding_borrowed_amount_coin}")
                return outstanding_borrowed_amount_fiat, outstanding_borrowed_amount_coin
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
        if not os.path.exists(filepath):
            ledger = {"positions": [], "balances": {"initial_balance":{}, "trailing_balances":{}, "total_balance":0}, "current_trades":{}}

            account_list_margin_hf = self.get_margin_account_details(accounts=True)
            account_list_margin_hf = account_list_margin_hf[account_list_margin_hf["available"]>0][["currency","total","available"]]

            for crypto in account_list_margin_hf["currency"]: 
                ledger["balances"]["initial_balance"][crypto] = account_list_margin_hf[account_list_margin_hf["currency"]==crypto]["available"]
            return ledger 
        with open(filepath, "r") as file:
            return json.load(file)
        
    def save_data(self, filepath):
        with open(filepath, "w") as file:
            json.dump(self.ledger_data, file, indent=4)

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

# Usage example
exchange = 'bitget'
coin = "MATIC"
fiat = "USDT"
slippage = 0.01
leverage = 2

trader = MainApiTrader(exchange, coin, fiat,slippage, leverage)

symbol = coin + "-"+ fiat
side = "buy"
funds = float(15)
amount = float(15)
size = 1
stopPrice= 12



# balance = trader.get_account_list(account_type="spot")


balance_matic = trader.get_account_list(curr="MATIC", account_type="crossed_margin")["totalAmount"].values[0]   #, 
balance_usdt = trader.get_account_list(curr="USDT", account_type="crossed_margin")["totalAmount"].values[0]
print("MATIC", balance_matic, "USDT", balance_usdt)
# balance = balance[balance["coin"]=="USDT"]["available"].values.astype(float)[0]

# transfer_to_spot = trader.inner_transfer(currency="USDT", amount=balance, from_acc="spot", to_acc="crossed_margin")   #  USDT-M  # BTC/USDT:USDT   #spot
# transfer_to_margin = trader.inner_transfer(currency="USDT", amount=balance, from_acc="crossed_margin", to_acc="spot")

# current_price = trader.get_price(symbol="BTC/USDT")

# # test_limit_order = trader.execute_margin_order(symbol=symbol, size=0.5, order_type="limit", side=side,price=14.39)
current_price = trader.get_price(symbol=symbol)
sl_price = current_price * (1 - slippage)
tp_price  = current_price * (1 + slippage)
stop_order =  trader.execute_stop_order(amount=balance_usdt, order_type="market", side=side, sl_price=sl_price, auto_repay=False)

trader.execute_auto_borrow_margin_order(amount=amount, side=side)

oppo_side = "sell" if side == "buy" else "buy"
balance_matic = trader.get_account_list(curr="MATIC", account_type="crossed_margin")["totalAmount"].values[0] 
trader.execute_auto_repay_margin_order(amount=balance_matic, side=oppo_side)



# ####################################  Transfer funds to high_frequency acc. 

# # trader.inner_transfer(currency="USDT", from_acc="margin", to_acc="margin_v2", amount="42")
# # trader.inner_flex_transfer(currency="USDT",  amount=1, from_acc="TRADE_HF", to_acc="MARGIN")
# # trader.inner_flex_transfer(currency="USDT",  amount=2, from_acc="MARGIN", to_acc="MAIN")

# # acc_list = trader.get_account_list()  #curr="BTC", type="MARGIN"  type="margin"
# # acc_details = trader.get_account_details("64f22f63e9059c0007c2122a")
# # margin_account_list = trader.get_margin_account_details()
# # margin_avax_balance = trader.get_margin_account_details(curr="AVAX")
# # usd_price = trader.get_usdt_price(base ="USD", currencies="BTC")
# usdt_price =  trader.get_price("BTC-USDT")
# eur_price = trader.get_price("BTC-EUR")

# calc_fiat = trader.calculate_balance_in_fiat(fiat="USDT", getSum=True)
# calc_usdt_sum = trader.calculate_balance_in_fiat(coin=coin, fiat="USDT", getSum=True, account_type="margin")
# calc_usdt_individual = trader.calculate_balance_in_fiat(coin=coin, fiat="USDT", account_type="margin", get_only_trade_pair_bal=True)


# # open_orders = trader.get_open_hf_orders(symbol="QNT-USDT", type="MARGIN_TRADE")


# margin_borrow_histo = trader.get_margin_borrowing_history("USDT")
# margin_repay_histo = trader.get_margin_repayment_history("USDT")
# check = trader.check_if_is_still_borrowing("SHIB", "USDT")

# # margin_details = trader.get_margin_account_details()
# # taker_fee, maker_fee = trader.get_trading_fees(0)
# # act_taker_fee, act_maker_fee= trader.get_actual_trading_fees(symbol)

# # margin_details_fiat = trader.get_margin_account_details(curr="USDT")["available"]
# # margin_details_coin = trader.get_margin_account_details(curr=coin)["available"]

# # get_account_details = trader.get_account_list(type="margin")
# # test_order_id, test_order_borrow_size = trader.execute_auto_borrow_margin_order_test(symbol=symbol, funds=funds, side="buy")
# # test_order_details_hf = trader.get_hf_order_details_by_orderId(order_id="65f59247b7253400073ff561", symbol=symbol)

# # filled_list = trader.get_filled_order_list()

# date_string = "2024-04-04 00:00"
# date_ob = dt.datetime(2024,3,12,0,0)

# timestamp_1 = trader.convert_datetime_to_timestamp(date_string)
# timestamp_2 = trader.convert_datetime_to_timestamp(date_ob)

# date = trader.convert_timestamp_to_datetime(datetime.now().timestamp()*1000)


# trader.cancel_all_orders()
# stop_buy_id = trader.execute_stop_order(funds=funds, order_type="market", side="buy", stopPrice=stopPrice, stop="entry")
# al_order_details = trader.get_order_details_list(startAt=date_string)
# # stop_buy_id = 'vs8hopgiha2qbnj4003upuu3'
# print(stop_buy_id) 
# order_details_margin = trader.get_order_details_by_id(stop_buy_id)
# order_details_stop_margin = trader.get_order_details_by_id(stop_buy_id, is_stop=True)
# # order_details_margin = trader.get_order_details_by_id(stop_buy_id)
# cancel_stop = trader.cancel_order_by_order_id(stop_buy_id)
# order_details_margin_active = trader.get_order_details_by_id(active=True)
# stop_buy_details = trader.get_stop_order_details(stop_buy_id)
# check = trader.check_order_triggered(stop_buy_id, is_stop_order=True)
# trader.cancel_stop_order()




# test_auto_borrow_order_id, borrowSize = trader.execute_auto_borrow_margin_order(symbol=symbol, funds=funds, side="buy")

# time.sleep(1)
# order_details_margin = trader.get_order_details_by_id(orderId=test_auto_borrow_order_id)
# print(current_price)
# print(order_details_margin["dealFunds"]/order_details_margin["dealSize"])

# cancel_all_test = trader.cancel_all_orders()
# cancel_test = trader.cancel_order_by_order_id('65f5c6394a3a69000761f39d')



# debt = trader.get_margin_account_details() #curr="PYTH"
# debt = debt[debt.liability>0]
# debt.set_index("currency", inplace=True)
# liab = debt.loc["PYTH", "liability"]
# trader.repay_funds(curr="PYTH",size=liab)
# test_order_details = trader.get_order_details_by_id(orderId='65f5c6394a3a69000761f39d')





# test_order_details = trader.get_order_details_by_id_by_orderId('65f59247b7253400073ff561')   #'65f59247b7253400073ff561'


# test_if_is_still_borrowing = trader.check_if_is_still_borrowing(currency="AVAX")

# filled_orders = trader.get_filled_hf_orders_list("BTC-USDT", trade_type="MARGIN_TRADE")
# test = trader.get_account_list(curr="QNT",type="trade")["available"]


# test_order = trader.execute_auto_borrow_hf_margin_order_test(symbol, size=size, order_type="market", side=side)
print("API Class successfully initialized")