# import ta
import talib as ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import configparser
import os 
import json
from matplotlib.lines import Line2D
import pandas_ta as pta
from scipy.signal import argrelextrema
import requests
import time
import matplotlib.gridspec as gridspec 
from datetime import timedelta

# technical indicators 
from ta.trend import MACD, EMAIndicator, ADXIndicator, AroonIndicator, DPOIndicator, CCIIndicator, AroonIndicator, MACD, ADXIndicator, EMAIndicator, SMAIndicator, IchimokuIndicator, KSTIndicator, MassIndex, PSARIndicator, TRIXIndicator, VortexIndicator
from ta.momentum import RSIIndicator, UltimateOscillator, WilliamsRIndicator, AwesomeOscillatorIndicator, KAMAIndicator, PercentagePriceOscillator
from ta.volatility import BollingerBands,  AverageTrueRange, DonchianChannel, KeltnerChannel, UlcerIndex
from ta.others import DailyReturnIndicator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, ForceIndexIndicator, NegativeVolumeIndexIndicator ,VolumePriceTrendIndicator, VolumeWeightedAveragePrice, ChaikinMoneyFlowIndicator, EaseOfMovementIndicator, ForceIndexIndicator





class Histo_Loader:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.pairs = {key.upper(): value for key, value in self.config.items('pairs')}
        self.intervals = self.config.get('Interval', 'intervals').split(',')
        self.currency = self.config.get('trading_settings', 'currency')
        self.macd_set = self._extract_macd_settings()
        self.rsi_set = list(map(int, self.config.get('indicator_settings', 'rsi').split(',')))
        self.sma_set = list(map(int, self.config.get('indicator_settings', 'sma').split(',')))
        self.ema_set = list(map(int, self.config.get('indicator_settings', 'ema').split(','))) 
        self.bb_set = list(map(int, self.config.get('indicator_settings', 'bb').split(',')))
        self.eom_set = int(self.config.get('indicator_settings', 'eom'))
        self.atr_set = int(self.config.get('indicator_settings', 'atr'))
        self.adx_set = int(self.config.get('indicator_settings', 'adx')) 
        self.start_date = self.config.get('dates', 'start_date')
        self.end_date = self.config.get('dates', 'end_date')
        # self.quandl_api_key = self.config.get('quandl', 'api_key') # Added
        self.pair_data = {}  
        self.filepath = self.config.get('paths', 'data')
        
        # Loop through each pair and interval, fetch data, calculate indicators, and save to CSV
        for pair, symbol in self.pairs.items():
            print(pair)
            for interval in self.intervals:
                print(interval)
                df = self.get_historical_data(symbol, interval, self.start_date, self.end_date)
                if df is not None:
                    
                    self.pair_data[(pair, interval)] = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                    
                    for rsi_period in self.rsi_set:
                        if len(df) >= rsi_period:  
                            rsi = RSIIndicator(df['Close'], rsi_period)
                            df[f'rsi_{rsi_period}'] = rsi.rsi()
                        else:
                            df[f'rsi_{rsi_period}'] = np.nan

                    for strategy, periods in self.macd_set.items():
                        if len(df) >= max(periods):  # Assuming all periods are window sizes
                            macd = MACD(df['Close'], window_slow=periods[0], window_fast=periods[1], window_sign=periods[2])
                            df[f'macd_{strategy}_line'] = macd.macd()
                            df[f'macd_{strategy}_signal'] = macd.macd_signal()
                            df[f'macd_{strategy}_histogram'] = macd.macd_diff()
                        else:
                            df[f'macd_{strategy}_line'] = np.nan
                            df[f'macd_{strategy}_signal'] = np.nan
                            df[f'macd_{strategy}_histogram'] = np.nan

                    for sma_period in self.sma_set:
                        if len(df) >= sma_period:
                            sma = SMAIndicator(df['Close'], sma_period)
                            df[f'sma_{sma_period}'] = sma.sma_indicator()
                        else:
                            df[f'sma_{sma_period}'] = np.nan

                    for ema_period in self.ema_set:
                        if len(df) >= ema_period:
                            ema = EMAIndicator(df['Close'], ema_period)
                            df[f'ema_{ema_period}'] = ema.ema_indicator()
                        else:
                            df[f'ema_{ema_period}'] = np.nan

                    for bb_period in self.bb_set:
                        if len(df) >= bb_period:
                            bb = BollingerBands(df['Close'], bb_period)
                            df[f'bb_{bb_period}_lband'] = bb.bollinger_lband()
                            df[f'bb_{bb_period}_mavg'] = bb.bollinger_mavg()
                            df[f'bb_{bb_period}_hband'] = bb.bollinger_hband()
                        else: 
                            df[f'bb_{bb_period}_lband'] = np.nan
                            df[f'bb_{bb_period}_mavg'] = np.nan
                            df[f'bb_{bb_period}_hband'] = np.nan

                    if len(df) >= self.eom_set:
                        eom = EaseOfMovementIndicator(df['High'], df['Low'], df['Volume'], self.eom_set) # 14 is the default window
                        df['eom'] = eom.ease_of_movement()
                        df['eom_signal'] = eom.sma_ease_of_movement()  # EOM with a signal line
                    else:
                        df['eom'] = np.nan
                        df['eom_signal'] = np.nan

                    # Calculate ATR (Average True Range)
                    if len(df) >= self.atr_set:  
                        atr = AverageTrueRange(df['High'], df['Low'], df['Close'])
                        df['atr'] = atr.average_true_range()
                    else: 
                        df['atr'] = np.nan

                    if len(df) >= self.adx_set:  
                        adx = ADXIndicator(df['High'], df['Low'], df['Close'])
                        df['adx'] = adx.adx()
                    else:
                        df['adx'] = np.nan


                    obv = OnBalanceVolumeIndicator(df['Close'], df['Volume'])
                    df['obv'] = obv.on_balance_volume()

                    # Ultimate Oscillator
                    ultimate_osc = UltimateOscillator(df['High'], df['Low'], df['Close'])
                    df['ultimate_osc'] = ultimate_osc.ultimate_oscillator()

                    # Williams %R
                    williams_r = WilliamsRIndicator(df['High'], df['Low'], df['Close'])
                    df['williams_r'] = williams_r.williams_r()

                    # Chaikin Money Flow
                    cmf = ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'])
                    df['chaikin_money_flow'] = cmf.chaikin_money_flow()

                    # Force Index
                    force_index = ForceIndexIndicator(df['Close'], df['Volume'])
                    df['force_index'] = force_index.force_index()

                    # Negative Volume Index
                    nvi = NegativeVolumeIndexIndicator(df['Close'], df['Volume'])
                    df['negative_volume_index'] = nvi.negative_volume_index()

                    vpt = VolumePriceTrendIndicator(df['Close'], df['Volume'])
                    df['vpt'] = vpt.volume_price_trend()

                    vwma = VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume'])
                    df['vwma'] = vwma.volume_weighted_average_price()

                    cmf = ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], 20) # 20 is the default window
                    df['cmf'] = cmf.chaikin_money_flow()                    

                    fi = ForceIndexIndicator(df['Close'], df['Volume'])
                    df['fi'] = fi.force_index()

                    # Calculate Donchian Channel
                    dc = DonchianChannel(df['High'], df['Low'], df['Close'])
                    df['dc_high'] = dc.donchian_channel_hband()
                    df['dc_low'] = dc.donchian_channel_lband()

                    # Calculate Keltner Channel
                    kc = KeltnerChannel(df['High'], df['Low'], df['Close'])
                    df['kc_high'] = kc.keltner_channel_hband()
                    df['kc_low'] = kc.keltner_channel_lband()

                    # Calculate Ulcer Index
                    ui = UlcerIndex(df['Close'])
                    df['ui'] = ui.ulcer_index()

                
                    # Calculate Aroon Indicator
                    aroon = AroonIndicator(df['Close'])
                    df['aroon_down'] = aroon.aroon_down()
                    df['aroon_up'] = aroon.aroon_up()

                    # Calculate DPO (Detrended Price Oscillator)
                    dpo = DPOIndicator(df['Close'])
                    df['dpo'] = dpo.dpo()

                    ichimoku_indicator = IchimokuIndicator(df['High'], df['Low'])
                    df['ichimoku_a'] = ichimoku_indicator.ichimoku_a()
                    df['ichimoku_b'] = ichimoku_indicator.ichimoku_b()
                    df['ichimoku_base_line'] = ichimoku_indicator.ichimoku_base_line()
                    df['ichimoku_conversion_line'] = ichimoku_indicator.ichimoku_conversion_line()

                    kst_indicator = KSTIndicator(df['Close'])
                    df['kst'] = kst_indicator.kst()
                    df['kst_diff'] = kst_indicator.kst_diff()
                    df['kst_signal'] = kst_indicator.kst_sig()

                    mass_index = MassIndex(df['High'], df['Low'])
                    df['mass_index'] = mass_index.mass_index()

                    psar_indicator = PSARIndicator(df['High'], df['Low'], df['Close'])
                    df['psar'] = psar_indicator.psar()

                    trix_indicator = TRIXIndicator(df['Close'])
                    df['trix'] = trix_indicator.trix()

                    vortex_indicator = VortexIndicator(df['High'], df['Low'], df['Close'])
                    df['vortex_indicator_neg'] = vortex_indicator.vortex_indicator_neg()
                    df['vortex_indicator_pos'] = vortex_indicator.vortex_indicator_pos()

                    aroon_ind = AroonIndicator(df['Close'])
                    df['aroon_up'] = aroon_ind.aroon_up()
                    df['aroon_down'] = aroon_ind.aroon_down()

                    cci = CCIIndicator(df['High'], df['Low'], df['Close'])
                    df['cci'] = cci.cci()

                    macd = MACD(df['Close'])
                    df['macd_signal'] = macd.macd_signal()

                    df['daily_log_return'] = np.log(df['Close'] / df['Close'].shift())

                    awesome_osc = AwesomeOscillatorIndicator(df['High'], df['Low'])
                    df['awesome_oscillator'] = awesome_osc.awesome_oscillator()

                    kama = KAMAIndicator(df['Close'])
                    df['kama'] = kama.kama()

                    ppo = PercentagePriceOscillator(df['Close'])
                    df['ppo'] = ppo.ppo()

                    # Save to CSV with scenario/settings in the filename
                    
                    df.to_csv(os.path.join(self.filepath, f'data_{interval}', f"{symbol}_{interval}_all_indicators.csv"))
                    self.pair_data[(pair, interval)].to_csv(os.path.join(self.filepath,f'data_{interval}', f"{symbol}_{interval}_hist_data.csv"))


    def _load_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def _extract_macd_settings(self):
        macd_str = self.config.get('indicator_settings', 'macd')
        macd_dict = eval(macd_str)
        return {k: list(map(int, v)) for k, v in macd_dict.items()}

    def get_historical_data(self, symbol, interval, start_date, end_date):
        url = 'https://api.binance.com/api/v3/klines'

        if isinstance(self.start_date, str):
            self.start_date = datetime.strptime(self.start_date, '%d-%m-%Y')
        if isinstance(self.end_date, str):
            self.end_date = datetime.strptime(self.end_date, '%d-%m-%Y')
        
        start_ts = int(self.start_date.timestamp() * 1000)
        end_ts = int(self.end_date.timestamp() * 1000)
        
        limit = 1000
        
        data_df = pd.DataFrame()

        while True:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_ts,
                'limit': limit}

            r = requests.get(url, params=params)

            if r.status_code != 200:
                print(f"Error fetching data from Binance API. Status code: {r.status_code}")
                return None

            #print("Params: ", params)  # Print the parameters

            #print("Response: ", r.text)  # Print the response text

            data = r.json()

            df = pd.DataFrame(data, index=range(len(data)))
            df = df.rename(columns={0: 'Open time', 1: 'Open', 2: 'High', 3: 'Low', 4: 'Close', 5: 'Volume', 6: 'Close time', 7: 'Quote asset volume', 8: 'Number of trades', 9: 'Taker buy base asset volume', 10: 'Taker buy quote asset volume', 11: 'Ignore'})
        
            df['Open'] = pd.to_numeric(df['Open'])
            df['High'] = pd.to_numeric(df['High'])
            df['Low'] = pd.to_numeric(df['Low'])
            df['Close'] = pd.to_numeric(df['Close'])
            df['Volume'] = pd.to_numeric(df['Volume'])

            df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
            df = df.set_index('Open time')
            data_df = pd.concat([data_df, df], axis =0)

            if data_df.empty:
                return None

            start_ts = df.iloc[-1]['Close time'] + 1

            if len(data) < limit or start_ts >= end_ts:
                break

            time.sleep(1)  # Binance rate limit
        data_df = data_df[~data_df.index.duplicated(keep='first')]
        
        return data_df


    
    #%%
config_path = r'C:\Users\mauri\Documents\Trading Bot\Python\AlgoTrader\Config\histo_data_crypto_config.ini'

loader = Histo_Loader(config_path)

# %%
