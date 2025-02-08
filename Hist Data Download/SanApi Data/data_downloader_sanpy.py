import pandas as pd
import san 
import numpy as np 
import datetime as dt
import os
import time
from san import AsyncBatch, Batch
import json
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import sys
import platform
import re

def get_running_environment():
    if 'microsoft-standard' in platform.uname().release:
        return 'wsl'
    elif platform.system() == 'Windows':
        return 'windows'
    else:
        return 'unknown'

# Convert path based on environment
def get_converted_path(path):
    return convert_path(path, env)

# Detect environment
env = get_running_environment()


san.ApiConfig.api_key = "fu7l3ih6egrecgll_hgafph3rk734krj6"

base_path = os.path.dirname(os.path.realpath(__file__))
hist_data_download_path = os.path.dirname(base_path)
crypto_bot_path = os.path.dirname(hist_data_download_path)


if env == 'wsl':
    crypto_bot_path = "/mnt/c/Users/mauri/Documents/Trading Bot/Python/AlgoTrader"
else:
    base_path = os.path.dirname(os.path.realpath(__file__))
    hist_data_download_path = os.path.dirname(base_path)
    crypto_bot_path = os.path.dirname(hist_data_download_path)

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

class Data_Downloader_All:
    def __init__(self) -> None:
        self.api_key = 'fu7l3ih6egrecgll_hgafph3rk734krj6'
        san.ApiConfig.api_key = self.api_key 
       
        #config and paths 
        # config_filename = "config_san_api_data_downloader.ini"
        # self.config = utils.read_config_file(os.path.join(config_path,config_filename))
        # self.intervals =utils.get_config_value(self.config,'download_settings',"intervals")
        self.main_path = os.path.join(san_api_data_path, "Main data files")
        
        self.download_path = histo_data_path_crypto
        self.dataset_path = dataset_path_crypto

        self.today = dt.date.today()
        self.logger = logger
        self.configure_logger()

        self.logger_data = logger
        self.configure_dataset_logger()

        # files and co
         #retrieve all assets and create mapping with san coin names
        all_asset_file_path = f"{main_data_files_path}/all_assets.xlsx" 
        self.all_assets = pd.read_excel(os.path.join(main_data_files_path, "all_assets.xlsx"), header=0)
        
        self.instrument_pool_file = pd.read_excel(os.path.join(self.main_path, "instrument_pool.xlsx"),sheet_name="instrument pool", header=0)
        self.instrument_pool = self.instrument_pool_file['slug'].tolist()

        self.available_coins_kucoin = pd.read_excel(os.path.join(self.main_path,"kucoin_available_coins.xlsx"), header=0)
        self.instrument_pool_kucoin = self.get_slugs_available_on_kucoin()

        self.metrics = pd.read_excel(os.path.join(self.main_path, "metrics_overview.xlsx"), sheet_name="All metrics",header=0)
        self.t1_metrics = pd.read_excel(os.path.join(self.main_path, "metrics_overview.xlsx"), sheet_name="Tier 1 metrics",header=0)
        self.t2_metrics = pd.read_excel(os.path.join(self.main_path, "metrics_overview.xlsx"), sheet_name="Tier 2 metrics",header=0)
        self.nft_metrics = pd.read_excel(os.path.join(self.main_path, "metrics_overview.xlsx"), sheet_name="NFT metrics",header=0)
        self.special_metrics_queryable = pd.read_excel(os.path.join(self.main_path, "metrics_overview.xlsx"), sheet_name="special metrics query all",header=0)
        self.special_metrics_pool = pd.read_excel(os.path.join(self.main_path, "metrics_overview.xlsx"), sheet_name="special metrics",header=0)
        self.new_metrics = pd.read_excel(os.path.join(self.main_path, "metrics_new_creation.xlsx"))
        # self.failed_consolidated_special_metrics =  pd.read_excel(os.path.join(self.main_path, "consolidation_fails_special_metrics.xlsx"), header=0)

        #metric bathes
        self.t1_metrics_batch1 = pd.read_excel(os.path.join(self.main_path, "metrics_overview.xlsx"), sheet_name="Tier 1 metrics 1h batch 1",header=0)
        self.t1_metrics_batch2 = pd.read_excel(os.path.join(self.main_path, "metrics_overview.xlsx"), sheet_name="Tier 1 metrics 1h batch 2",header=0)
        self.t1_metrics_batch3 = pd.read_excel(os.path.join(self.main_path, "metrics_overview.xlsx"), sheet_name="Tier 1 metrics 1h batch 3",header=0)
        self.t1_metrics_batch4 = pd.read_excel(os.path.join(self.main_path, "metrics_overview.xlsx"), sheet_name="Tier 1 metrics 1h batch 4",header=0)
        self.t1_metrics_batch5 = pd.read_excel(os.path.join(self.main_path, "metrics_overview.xlsx"), sheet_name="Tier 1 metrics 1h batch 5",header=0)
        self.t1_metrics_reload = pd.read_excel(os.path.join(self.main_path, "metrics_overview.xlsx"), sheet_name="Tier 1 metrics reload",header=0)
        self.special_metrics_pool_reload = pd.read_excel(os.path.join(self.main_path, "metrics_overview.xlsx"), sheet_name="special metrics query reload",header=0)

    def check_available_metrics(self, asset):
        available_metrics = san.available_metrics_for_slug(asset)
        return available_metrics

    def check_available_instruments(self):
        overview = san.get('projects/all')
        return overview
    
    def update_available_instruments(self):
        overview = san.get('projects/all')
        overview.to_excel(os.path.join(self.main_path, "all_assets.xlsx"), index=False)
    
    def download_all_slugs_per_metric(self):
        self.logger.info("############################             Starting to download all slugs per metric              ##################################################")
        request_counter = 0
        all_metrics_per_instrument = {}

        for asset in self.instrument_pool: 

                if request_counter < 595:
                    self.logger.info(f'Downloading available metrics for {asset}')
                    available_metrics = self.check_available_metrics(asset)
                    all_metrics_per_instrument[asset] = available_metrics
                    request_counter += 1
                else:
                    self.logger.info(f'#################################      Request limit reached sleeping for 61 seconds      ############################################')
                    time.sleep(61)
                    request_counter = 0   

        # Determine the maximum length among all lists
        max_length = max(len(metrics) for metrics in all_metrics_per_instrument.values())

        # Fill in missing values with None to ensure equal length
        for asset, metrics_list in all_metrics_per_instrument.items():
            while len(metrics_list) < max_length:
                metrics_list.append(None)

        # Convert the dictionary into a DataFrame
        self.all_metrics_per_instrument_df = pd.DataFrame(all_metrics_per_instrument)
        self.all_slugs_per_metric = self.group_slugs_by_metric(self.all_metrics_per_instrument_df)
        file_path = os.path.join(self.main_path, 'available_slugs_per_metric.json')

        with open(file_path, 'w') as json_file:
            json.dump(self.all_slugs_per_metric, json_file, indent=4)

        self.logger.info(f'Data saved to {file_path}')
        self.logger.info('###########################################      All metrics fetched. Data file saved. Ending process.         #####################################')


    def group_slugs_by_metric(self, metrics_per_instrument_df):
        
        metrics = self.metrics['metrics']
        metric_to_slugs = {metric: [] for metric in metrics}

        # Iterate through each column (metric)
        for metric in metric_to_slugs.keys():

            for slug in metrics_per_instrument_df.columns:
                if metric in metrics_per_instrument_df.loc[:,slug].values:
                    if slug not in metric_to_slugs[metric]: 
                        metric_to_slugs[metric].append(slug)

        return metric_to_slugs


    def check_data_availablity(self):
        
        file_path = os.path.join(self.main_path, 'available_slugs_per_metric.json')
        with open(file_path,"r") as json_file:
            available_slugs_per_metric = json.load(json_file)

        # Create a dictionary to store key and respective list length
        lengths_dict = {}
        
        for key, value in available_slugs_per_metric.items():
            if isinstance(value, list):
                lengths_dict[key] = len(value)

        # Convert the dictionary into a DataFrame
        self.overview_slugs_per_metric = pd.DataFrame(list(lengths_dict.items()), columns=['Key', 'List_Length'])

        ###################           uncomment for saving the file           ########################################
        self.overview_slugs_per_metric.to_excel(os.path.join(self.main_path, "metric_data_statistics.xlsx"))
                                                
        # If you still want to get the longest list and its key, you can use the following code:
        self.longest_key = self.overview_slugs_per_metric['Key'][self.overview_slugs_per_metric['List_Length'].idxmax()]
        self.longest_list_length = self.overview_slugs_per_metric['List_Length'].max()
        self.largest_instrument_pool_per_metric = available_slugs_per_metric[self.longest_key]



    def query_price_data(self, assets = None, start_date_period = None, end_date_period=None, frequency=None):        
        self.logger.info("#######################################                 starting to download price data         ##############################################")
        api_call_counter = 0

        #get function complexity
        complexity = san.metric_complexity(metric="price_usd", from_date=start_date_period, to_date=end_date_period, interval=frequency, format=None)
        if complexity >=50000:
            query_interval = utils.split_intervals(start_date_period, end_date_period, complexity, frequency,format=format)
            self.logger.info(f"Complexity is {complexity}. Splitting interval in {query_interval}")
        else:
            self.logger.info("Complexity is below 50k using complete interval for query.")
            query_interval = [start_date_period, end_date_period]
        
        data_index = utils.generate_time_series(start_date_period, end_date_period, frequency=frequency, format=format)
        data_index = pd.to_datetime(data_index)
        data_dict = dict()
        
        for slug in assets:
        
            metric_data = pd.DataFrame(index=data_index, columns=[slug])
            # metric_data.index = metric_data.index.tz_localize() 

            for idx in range(0,len(query_interval)-1):
                if api_call_counter < 600:
                    data = san.get( "price_usd", slug = slug, from_date=query_interval[idx], to_date=query_interval[idx+1], interval=frequency )
                    # data.index = data.index.tz_convert("Europe/Berlin")
                    if len(data) > 0: 
                        data.index = data.index.tz_localize(None)
                        metric_data.loc[metric_data.index, slug] = data.loc[data.index, 'value']
                    else:
                        pass

                    api_call_counter += 1
                        
                else:
                    self.logger.info("Api call limit reached sleeping for 61 seconds")
                    time.sleep(61)
                    api_call_counter = 0
                self.logger.info(f"Data downloaded for {slug}")
            
            # metric_data = metric_data.dropna()
            data_dict[slug] = metric_data
        
        data_df = pd.DataFrame(index=data_index, columns=data_dict.keys())

        for df in data_dict.keys():
            data_df.loc[data_df.index, df] = data_dict[df][df]

        data_df.to_excel(os.path.join(self.main_path, (f'price_usd_{frequency}.xlsx')))


    def query_get_many_metric(self, metric, assets = None, start_date_period = None, end_date_period=None, frequency=None, use_last_available_date=None, format=None):        

            directory_path = self.download_path + f"/{metric}/{frequency}/"
            
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            #get function complexity
            complexity = self.calculate_complexity_score(assets, start_date_period, end_date_period, frequency, format=format)

            if complexity >=50000:
                query_interval = utils.split_intervals(start_date_period, end_date_period, complexity, frequency,None, format=format)
                self.logger.info(f"Complexity is {complexity}. Splitting interval in {len(query_interval)-1} intervals")
            else:
                self.logger.info("Complexity is below 50k using complete interval for query.")
                query_interval = [start_date_period, end_date_period]
            
            data_index = utils.generate_time_series(start_date_period, end_date_period, frequency=frequency, format=format)
            data_index = pd.to_datetime(data_index, utc=True)
            
            download_data = pd.DataFrame(index=data_index, columns=assets)
            download_data.index = pd.to_datetime(download_data.index, utc=True)

            calls_remaining_before_request = int(san.api_calls_remaining()["month_remaining"])
            calls_remaining_hour = int(san.api_calls_remaining()['hour_remaining'])
            calls_remaining_minute = int(san.api_calls_remaining()['minute_remaining'])                                                                                  
            self.logger.info(f"Api calls remaining :   {calls_remaining_before_request}")

            query_counter = 0
            for idx in range(0, len(query_interval) - 1):
                while calls_remaining_hour < 10 or calls_remaining_minute  < 5:
                    if calls_remaining_hour < 10:
                        self.logger.info("Api call limit reached, sleeping for 1 hour")
                        time.sleep(3602)
                        calls_remaining_hour = int(san.api_calls_remaining()['hour_remaining'])
                    else:
                        self.logger.info("Api call limit reached, sleeping for 61 seconds")
                        time.sleep(61)
                        calls_remaining_minute = int(san.api_calls_remaining()['minute_remaining']) 
                    
                self.logger.info(f"Retrieving data for time period {query_interval[idx]} - {query_interval[idx+1]}.")
                if query_counter >=450:
                    self.logger.info("Api call limit reached, sleeping for 61 seconds")
                    time.sleep(61)
                    query_counter = 0    
                data = san.get_many(metric, slugs=assets, from_date=query_interval[idx], to_date=query_interval[idx+1], interval=frequency)
                query_counter += 1
                data.index = pd.to_datetime(data.index, utc=True)
                if len(data) > 0:    
                    for asset in assets:
                        if asset in data.columns:
                            download_data.loc[data.index, asset] = data[asset]

            download_data = download_data.sort_index()           
            set_assets = set(assets)
            set_downloaded_assets = set(download_data.columns[download_data.notna().any()])
            non_download = list(set_assets-set_downloaded_assets)
            calls_remaining_after_request = int(san.api_calls_remaining()["month_remaining"])

            self.logger.info(f"Api calls used: {calls_remaining_before_request-calls_remaining_after_request}, Api calls remaining : {calls_remaining_after_request}")
            return download_data, non_download
    

    def download_all_get_many_metrics(self, metrics= None, instrument_pool = None, start_date_period = None, end_date_period=None, frequency=None, use_last_available_date=None, only_new_assets=None, format=None):
        
        if os.path.exists(os.path.join(self.main_path, f"missing_get_many_metrics_{frequency}.xlsx")):
            self.missing_get_many_metrics = pd.read_excel(os.path.join(self.main_path, f"missing_get_many_metrics_{frequency}.xlsx"),header=0)["Metrics"].tolist()
        else:
            self.missing_get_many_metrics = []

        instrument_pool = np.unique(instrument_pool).tolist()
        data_index = utils.generate_time_series(start_date_period, end_date_period, frequency=frequency, format=format)
        data_index = pd.to_datetime(data_index, utc =True)
        
        if use_last_available_date ==True:
            end_date_period = dt.date.today().strftime(format)
            updated_data_index = utils.generate_time_series(start_date_period, end_date_period, frequency=frequency, format=format)
    
        if pd.isnull(data_index).any():
            self.logger.error("Date parsing resulted in NaT values. Please check the input dates and format.")
            return
        
        if not data_index.is_unique:
            self.logger.error("Duplicate dates found in data index. Please check the generate_time_series function and the input parameters.")
            return

        metric_counter = 1
        
        for metric in metrics['Metrics']:

            self.logger.info("###############################################################################################################################################")
            self.logger.info(f"#                                         Downloading data for {metric}                                                                      #")
            self.logger.info(f"#                                         Metric {metric_counter} of {len(metrics['Metrics'])}                                               #")
            self.logger.info('###############################################################################################################################################')
            metric_counter += 1

            file_path = self.download_path+ f'/{metric}/{frequency}/' + f'{metric}_{frequency}.csv'
            use_existing_data = False
            
            if os.path.exists(file_path):
                use_existing_data = True
                metric_data = pd.read_csv(file_path,index_col=0) #"Unnamed: 0"
                if len(metric_data.columns) >= 1:
                    metric_data = self.drop_duplicate_columns(metric_data)
                metric_data.index = pd.to_datetime(metric_data.index, utc=True)

                updated_instrument_pool = list(set(instrument_pool) - set(metric_data.columns.tolist()))

                if only_new_assets==True:
                    instrument_pool = updated_instrument_pool

                    if len(instrument_pool) <= 0:
                        self.logger.error("No updated instruments to query found. Please investigate")
                        break

                metric_data_column_addition = pd.DataFrame(index=metric_data.index, columns=updated_instrument_pool)
                metric_data_column_addition.index = pd.to_datetime(metric_data_column_addition.index, utc=True)
                metric_data = pd.concat([metric_data, metric_data_column_addition], axis=1)
                metric_data.index = pd.to_datetime(metric_data.index, utc=True)
                
                metric_data = metric_data.T.reset_index()
                metric_data = metric_data.drop_duplicates(subset='index',keep="first")
                metric_data = metric_data.set_index("index").T

                if use_last_available_date == True:
                    lowest_date = [] 
                    for col in metric_data.columns:
                        last_date = metric_data[col].dropna().index.max()
                        lowest_date.append(pd.Timestamp(last_date))
                        
                    start_date_period = min(lowest_date)
                    end_date_period = dt.date.today().strftime(format)
                    download_index = utils.generate_time_series(start_date_period, end_date_period, frequency=frequency, format=format)
                    download_index = pd.to_datetime(download_index, utc=True)
                    download_data = pd.DataFrame(index=download_index)
                    new_timestamps =  download_index.difference(metric_data.index).sort_values()
                    metric_data_index_addition = pd.DataFrame(index=new_timestamps, columns=metric_data.columns)
                    metric_data_index_addition.index = pd.to_datetime(metric_data_index_addition.index, utc=True)
                    metric_data = pd.concat([metric_data, metric_data_index_addition], axis=0)       
                else:
                    download_data = pd.DataFrame(index=data_index)
                    download_index = utils.generate_time_series(start_date_period, end_date_period, frequency=frequency, format=format)
                    download_index = pd.to_datetime(download_index, utc=True)
                    new_timestamps =  download_index.difference(metric_data.index).sort_values()
                    metric_data_index_addition = pd.DataFrame(index=new_timestamps, columns=metric_data.columns)
                    metric_data = pd.concat([metric_data, metric_data_index_addition], axis=0)
                    
            else:
                if use_last_available_date == True:
                    download_data = pd.DataFrame(index=updated_data_index)
                else:
                    download_data = pd.DataFrame(index=data_index)

            download_data.index = pd.to_datetime(download_data.index, utc=True)

            
            batch_size_slugs = 120
            n_downloads = (len(instrument_pool) // batch_size_slugs) +1
            missing_data_for_slug = list()
            
            try:
                download_job_counter = 1
                for idx in range(0, len(instrument_pool),batch_size_slugs):
                    
                    self.logger.info(f"Current_download_job at {download_job_counter} / {n_downloads}.")  
                    download_job_counter += 1
                    current_pool = instrument_pool[idx:idx+batch_size_slugs]
                    print(current_pool)
                    data, not_downloaded = self.query_get_many_metric(metric, assets = current_pool, start_date_period = start_date_period, end_date_period=end_date_period, frequency=frequency, use_last_available_date=use_last_available_date) 
                    download_data = pd.concat([download_data,data], axis=1)
                    missing_data_for_slug += not_downloaded
                
                if use_existing_data ==True: 
                    for asset in instrument_pool:
                        if asset in download_data.columns:
                            metric_data.loc[download_data.index,asset] = download_data[asset] 
                else:
                    metric_data = download_data    

            except Exception as e:

                self.missing_get_many_metrics.append(metric)
                missing_get_many_metrics = pd.DataFrame(columns=['Metrics'], data=self.missing_get_many_metrics)
                missing_get_many_metrics.to_excel(os.path.join(self.main_path, f"missing_get_many_metrics_{frequency}.xlsx"))
                self.logger.error(f"######################################################                 WARNING               ############################################")
                self.logger.error(f"An error has occured when retrieving {metric}. Please check.")
                self.logger.error(str(e))
                self.logger.error(f"######################################################                 WARNING               ############################################")
                pass
            
            self.logger.info(f"Data not downloaded for {len(missing_data_for_slug)} instruments. Missing Instruments are {missing_data_for_slug}")
            self.logger.info("#######################################################     Saving file    ##################################################")
            
            metric_data = metric_data[~metric_data.index.duplicated(keep='last')]
            metric_data = metric_data.sort_index(axis=1)
            metric_data = metric_data.sort_index()
            metric_data.to_csv(file_path)

            
    def download_all_special_metrics(self, metrics= None, instrument_pool = None, start_date_period = None, end_date_period=None, frequency=None, use_last_available_date=None, format=None):
        
        total_instrument_pool = instrument_pool
        data_index = utils.generate_time_series(start_date_period, end_date_period, frequency=frequency, format=format)
        data_index = pd.to_datetime(data_index)
        
        if pd.isnull(data_index).any():
            self.logger.error("Date parsing resulted in NaT values. Please check the input dates and format.")
            return
        
        if not data_index.is_unique:
            self.logger.error("Duplicate dates found in data index. Please check the generate_time_series function and the input parameters.")
            return


        metric_counter = 1
        self.missing_get_special_metrics = []
        100
        for metric in metrics['metrics']:
            
            self.logger.info("###############################################################################################################################################")
            self.logger.info(f"#                                         Downloading data for {metric}                                                                      #")
            self.logger.info(f"#                                         Metric {metric_counter} of {len(metrics['metrics'])}                                               #")
            self.logger.info('###############################################################################################################################################')
            metric_counter += 1
            missing_data_for_slug = list()

            directory_path = self.download_path + f"/{metric}/{frequency}/"
            
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            calls_remaining_before_request = int(san.api_calls_remaining()["month_remaining"])
            request_counter_hour = 0 
            request_counter_minute = 0

            self.logger.info("#############################################################################################")
            self.logger.info(f"Api calls remaining :   {calls_remaining_before_request}")
            
            if metric == "gas_used":
                instrument_pool = ["ethereum"]
            else:
                instrument_pool = total_instrument_pool
            
            for asset in instrument_pool:
                download_fail_check = False
                file_path = self.download_path+ f'/{metric}/{frequency}/' + f'{metric}_{frequency}_{asset}.csv'
                
                if os.path.exists(file_path):
                    metric_data = pd.read_csv(file_path)
                    metric_data = metric_data.loc[:, ~metric_data.columns.str.contains('Unnamed')]

                    if "datetime" in metric_data.columns:
                        metric_data.set_index("datetime", inplace=True)  
                    else:
                        metric_data.set_index(metric_data.columns[0], inplace=True)

                    metric_data.index = pd.to_datetime(metric_data.index, utc=True)

                    if use_last_available_date == True:
                        lowest_date = [] 
                        for col in metric_data.columns:
                            last_date = metric_data[col].dropna().index.max()
                            lowest_date.append(last_date)
                        
                        start_date_period = min(lowest_date)
                        end_date_period = dt.date.today().strftime(format)
                else:
                    metric_data = pd.DataFrame()
                    metric_data.index = pd.to_datetime(metric_data.index, utc=True)

                if frequency=="1h":
                    query_interval = utils.split_intervals(start_date_period, end_date_period, 100000, frequency,None, format=format)
                elif frequency=="5m": 
                    query_interval = utils.split_intervals(start_date_period, end_date_period, 1080000, frequency,None, format=format)
                else:
                    self.logger.info("Complexity is below 50k using complete interval for query.")
                    query_interval = [start_date_period, end_date_period]

                try:

                    for idx in range(0, len(query_interval) - 1):

                        if request_counter_hour > 29920 or request_counter_minute > 585:
                            if request_counter_hour > 29850:
                                self.logger.info("Api call limit reached, sleeping for 1 hour")
                                time.sleep(3602)
                                request_counter_hour = 0
                                
                            else:
                                self.logger.info("Api call limit reached, sleeping for 61 seconds")
                                time.sleep(61)
                                request_counter_minute = 0            

                        if metric =="ohlcv":
                            ohlcv_metric = metric+f"/{asset}"
                            data_df = san.get(ohlcv_metric, from_date=query_interval[idx], to_date=query_interval[idx+1], interval=frequency)
                        else:
                            try:
                                data_df = san.get(metric, slug= asset, from_date=query_interval[idx], to_date=query_interval[idx+1], interval=frequency)
                                request_counter_hour += 1
                                request_counter_minute +=1
                                data_df.index = pd.to_datetime(data_df.index, utc=True)
                                metric_data = pd.concat([metric_data, data_df], axis=0)
                            except Exception as e:
                                self.logger.error(f"{asset} could not be fetched please investigate. Reason is {e}")
                                download_fail_check = True 
                                pass
                    
                    if metric_data.empty or metric_data.isna().all().all() or download_fail_check == True:
                        missing_data_for_slug.append(asset)
                        self.logger.error(f"Data not downloaded for {asset} because there were no values available or the metric is not available for the instrument")
                    else:
                        self.logger.info(f"#######################################################     Saving file for {asset}   ")
                        metric_data = metric_data[~metric_data.index.duplicated(keep='last')]
                        metric_data = metric_data.sort_index()
                        metric_data.to_csv(file_path, index_label="datetime", index=True)
                        
                except Exception as e:
                    if san.is_rate_limit_exception(e):
                        print(f"Hit rate limit, will sleep for 61 seconds")
                        time.sleep(61)
                    else:
                        self.missing_get_special_metrics.append(metric)
                        self.missing_get_special_metrics = pd.DataFrame(self.missing_get_special_metrics)
                        self.missing_get_special_metrics.to_excel(os.path.join(self.main_path, f"missing_special_metrics_{frequency}.xlsx"))

                        self.logger.error(f"######################################################                 WARNING               ############################################")
                        self.logger.error(f"An error has occured when retrieving {metric}. Please check.")
                        self.logger.error(str(e))
                        self.logger.error(f"######################################################                 WARNING               ############################################")
                        pass
                    pass
                
    
            self.logger.info(f"Data not downloaded for {len(missing_data_for_slug)} instruments. Missing Instruments are {missing_data_for_slug}")
            calls_remaining_after_request = int(san.api_calls_remaining()["month_remaining"])
            self.logger.info('')
            self.logger.info(f"Api calls used: {calls_remaining_before_request-calls_remaining_after_request}, Api calls remaining : {calls_remaining_after_request}")
            self.logger.info("")
            self.logger.info("Download Job finished. Continouing with next job")
            
        self.missing_get_special_metrics = pd.DataFrame(self.missing_get_special_metrics)
        self.missing_get_special_metrics.to_excel(os.path.join(self.main_path, "missing_special_metrics.xlsx"))
    



    def execute_batch_request(self,metrics=None, instrument_pool=None, start_date_period = None, end_date_period=None, frequency=None, use_last_available_date=None, format=None):
        
        data_index = utils.generate_time_series(start_date_period, end_date_period, frequency=frequency, format=format)
        data_index = pd.to_datetime(data_index)

        if pd.isnull(data_index).any():
            self.logger.error("Date parsing resulted in NaT values. Please check the input dates and format.")
            return
        
        if not data_index.is_unique:
            self.logger.error("Duplicate dates found in data index. Please check the generate_time_series function and the input parameters.")
            return

        batch = Batch()
        variable_list = []

        for metric in metrics:

            complexity = self.calculate_complexity_score(instrument_pool, start_date_period, end_date_period, frequency, format=format)

            if complexity >=50000:
                query_interval = utils.split_intervals(start_date_period, end_date_period, complexity, frequency,None, format=format)
                self.logger.info(f"Complexity is {complexity}. Splitting interval in {len(query_interval)-1} intervals")
            else:
                self.logger.info("Complexity is below 50k using complete interval for query.")
                query_interval = [start_date_period, end_date_period]

            download_data = pd.DataFrame(index=data_index, columns=assets)
            download_data.index = pd.to_datetime(download_data.index, utc=True)

            calls_remaining_before_request = int(san.api_calls_remaining()["month_remaining"])
            calls_remaining_hour = int(san.api_calls_remaining()['hour_remaining'])
            calls_remaining_minute = int(san.api_calls_remaining()['minute_remaining'])                                                                                  
            self.logger.info(f"Api calls remaining :   {calls_remaining_before_request}")
            
            query_counter = 0

            for idx in range(0, len(query_interval) - 1):
                if (len(query_interval)-1)==2:
                    variable_name = metric
                else:
                    variable_name = metric + str(idx)
                
                variable_list.append(variable_name)
                # while calls_remaining_hour < 10 or calls_remaining_minute  < 5:
                #     if calls_remaining_hour < 10:
                #         self.logger.info("Api call limit reached, sleeping for 1 hour")
                #         time.sleep(3602)
                #         calls_remaining_hour = int(san.api_calls_remaining()['hour_remaining'])
                #     else:
                #         self.logger.info("Api call limit reached, sleeping for 61 seconds")
                #         time.sleep(61)
                #         calls_remaining_minute = int(san.api_calls_remaining()['minute_remaining']) 
                    
                self.logger.info(f"Retrieving data for time period {query_interval[idx]} - {query_interval[idx+1]}.")
                if query_counter >=550:
                    self.logger.info("Api call limit reached, sleeping for 61 seconds")
                    time.sleep(61)
                    query_counter = 0
                
                batch.get_many( 
                    metric,
                    slugs=[instrument_pool],
                    from_date = query_interval[idx],
                    to_date = query_interval[idx+1],
                    interval = frequency)
                query_counter +=1

        variable_list = batch.execute(max_workers=10)

        calls_remaining_before_request = int(san.api_calls_remaining()["month_remaining"])
        request_counter_hour = 0 
        request_counter_minute = 0

        self.logger.info("#############################################################################################")
        self.logger.info(f"Api calls remaining :   {calls_remaining_before_request}")

        downloaded_data =  {}

        for variable in variable_list:
            # file_path = self.download_path+ f'/{variable}/{frequency}/' + f'{variable}_{frequency}.csv'
            for asset in instrument_pool:
                value = variable[asset]
                downloaded_data[asset][variable] = value

        for asset in instrument_pool:
            directory_path = os.path.join(self.download_path, asset, frequency)
            file_name = f"{asset}_{frequency}_consolidated_data.csv"

            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            tmp_df = pd.DataFrame().from_dict(download_data[asset])


    def download_all_metrics(self,metrics = None, instrument_pool=None, start_date=None,end_date=None, frequency=None, 
                             use_last_available_date=False, only_new_assets=None, format=None, pooled_metrics=None):
        metrics_get_many = []
        metrics_special = []

        for metric in metrics:
            if metric in self.t1_metrics:
                metrics_get_many.append(metric)
            elif metric in self.special_metrics_pool:
                metrics_special.append(metric)
        
        self.download_all_get_many_metrics(metrics=metrics_get_many, instrument_pool=instrument_pool, start_date_period=start_date, end_date_period=end_date,
                                           frequency=frequency, use_last_available_date=use_last_available_date, format=format)
        self.download_all_special_metrics(metrics=metrics_special, instrument_pool=instrument_pool, start_date_period=start_date, end_date_period=end_date,
                                           frequency=frequency, use_last_available_date=use_last_available_date, format=format)
        self.special_metrics_consolidator(metrics=metrics_special, frequency=frequency)

        if pooled_metrics:
            for metric in pooled_metrics.columns:
                slugs = pooled_metrics.metric.values.tolist()

                if metric in self.t1_metrics:
                    self.download_all_get_many_metrics(metrics=metric, instrument_pool=slugs, start_date_period=start_date, end_date_period=end_date,
                                           frequency=frequency, use_last_available_date=use_last_available_date, format=format)
                elif metric in self.special_metrics_pool:
                    self.download_all_special_metrics(metrics=metric, instrument_pool=slugs, start_date_period=start_date, end_date_period=end_date,
                                           frequency=frequency, use_last_available_date=use_last_available_date, format=format)

                    self.special_metrics_consolidator(metrics=metric, frequency=frequency)

        self.special_metrics_creator(frequency=frequency)
        
    #########################################################################################################################################################################################################
    #                                                               extra functions
    #########################################################################################################################################################################################################

    def calculate_complexity_score(self, assets, start_date=None, end_date=None, frequency=None, format=None):
                
        dates = utils.generate_time_series(start_date=start_date, end_date=end_date, frequency=frequency, format=format)
        dates = pd.to_datetime(dates)

        n_assets = len(assets)
        n_days= len(dates)
        n_fields = n_assets+1
        n_years = max(dates[-1].year-dates[0].year,2)/2 
        subscribtion_plan_divisor = 5
        weight = 1
        complexity = (n_days*n_fields*weight*n_years)/subscribtion_plan_divisor
    
        # if frequency == "1h":
        #     complexity = complexity*2
        # elif frequency == "1d":
        #     complexity = complexity*1.5

        complexity = complexity*2
    
        self.logger.info(f"Complexity_score_is: {complexity}")
        return complexity
    
    def drop_duplicate_columns(self, df):
        unique_columns = {}
        columns_to_keep = []
        
        for column in df.columns:
            original_name = column.split('.')[0]  # Extract the base name of the column
            if original_name not in unique_columns:
                unique_columns[original_name] = df[column]
                columns_to_keep.append(df[column].rename(original_name))
        
        # Concatenate all unique columns into a new DataFrame all at once
        df_cleaned = pd.concat(columns_to_keep, axis=1)
        return df_cleaned
    
    def get_slugs_available_on_kucoin(self):
        
        kucoin_tickers  = [instrument for instrument in self.all_assets['ticker'] if instrument in self.available_coins_kucoin['Currency'].tolist()]  
        
        ticker_to_slug_mapping = dict(zip(self.all_assets['ticker'], self.all_assets['slug']))
        instrument_pool_kucoin = [ticker_to_slug_mapping[ticker] for ticker in kucoin_tickers if ticker in ticker_to_slug_mapping]
        instrument_pool_kucoin = np.unique(instrument_pool_kucoin).tolist()

        return instrument_pool_kucoin
    
    def configure_logger(self):

        #logger
        current_datetime = dt.datetime.now()
        timestamp = current_datetime.strftime('%d-%m-%Y_%H-%M')
        log_file_name = f"Class_downloader_log_{timestamp}.txt"
        log_file_path = os.path.join(logging_path, log_file_name)

        if not os.path.exists(logging_path):
            os.makedirs(logging_path)

        self.logger.add(log_file_path, rotation="500 MB", level="INFO")


    def configure_dataset_logger(self):

        #logger
        current_datetime = dt.datetime.now()
        timestamp = current_datetime.strftime('%d-%m-%Y_%H-%M')
        log_file_name = f"dataset_logger_{timestamp}.txt"
        log_file_path = os.path.join(logging_path, log_file_name)

        if not os.path.exists(logging_path):
            os.makedirs(logging_path)

        self.logger_data.add(log_file_path, rotation="500 MB", level="INFO")

    def special_metrics_consolidator(self, metrics = None, frequency=None):
        
        failed_metrics_path = os.path.join(self.main_path, "consolidation_fails_special_metrics.xlsx")
        if os.path.exists(failed_metrics_path):
            failed_special_metrics = pd.read_excel(failed_metrics_path)["metrics"].tolist()
        else:
            failed_special_metrics = []
            
        for metric in metrics['metrics']:
            self.logger.info("############################################################################################################################################################")
            self.logger.info(f"Beginning consolidation for {metric}")
            self.logger.info("############################################################################################################################################################")
            # if metric == "ohlc" or metric == "ohlcv":
            #     pass
            # else:
            try:
                directory_path = self.download_path + f"/{metric}/{frequency}/"

                if os.path.exists(directory_path):
                    
                    largest_file = None
                    largest_size = 0
                    multi_column_metrics = None 

                    # Iterate over the files to find the largest one
                    for filename in os.listdir(directory_path):
                        if filename.endswith('.csv') and not filename.endswith(f'{frequency}.csv'):
                            
                            file_path = os.path.join(directory_path, filename)
                            file_size = os.path.getsize(file_path)

                            if file_size > largest_size:
                                largest_size = file_size
                                largest_file = filename

                    largest_file_path = os.path.join(directory_path, largest_file)
                    largest_data = pd.read_csv(largest_file_path)
                    largest_data = largest_data.loc[:, ~largest_data.columns.str.contains('Unnamed')]
                    largest_data.dropna(inplace=True)
                    largest_data.set_index("datetime", inplace=True)
                    consolidated_data = pd.DataFrame(index = largest_data.index)
                    consolidated_data_2 = pd.DataFrame(index = largest_data.index)
                    consolidated_data_3 = pd.DataFrame(index = largest_data.index)
                    consolidated_data_4 = pd.DataFrame(index = largest_data.index)
                    consolidated_data_5 = pd.DataFrame(index = largest_data.index)
                    consolidated_data_6 = pd.DataFrame(index = largest_data.index)
                    
                    
                    multi_column_metrics = largest_data.columns[largest_data.notna().any()].tolist()
                    consolidated_dfs = {0:consolidated_data, 1:consolidated_data_2,2:consolidated_data_3, 3:consolidated_data_4, 4:consolidated_data_5, 5:consolidated_data_6}

                    for filename in os.listdir(directory_path):
                        filesize = os.path.getsize(os.path.join(directory_path, filename))
                        if filename.endswith('.csv') and not filename.endswith(f'{frequency}.csv'):
                            
                            # Split the filename by '_' and take the third part as the asset name
                            metric_string = filename[:-4]
                            parts = metric_string.split('_')
                            if len(parts) >= 3:
                                asset_name = parts[-1]
                                file_path = os.path.join(directory_path, filename)
                                # Read the CSV and add an 'Asset' column
                                
                                df = pd.read_csv(file_path)
                                df = df.loc[:, ~df.columns.str.contains('^Unnamed:')]

                                if len(df) ==0:
                                    pass
                                elif len(multi_column_metrics) > 1:
                                    df_counter = 0
                                    df.dropna(inplace=True)
                                    df.set_index("datetime", inplace=True)
                                    for col in multi_column_metrics:
                                        df_tmp = df[col]
                                        df_tmp.name = asset_name
                                        df_addition = pd.DataFrame(index=consolidated_data.index, columns=[asset_name])
                                        consolidated_dfs[df_counter] = pd.concat([consolidated_dfs[df_counter], df_addition],axis=1)
                                        consolidated_dfs[df_counter][asset_name].update(df_tmp)
                                        consolidated_dfs[df_counter] = consolidated_dfs[df_counter].drop_duplicates().sort_index()
                                        df_counter += 1
                                else:
                                    df.dropna(inplace=True)
                                    df.set_index("datetime", inplace=True)   
                                    df.rename(columns={df.columns[-1]: asset_name}, inplace=True)
                                    df_addition = pd.DataFrame(index=consolidated_data.index, columns=[asset_name])
                                    consolidated_data = pd.concat([consolidated_data, df_addition],axis=1)
                                    consolidated_data.rename(columns={consolidated_data.columns[-1]: asset_name}, inplace=True)
                                    consolidated_data.loc[df.index, asset_name] = df.loc[df.index, asset_name]
                                    consolidated_data[asset_name].update(df[asset_name])

                    # Save the consolidated DataFrame for the metric
                    if len(multi_column_metrics) >1: 
                        df_counter = 0
                        self.logger.info("Saving sub dataframes")
                        for col in multi_column_metrics:
                            if not consolidated_dfs[df_counter].empty:
                                directory_path = self.download_path + f"/{metric}/{frequency}/{col}/"
                                if not os.path.exists(directory_path):
                                    os.makedirs(directory_path)
                                    
                                self.logger.info(f"Saving file for {metric}_{col} at frequency {frequency}.")
                                save_path = directory_path + f"/{metric}_{frequency}_{col}.csv"
                                consolidated_dfs[df_counter].drop_duplicates().sort_index()
                                consolidated_dfs[df_counter].to_csv(save_path)
                                df_counter += 1

                        if metric in failed_special_metrics:
                            failed_special_metrics.remove(metric)
                        failed_special_metrics = np.unique(failed_special_metrics).tolist()
                        failed_metrics = pd.DataFrame(data=failed_special_metrics, columns=["metrics"])
                        failed_metrics.to_excel(failed_metrics_path)

                    elif not consolidated_data.empty:
                        self.logger.info(f"Saving file for {metric} at frequency {frequency}.")
                        save_path = directory_path + f"/{metric}_{frequency}.csv"
                        if metric in failed_special_metrics:
                            failed_special_metrics.remove(metric)
                        failed_special_metrics = np.unique(failed_special_metrics).tolist()
                        failed_metrics = pd.DataFrame(data=failed_special_metrics, columns=["metrics"])
                        failed_metrics.to_excel(failed_metrics_path)

                        consolidated_data = consolidated_data.drop_duplicates().sort_index()
                        consolidated_data.to_csv(save_path)
                    else:
                        self.logger.error(f"No data found for metric: {metric} at frequency: {frequency}")
                        failed_special_metrics.append(metric)
                        failed_special_metrics = np.unique(failed_special_metrics).tolist()
                        failed_metrics = pd.DataFrame(data=failed_special_metrics, columns=["metrics"])
                        failed_metrics.to_excel(failed_metrics_path)
                else:
                    self.logger.error(f"The selected metric is not available at this frequency: {frequency}")
            except:
                self.logger.error(f"The selected metric {metric} can not be consolidated for {frequency}. Saving metric to failed metrics file")
                failed_special_metrics.append(metric)
                failed_special_metrics = np.unique(failed_special_metrics).tolist()
                failed_metrics = pd.DataFrame(data=failed_special_metrics, columns=["metrics"])
                failed_metrics.to_excel(failed_metrics_path)

    
    def load_and_clean_data(self, file_path):
        df = pd.read_csv(file_path, header=0)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        df = df.fillna(0)
        df.columns = ['datetime'] + list(df.columns[1:])  # Rename the first column to 'datetime'
        return df

    def align_dataframes(self, df1, df2):
        # Remove duplicate column names to avoid conflicts
        df1.columns = pd.Index([col if col == 'datetime' else col + '_1' for col in df1.columns])
        df2.columns = pd.Index([col if col == 'datetime' else col + '_2' for col in df2.columns])
        
        # Merge dataframes on 'datetime' to align them and fill missing values with 0
        df_merged = pd.merge(df1, df2, on='datetime', how='outer').fillna(0)
        return df_merged

    def extract_numeric_values(self, input_file):
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", input_file)
        return [float(num) for num in numbers]
    
    def handle_special_calc(self, cum_sum_greater, cum_sum_less, calc_as, common_columns):
        result = cum_sum_greater.copy()
        if '/' in calc_as:
            with np.errstate(divide='ignore', invalid='ignore'):
                for col in common_columns:
                    result[col] = np.where(cum_sum_less[col] != 0, cum_sum_greater[col] / cum_sum_less[col], 0)
        elif '-' in calc_as:
            for col in common_columns:
                result[col] = cum_sum_greater[col] - cum_sum_less[col]
        elif '*' in calc_as:
            for col in common_columns:
                result[col] = cum_sum_greater[col] * cum_sum_less[col]
        elif '+' in calc_as:
            for col in common_columns:
                result[col] = cum_sum_greater[col] + cum_sum_less[col]
        else:
            raise ValueError(f"Unsupported operation {calc_as}.")
        return result

    def calculate_cumulative_sums(self, required_inputs, calc_as, frequency):
        cum_sum_lower = None
        cum_sum_upper = None
        operators = ['/', '*', '+', '-']
        
        for operator in operators:
            if operator in calc_as:
                values = calc_as.split(operator)
                upper_boundary = float(values[0])
                lower_boundary = float(values[1])
                break

        for input_file in required_inputs:
            print(input_file)
            input_file_values = input_file.replace("1M", "1000000").replace("100k", "100000").replace("1B", "1000000000").replace("10M", "10000000").replace("100M", "100000000")
            numeric_values = self.extract_numeric_values(input_file_values)
        
            maximum_value = max(numeric_values)
            minimum_value = min(numeric_values)
            
            df = self.load_and_clean_data(os.path.join(self.download_path, input_file.strip(), frequency, input_file.strip() + "_" + frequency + ".csv"))
            
            if maximum_value <= lower_boundary:
                if cum_sum_lower is None:
                    df.set_index(df.columns[0], inplace=True)
                    if df is not None:
                        cum_sum_lower = df
                else:
                    df.set_index(df.columns[0], inplace=True)
                    if df is not None:
                        cum_sum_lower += df
            elif minimum_value >= upper_boundary:
                if cum_sum_upper is None:
                    df.set_index(df.columns[0], inplace=True)
                    if df is not None:
                        cum_sum_upper = df
                else:
                    df.set_index(df.columns[0], inplace=True)
                    if df is not None:
                        cum_sum_upper += df
        
        common_columns = list(set(cum_sum_lower.columns).intersection(cum_sum_upper.columns))
        
        cum_sum_lower = cum_sum_lower[common_columns]
        cum_sum_upper = cum_sum_upper[common_columns]
        
        return cum_sum_lower, cum_sum_upper, common_columns

    def special_metrics_creator(self, frequency=None):
        failed_metrics_path = os.path.join(self.main_path, "creation_fails_special_metrics.xlsx")
        
        if os.path.exists(failed_metrics_path):
            failed_special_metrics = pd.read_excel(failed_metrics_path)["metrics"].tolist()
        else:
            failed_special_metrics = []

        new_metrics_path = os.path.join(self.main_path, "metrics_new_creation.xlsx")
        new_metrics_df = pd.read_excel(new_metrics_path)

        for index, row in new_metrics_df.iterrows():
            metric = row['signal']
            signal_name = row['signal_name']
            calc_as = row['calc as']
            required_inputs = [input.strip() for input in row['req_inputs'].split(',')]  # Remove whitespaces

            self.logger.info("############################################################################################################################################################")
            self.logger.info(f"Beginning creation for {signal_name}")
            self.logger.info("############################################################################################################################################################")

            try:
                input_data = [self.load_and_clean_data(os.path.join(self.download_path, input.strip(), frequency, input.strip() + "_" + frequency + ".csv")) for input in required_inputs]
                
                # Align dataframes by 'datetime' and fill missing values with 0
                aligned_data = input_data[0]
                for df in input_data[1:]:
                    aligned_data = self.align_dataframes(aligned_data, df)
                
                # Extract common columns after alignment, excluding the 'datetime' column
                common_columns = [col for col in aligned_data.columns if col != 'datetime' and ('_1' in col or '_2' in col)]
                common_columns = [col.split('_')[0] for col in common_columns]

                result = aligned_data[['datetime']].copy()
                if calc_as in ['/', '+', '-', '*']:
                    temp_result = {}
                    for col in common_columns:
                        col_1 = f"{col}_1"
                        col_2 = f"{col}_2"
                        if col_1 in aligned_data.columns and col_2 in aligned_data.columns:
                            if calc_as == '/':
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    temp_result[col] = np.where(aligned_data[col_2] != 0, aligned_data[col_1] / aligned_data[col_2], 0)
                            elif calc_as == '+':
                                temp_result[col] = aligned_data[col_1] + aligned_data[col_2]
                            elif calc_as == '-':
                                temp_result[col] = aligned_data[col_1] - aligned_data[col_2]
                            elif calc_as == '*':
                                temp_result[col] = aligned_data[col_1] * aligned_data[col_2]
                    result = pd.concat([result, pd.DataFrame(temp_result)], axis=1)
                elif any(char.isdigit() for char in str(calc_as)):
                    cum_sum_lower, cum_sum_upper, common_columns = self.calculate_cumulative_sums(required_inputs, calc_as, frequency)
                    result_values = self.handle_special_calc(cum_sum_upper, cum_sum_lower, calc_as, common_columns)
                    result = pd.merge(result, result_values, on='datetime', how='outer').fillna(0)
                else:
                    if signal_name == 'nupl':
                        realized_value = aligned_data.filter(regex='_1$').rename(columns=lambda x: x.rstrip('_1'))
                        market_cap = aligned_data.filter(regex='_2$').rename(columns=lambda x: x.rstrip('_2'))
                        temp_result = {}
                        for col in realized_value.columns:
                            if col in market_cap.columns:
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    temp_result[col] = np.where(market_cap[col] != 0, (market_cap[col] - realized_value[col]) / market_cap[col], 0)
                        result = pd.concat([result, pd.DataFrame(temp_result)], axis=1)
                    elif signal_name == 'rvt_ratio':
                        realized_value = aligned_data.filter(regex='_1$').rename(columns=lambda x: x.rstrip('_1'))
                        transaction_volume = aligned_data.filter(regex='_2$').rename(columns=lambda x: x.rstrip('_2'))
                        temp_result = {}
                        for col in realized_value.columns:
                            if col in transaction_volume.columns:
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    temp_result[col] = np.where(transaction_volume[col] != 0, realized_value[col] / transaction_volume[col], 0)
                        result = pd.concat([result, pd.DataFrame(temp_result)], axis=1)
                    else:
                        raise ValueError(f"Metric {metric} is not recognized or calc_as is not specified correctly.")

                # Ensure no duplicate columns in the final result
                result = result.loc[:, ~result.columns.duplicated()]
                result = result.sort_values(by='datetime')
                result.set_index('datetime', inplace=True)
                result = result.sort_index(axis=1)
                
                if not os.path.exists(os.path.join(self.download_path, signal_name, frequency)):
                    os.makedirs(os.path.join(self.download_path, signal_name, frequency))
                    
                result.to_csv(os.path.join(self.download_path, signal_name, frequency, signal_name + "_" + frequency + ".csv"), index=True)
                self.logger.info(f"Data saved for {signal_name} at frequency {frequency}.")
                    
            except Exception as e:
                self.logger.error(f"The selected metric {metric} can not be created for {frequency}. Error: {str(e)}. Saving metric to failed metrics file")
                failed_special_metrics.append(metric)
                failed_special_metrics = list(set(failed_special_metrics))
                failed_metrics = pd.DataFrame(data=failed_special_metrics, columns=["metrics"])
                failed_metrics.to_excel(failed_metrics_path)
                pass
            # Add more metrics here based on provided data and required calculations
                
        
        
#****************************************   INIT CLASS   ############################################################            
# downloader_class = Data_Downloader_All()


#########################################     DOWNLOAD METRICS ########################################################
# downloader_class.download_all_available_metrics()
# downloader_class.check_data_availablity()
# downloader_class.query_price_data(downloader_class.instrument_pool, "2017-01-01", "2023-10-25", "1d")
# downloader_class.query_get_many_metric("price_usd", downloader_class.instrument_pool, "2017-01-01", "2023-10-25", "1d")

#query many
# downloader_class.download_all_get_many_metrics(metrics=downloader_class.t1_metrics_reload, instrument_pool=downloader_class.instrument_pool_kucoin, 
#                                                start_date_period="2017-01-01", end_date_period="2023-11-19", frequency="1d", 
#                                                use_last_available_date=False, only_new_assets=None, format=None)

# query special
# downloader_class.download_all_special_metrics(downloader_class.special_metrics_pool_reload, downloader_class.instrument_pool_kucoin,"2018-01-01", "2023-11-30", "1h")


#************************************** CONSOLIDATE METRICS HERE ######################################
# downloader_class.special_metrics_consolidator(metrics = downloader_class.special_metrics_pool_reload, frequency=("1d"))
# metrics_to_reload = pd.DataFrame(columns=["metrics"], data=["exchange_funds_flow"])
# downloader_class.special_metrics_consolidator(metrics = metrics_to_reload, frequency=("1h"))


#***************************************  CREATE SPECIAL METRICS HERE ********************************
# downloader_class.special_metrics_creator(frequency="1d")


# downloader_class.calculate_complexity_score(downloader_class.instrument_pool, "2017-01-01", "2023-10-25", "1d")

#query metric batches
# downloader_class.download_all_get_many_metrics(metrics=downloader_class.t1_metrics_batch1, instrument_pool=downloader_class.instrument_pool_kucoin, 
#                                                start_date_period="2018-01-01", end_date_period="2023-11-05", frequency="1h", 
#                                                use_last_available_date=False, only_new_assets=None, format=None)

# downloader_class.logger.info("############################################################################################################################################")
# downloader_class.logger.info("Download for Batch 1 finished")
# downloader_class.logger.info("############################################################################################################################################")
                                               

# downloader_class.download_all_get_many_metrics(metrics=downloader_class.t1_metrics_batch2, instrument_pool=downloader_class.instrument_pool_kucoin, 
#                                                start_date_period="2018-01-01", end_date_period="2023-11-05", frequency="1h", 
#                                                use_last_available_date=False, only_new_assets=None, format=None)


# downloader_class.logger.info("############################################################################################################################################")
# downloader_class.logger.info("Download for Batch 2 finished")
# downloader_class.logger.info("############################################################################################################################################")
                                               

# downloader_class.download_all_get_many_metrics(metrics=downloader_class.t1_metrics_batch3, instrument_pool=downloader_class.instrument_pool_kucoin, 
#                                                start_date_period="2018-01-01", end_date_period="2023-11-05", frequency="1h", 
#                                                use_last_available_date=False, only_new_assets=None, format=None)

# downloader_class.logger.info("############################################################################################################################################")
# downloader_class.logger.info("Download for Batch 3 finished")
# downloader_class.logger.info("############################################################################################################################################")


# downloader_class.download_all_get_many_metrics(metrics=downloader_class.t1_metrics_batch4, instrument_pool=downloader_class.instrument_pool_kucoin, 
#                                                start_date_period="2018-01-01", end_date_period="2023-11-05", frequency="1h", 
#                                                use_last_available_date=False, only_new_assets=None, format=None)

# downloader_class.logger.info("############################################################################################################################################")
# downloader_class.logger.info("Download for Batch 4 finished")
# downloader_class.logger.info("############################################################################################################################################")

# downloader_class.download_all_get_many_metrics(metrics=downloader_class.t1_metrics_batch5, instrument_pool=downloader_class.instrument_pool_kucoin, 
#                                                start_date_period="2018-01-01", end_date_period="2023-11-12", frequency="1h", 
#                                                use_last_available_date=False, only_new_assets=None, format=None)

# downloader_class.logger.info("############################################################################################################################################")
# downloader_class.logger.info("Download for Batch 5 finished")
# downloader_class.logger.info("############################################################################################################################################")

# downloader_class.download_all_get_many_metrics(metrics=downloader_class.t1_metrics_batch6, instrument_pool=downloader_class.instrument_pool_kucoin, 
#                                                start_date_period="2018-01-01", end_date_period="2023-11-13", frequency="1h", 
#                                                use_last_available_date=False, only_new_assets=None, format=None)

# downloader_class.logger.info("############################################################################################################################################")
# downloader_class.logger.info("Download for Batch 6 finished")
# downloader_class.logger.info("############################################################################################################################################")

# path = r"C:\Users\mauri\Documents\Trading Bot\Python\AlgoTrader\Hist Data Download\SanApi Data\Datasets\Inputs\dataset_base_bitcoin.xlsx"
# btc_data = downloader_class.dataset_loader(dataset_config_path=path)

# print("Success initiating downloader class")