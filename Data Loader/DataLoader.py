import gc 
gc.collect()
import os
import threading
import time
import sys
import json
import pandas as pd
import numpy as np
import kucoin
import ntplib
from time import ctime
import requests
import hashlib
import base64
import hmac
import datetime as dt
from datetime import timedelta
import urllib.parse
from loguru import logger
import configparser
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import joblib
from joblib import Parallel, delayed
import platform
import tables
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO

from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from statsmodels.tsa.seasonal import seasonal_decompose
from functools import lru_cache


from scipy.optimize import minimize
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA, TruncatedSVD
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import OLS
import multiprocessing as mp

# from keras.layers import Input, Dense, Lambda
# from keras.models import Model
# from keras import backend as K
# from keras.losses import mse

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
import tensorflow.keras.backend as K

#################################################################################################################################################################
#
#
#                                                                  OUTSIDE CLASS
#
#################################################################################################################################################################

# Helper functions moved outside the class
def save_transformer(transformer, col_name, method, output_path):
    file_path = os.path.join(output_path, f'{col_name}_{method}_transformer.joblib')
    joblib.dump(transformer, file_path)

def load_transformer(col_name, method, output_path):
    file_path = os.path.join(output_path, f'{col_name}_{method}_transformer.joblib')
    if os.path.exists(file_path):
        return joblib.load(file_path)
    else:
        raise FileNotFoundError(f"No transformer found for column {col_name}")

def initialize_transformer(method, lower_bound_min_max):
    """_summary_

    Args:
        method (_type_): _description_
        lower_bound_min_max (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    transformer_dict = {
        'standardize': StandardScaler(),
        'scale': MinMaxScaler(feature_range=(lower_bound_min_max, 1)),
        'robust_scale': RobustScaler(),
        'power_transform': PowerTransformer(method='yeo-johnson'),
    }
    return transformer_dict.get(method)

def apply_custom_transformation(data, method, lower_bound_min_max):
    smoothing_constant = 1e-16
    if method == 'log_transform':
        return np.log1p(data + smoothing_constant)
    elif method == 'log_then_scale':
        logged = np.log1p(data + smoothing_constant)
        transformer = MinMaxScaler(feature_range=(lower_bound_min_max, 1))
        return transformer.fit_transform(logged)
    elif method == 'log_then_standardize':
        logged = np.log1p(data + smoothing_constant)
        transformer = StandardScaler()
        return transformer.fit_transform(logged)
    elif method == 'difference_transform':
        return pd.Series((data-smoothing_constant).flatten()).pct_change().values.reshape(-1, 1)
    elif method == 'standardize_then_scale':
        transformer = StandardScaler()
        transformed = transformer.fit_transform(data)
        min_max_scaler = MinMaxScaler(feature_range=(lower_bound_min_max, 1))
        return min_max_scaler.fit_transform(data)
    elif method == 'yeo_johnson_then_scale':
        transformer = PowerTransformer(method='yeo-johnson')
        transformed = transformer.fit_transform(data)
        min_max_scaler = MinMaxScaler(feature_range=(lower_bound_min_max, 1))
        return min_max_scaler.fit_transform(transformed)
    elif method == 'yeo_johnson_then_standardize':
        transformer = PowerTransformer(method='yeo-johnson')
        transformed = transformer.fit_transform(data)
        standard_scaler = StandardScaler()
        return standard_scaler.fit_transform(transformed)
    elif method == "square":
        transformed = np.square(data)
        return transformed
    elif method == "square_then_log":
        transformed = np.log1p(np.square(data+smoothing_constant))
        return transformed
    elif method == 'cube_root':
        transformed = np.cbrt(data)
        return transformed
    elif method == 'cube_root_then_scale':
        cubed = np.cbrt(data)
        transformer = MinMaxScaler(feature_range=(lower_bound_min_max, 1))
        return transformer.fit_transform(cubed)
    elif method == 'cube_root_then_standardize':
        cubed = np.cbrt(data)
        transformer = StandardScaler()
        return transformer.fit_transform(cubed)
    else:
        raise ValueError("Unsupported method specified.")

def transform_column(col, data_df, method, lower_bound_min_max, rolling_window, use_roll_win, categorical_columns):
    smoothing_constant = 1e-16
    
    if col in categorical_columns:
        return col, data_df[col].values, None  # Return original if categorical

    transformed_series = None
    current_column_data = data_df[col].copy(deep=True)
    current_column_data = current_column_data.replace([np.inf, -np.inf, np.nan], 0)
    current_column_data = current_column_data + smoothing_constant

    transformer_dict = {
        'standardize': StandardScaler(),
        'normalize': MinMaxScaler(feature_range=(lower_bound_min_max, 1)),
        'scale': RobustScaler(),
        'power_transform': PowerTransformer(method='yeo-johnson'),
    }

    transformer = transformer_dict.get(method)

    if rolling_window is not None and use_roll_win:
        # Apply transformation on a rolling window
        transformed_series = current_column_data.rolling(rolling_window, min_periods=1).apply(
            apply_transformer_on_window, args=(method, lower_bound_min_max, transformer_dict), raw=False
        )

    if transformed_series is not None:
        if isinstance(transformed_series, np.ndarray):
            transformed_series = transformed_series.ravel()
            transformed_series = pd.Series(transformed_series, index=current_column_data.index)
            transformed_series = transformed_series.replace([np.inf, -np.inf, np.nan], 0)
            if transformed_series.ndim == 1:
                return col, transformed_series, transformer
            else:
                return col, transformed_series.ravel(), transformer
        else:
            transformed_series = pd.Series(transformed_series, index=current_column_data.index)
            transformed_series = transformed_series.replace([np.inf, -np.inf], 0)
            return col, transformed_series, transformer
    else:
        return col, current_column_data.values, transformer

def apply_transformer_on_window(window, method, lower_bound_min_max, transformer_dict):
    if method == 'log_transform':
        transformed = np.log1p(window)
    elif method == 'log_then_normalize':
        logged = np.log1p(window)
        transformer = MinMaxScaler(feature_range=(lower_bound_min_max, 1))
        transformed = transformer.fit_transform(logged.values.reshape(-1, 1)).flatten()
    elif method == 'log_then_standardize':
        logged = np.log1p(window)
        transformer = StandardScaler()
        transformed = transformer.fit_transform(logged.values.reshape(-1, 1)).flatten()
    elif method == 'yeo_johnson_then_normalize':
        transformer = PowerTransformer(method='yeo-johnson')
        transformed_yeo = transformer.fit_transform(window.values.reshape(-1, 1))
        min_max_scaler = MinMaxScaler(feature_range=(lower_bound_min_max, 1))
        transformed = min_max_scaler.fit_transform(transformed_yeo).flatten()
    elif method == 'yeo_johnson_then_standardize':
        transformer = PowerTransformer(method='yeo-johnson')
        transformed = transformer.fit_transform(window.values.reshape(-1, 1))
        standard_scaler = StandardScaler()
        transformed = standard_scaler.fit_transform(transformed).flatten()
    elif method == "square":
        transformed = np.square(window)
    elif method == "square_then_log":
        transformed = np.log1p(np.square(window))
    elif method == 'cube_root':
        transformed = np.cbrt(window)
    elif method == 'cube_root_then_normalize':
        logged = np.cbrt(window)
        standard_scaler = MinMaxScaler()
        transformed = standard_scaler.fit_transform(logged.values.reshape(-1, 1)).flatten()
    elif method == 'cube_root_then_standardize':
        logged = np.cbrt(window)
        standard_scaler = StandardScaler()
        transformed = standard_scaler.fit_transform(logged.values.reshape(-1, 1)).flatten()
    elif method in transformer_dict:
        transformer = transformer_dict[method]
        transformed = transformer.fit_transform(window.values.reshape(-1, 1)).flatten()
    else:
        transformed = window.to_numpy().flatten()
    return transformed[-1]

def build_vae(input_dim, latent_dim=2, intermediate_dim=64):
    # Encoder
    inputs = Input(shape=(input_dim,))
    h = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(input_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # VAE model
    vae = Model(inputs, x_decoded_mean)

    # Loss function
    reconstruction_loss = mse(inputs, x_decoded_mean) * input_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    vae.compile(optimizer='adam')
    return vae, Model(inputs, z_mean)


def get_running_environment():
    if 'microsoft-standard' in platform.uname().release:
        return 'wsl'
    elif platform.system() == 'Windows':
        return 'windows'
    else:
        return 'unknown'


def convert_path(path, env):
    if env == 'wsl':
        # Convert Windows path to WSL path
        return path.replace('C:\\', '/mnt/c/').replace('\\', '/')
    elif env == 'windows':
        # Convert WSL path to Windows path
        return path.replace('/mnt/c/', 'C:\\').replace('/', '\\')
    else:
        return path

# Convert path based on environment
def get_converted_path(path):
    return convert_path(path, env)

# Detect environment
env = get_running_environment()

base_path = os.path.dirname(os.path.realpath(__file__))

#################################################################################################################################################################
#
#
#                                                                  PATHS
#
#################################################################################################################################################################


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
    crypto_bot_path = "/mnt/c/Users/mauri/Documents/Trading Bot/Python/AlgoTrader"
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
                AWS_path, GOOGLE_path, histo_data_path,
                data_loader, hist_data_download_kucoin, strategy_path]

# Add paths to sys.path and verify
for path in paths_to_add:
    if path not in sys.path:
        sys.path.append(path)

import mo_utils as utils
import data_downloader_sanpy
from utilsAWS import S3Utility
from utilsGoogleDrive import GoogleDriveUtility
from StrategyEvaluator import StrategyEvaluator
from TAIndicatorCreator import TAIndicatorCreator
from KuCoin_Prices import KuCoinDataDownloader

class DataLoader(data_downloader_sanpy.Data_Downloader_All):
    def __init__(self, logger_input=None, filename=None):
        
        if logger_input is None:
            self.logger = logger
            self.configure_logger()
            
        else:
            self.logger = logger_input

        #config
        config_path = utils.find_config_path() 
        config = utils.read_config_file(os.path.join(config_path,"AlgoTrader_config.ini"))
        
        data_downloader_sanpy.Data_Downloader_All.__init__(self)
        self.data_path = histo_data_path
        self.dataset_input_path = os.path.join(datasets_path,"Inputs")
        self.dataset_output_path = datasets_path

        self.today = dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S")

        #retrieve all assets and create mapping with san coin names
        all_asset_file_path = f"{main_data_files_path}/all_assets.xlsx" 
        self.all_assets = pd.read_excel(os.path.join(main_data_files_path, "all_assets.xlsx"), header=0)
        self.ticker_to_slug_mapping = dict(zip(self.all_assets['ticker'], self.all_assets['slug']))
        
        self.dataset_config_file = filename
        self.dataset_config_file_path = os.path.join(self.dataset_input_path,self.dataset_config_file)
        xls = pd.ExcelFile(self.dataset_config_file_path)

        #init dataset params
        self.dataset_params =  pd.read_excel(xls, sheet_name='params', header=0)

        # Initialize main metrics from 'main query' sheet, ensuring uniqueness
        self.dataset_main_metrics = pd.read_excel(xls, sheet_name='main query', header=0)['metrics'].tolist()
        self.dataset_main_metrics = np.unique(self.dataset_main_metrics)

        # Initialize pooled metrics from 'pooled query' sheet, removing unnamed columns
        self.dataset_pooled_metrics = pd.read_excel(xls, sheet_name='pooled query', header=0)
        self.dataset_pooled_metrics = self.dataset_pooled_metrics.loc[:, ~self.dataset_pooled_metrics.columns.str.contains('^Unnamed:')] if not self.dataset_pooled_metrics.empty else []

        #load params
        self.coins= self.dataset_params["main_coin"].apply(lambda x: x.upper()).tolist()
        self.fiat = self.dataset_params["fiat_coin"].values[0].upper()
        self.symbol = f"{self.coins[0]}-{self.fiat}"
        self.interval = self.dataset_params["frequency"].values[0]
        self.main_asset = self.dataset_params["main_coin"].values[0].upper()
        self.main_asset = self.ticker_to_slug_mapping[self.main_asset]
        self.fiat_coin = self.dataset_params["fiat_coin"].values[0]
        self.symbol = f"{self.main_asset}-{self.fiat_coin}" 
        self.frequency = self.dataset_params["frequency"].values[0]
        self.real_time = self.dataset_params["real_time"].values[0]
        self.start_date = pd.Timestamp(self.dataset_params["start_date"].values[0])
        self.end_date = pd.Timestamp(self.dataset_params["end_date"].values[0])
        self.pred_type = self.dataset_params["pred_type"].values[0]
        
        #preprocessing params
        self.nan_method = self.dataset_params["nan_method"].values[0]
        self.nan_window = self.dataset_params["nan_window"].values[0] if not pd.isnull(self.dataset_params["nan_window"].values[0]) else None

        #encoding params
        self.encode_method = self.dataset_params["encode_method"].values[0]
        
        #missing values params
        self.winsorize_limits = self.dataset_params["winsorize_limits"].values[0] if not pd.isnull(self.dataset_params["winsorize_limits"].values[0])  else None

        #data nature params
        self.use_log = self.dataset_params["use_log"].values[0]
        self.use_relative_change = self.dataset_params["use_relative_change"].values[0]

        #transformation params
        self.transform_method = self.dataset_params["transform_method"].values.tolist() if not np.any(pd.isnull(self.dataset_params["transform_method"].values)) else None
        self.transform_window = self.dataset_params["transform_window"].values[0]
        self.columns_to_transform = self.dataset_params["columns_to_transform"].values[0] if not pd.isnull(self.dataset_params["columns_to_transform"].values[0]) else None
        self.use_roll_window = self.dataset_params["use_rol_window"].values[0] if not pd.isnull(self.dataset_params["use_rol_window"].values[0]) else None
        self.transformers =  {}
        
        #clustering params
        self.clustering_method = self.dataset_params["clustering_method"].values[0] if not pd.isnull(self.dataset_params["clustering_method"].values[0]) else None
        self.explained_variance_threshold = self.dataset_params["explained_variance_threshold"].values[0] if not pd.isnull(self.dataset_params["explained_variance_threshold"].values[0]) else None
        self.scale_method_pca = self.dataset_params["scale_method_pca"].values[0] if not pd.isnull(self.dataset_params["scale_method_pca"].values[0]) else None
        
        #multicollinearity params
        self.multicollinearity_method = self.dataset_params["multicollinearity_method"].values[0] if not pd.isnull(self.dataset_params["multicollinearity_method"].values[0]) else None
        self.multicollinearity_threshold = self.dataset_params["multicollinearity_threshold"].values[0] if not pd.isnull(self.dataset_params["multicollinearity_threshold"].values[0]) else None
        
        #transform price data
        self.add_shifted_price = self.dataset_params["add_shifted_price"].values[0] if not pd.isnull(self.dataset_params["add_shifted_price"].values[0]) else None
        self.add_raw_price = self.dataset_params["add_raw_price"].values[0] if not pd.isnull(self.dataset_params["add_raw_price"].values[0]) else None
        
        #Init kucoin price downloader
        self.kucoin_price_loader = KuCoinDataDownloader()
        
        #init ta indicator creator
        self.TA_creator = TAIndicatorCreator()
        
        self.logger.info("DataLoader initialized")
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
        timestamp = current_datetime.strftime('%d-%m-%Y_%H-%M')  # <--- ist das bewusst so gewählt("%Y-%m-%d %H:%M")  <-- ja um besser nach der runtime zu filtern
        log_directory = "Data Loader log"
        log_directory = os.path.join(logger_path, log_directory)
        
        log_file_name = f"DataLoader_log_{timestamp}.txt"
        log_file_path_data_loader = os.path.join(log_directory, log_file_name)
        
        self.logger.add(log_file_path_data_loader, rotation="500 MB", level="DEBUG")
        


#################################################################################################################################################################
#
#                                                                              VIUSALISATION
#         
#################################################################################################################################################################

    def visualize_distribution(self, df, combined=False):
        if combined:
            fig, ax = plt.subplots(figsize=(10, 6))
            for column in df.columns:
                ax.scatter(df.index, df[column], s=0.2, label=column)
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.set_title('Scatter plot of all columns')
            # ax.legend()
        else:
            num_cols = len(df.columns)
            fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(10, 10*num_cols))
            for i, column in enumerate(df.columns):
                ax = axes[i]
                ax.scatter(df.index, df[column], s=0.2)
                ax.set_xlabel('Index')
                ax.set_ylabel(column)
                ax.set_title(f'Scatter plot of {column}')
        plt.tight_layout()
        plt.show()
        
    def plot_correlation_matrix(self, df, figsize=(55, 60), cmap='RdYlGn_r', vmax=1.0, vmin=-1.0):
        """
        Plots the correlation matrix of the input DataFrame and saves it to the specified path.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame containing the factors.
        save_path (str): Path to save the plot.
        figsize (tuple): Size of the figure to plot.
        cmap (str): Colormap to use for the heatmap.
        vmax (float): Maximum value for the colormap.
        vmin (float): Minimum value for the colormap.
        """
        # Compute the correlation matrix
        corr_matrix = df.corr()
        save_path = os.path.join(data_path_crypto,"Data Analysis","correlation matrices",f'{self.dataset_config_file.replace(".xlsx","")}_correlations.jpg')
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        
        # Set up the matplotlib figure
        sns.set(font_scale=2.5)
        sns.set_style('whitegrid')
        
        plt.figure(figsize=figsize)
        
        # Draw the heatmap with the mask and correct aspect ratio
        ax = sns.heatmap(corr_matrix, cmap=cmap, vmax=vmax, vmin=vmin, mask=mask,
                    linewidths=0.5, square=False, xticklabels=True, yticklabels=True)
        
        # Customize the ticks
        plt.yticks(rotation=0, fontsize=30)
        plt.xticks(rotation=90, fontsize=25)
        plt.autoscale(enable=True, tight=True)
        fig = ax.get_figure()

        if os.path.exists(save_path):
            os.remove(save_path)

        plt.savefig(save_path, dpi=300)
        
        plt.close(fig)
    

#################################################################################################################################################################
#
#                                                                              LOAD DATASET & DATASET IMPUTATION
#         
#################################################################################################################################################################    
    
    
    def load_dataset_from_config(self, coin=None, start=None, end=None):
        dataset_index = pd.date_range(start=start, end=end, freq=self.frequency)
        dataset_index = pd.to_datetime(dataset_index, utc=True)
        dataset = pd.DataFrame(index=dataset_index)
        failed_data_queries = []
        
        # The main change here is using ProcessPoolExecutor instead of ThreadPoolExecutor
        with ProcessPoolExecutor(max_workers=2) as executor:    #1 os.cpu_count() - 4
            futures = []
            try:
                for metric in self.dataset_main_metrics:
                    futures.append(executor.submit(DataLoader.load_and_process_metric, self.data_path, coin.upper(), self.fiat.upper(), self.frequency, metric, dataset_index.tolist(), [self.main_asset]))

                if len(self.dataset_pooled_metrics) > 0:
                    for metric in self.dataset_pooled_metrics.columns:
                        assets = self.dataset_pooled_metrics[metric].dropna().tolist()
                        slug_assets = [self.ticker_to_slug_mapping.get(ticker.upper(), ticker) for ticker in assets]
                        futures.append(executor.submit(DataLoader.load_and_process_metric, self.data_path, coin.upper(), self.fiat.upper(), self.frequency, metric, dataset_index.tolist(), slug_assets))

                for future in as_completed(futures):
                    metric, result_df = future.result()
                    try:
                        if result_df is not None:
                            if "ema" in metric and "extrema" not in metric:
                                result_df.rename(columns={result_df.columns[0]: metric}, inplace=True)
                            if "ma" in metric and "extrema" not in metric:
                                result_df.rename(columns={result_df.columns[0]: metric}, inplace=True)
                            if "wma" in metric and "extrema" not in metric:
                                result_df.rename(columns={result_df.columns[0]: metric}, inplace=True)
                            if "extrema" in metric and "rsi" in metric:
                                result_df.rename(columns={result_df.columns[0]: f"{result_df.columns[0]}_rsi",result_df.columns[1]: f"{result_df.columns[1]}_rsi"}, inplace=True)
                            
                            if "extrema" in metric and "rsi" not in metric:
                                result_df.rename(columns={result_df.columns[0]: f"{result_df.columns[0]}_price",result_df.columns[1]: f"{result_df.columns[1]}_price"}, inplace=True)
                                
                            dataset = pd.concat([dataset, result_df], axis=1)
                        else:
                            failed_data_queries.append(metric)
                    except Exception as e:
                        logger.error(f"Error processing future: {e}")
                        failed_data_queries.append(future)
            except Exception as e:
                self.logger.error(f"Exception occurred: {e}")
                for future in futures:
                    future.cancel()
                    
        dataset = dataset.loc[:, ~dataset.columns.duplicated()]
        dataset.drop(columns=["localized"], inplace=True, errors='ignore')
        dataset.index = dataset.index.tz_localize(None)
        dataset = dataset[dataset.index<=self.end_date]
        
        for col in dataset.columns:
            if dataset[col].isnull().all():
                logger.error(f"Column {col} is all NaN. Dropping column.")
                dataset.drop(columns=col, inplace=True)
            elif dataset[col].value_counts(normalize=True).max() > 0.99:
                logger.error(f"Column {col} has more than 99% of the values the same. Dropping column.")
                dataset.drop(columns=col, inplace=True)
        
        failed_data_df = pd.DataFrame(np.unique(failed_data_queries), columns=['Failed Metrics'])
        
        failed_data_df_path = os.path.join(self.dataset_output_path, "failed metrics", f"failed_metrics_dataset_{self.main_asset}.xlsx")    
        failed_data_df.to_excel(failed_data_df_path, index=False)
    

        dataset.sort_index(inplace=True)
        dataset.sort_index(axis=1, inplace=True)
        dataset.dropna(axis=1, how="all", inplace=True)
        
        # Extend each column name by the coin
        dataset.columns = [f"{col}_{coin}" for col in dataset.columns]
    
        return dataset

    def reorder_dataset_columns(self, dataset):
        # Reorder columns if they exist in the DataFrame
        price_usdt_col_name = dataset.columns[dataset.columns.str.contains("price_usdt_")]
   
        columns_order = ['open', 'high', 'low', 'close', 'turnover', "volume", 'price_shifted']
        existing_columns = [col for col in columns_order if col in dataset.columns]
        remaining_columns = [col for col in dataset.columns if col not in existing_columns and col not in ['close', 'price_raw', 'y_pred']]
        
        if self.add_raw_price:
            dataset = dataset[remaining_columns + existing_columns + ["price_raw", 'y_pred']]
        else:
            dataset = dataset[remaining_columns + existing_columns + ['y_pred']]
    
        return dataset
        
    def save_dataset(self, dataset):
        try:
            filename = self.dataset_config_file.split(".")[0] + "_processed"
            date = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
            alt_key = f"dataset_{self.main_asset}_{self.frequency}_{date}"
            key = filename if filename else alt_key

            save_path_hdf = os.path.join(self.dataset_output_path, "crypto datasets", "hdf5")

            if not os.path.exists(save_path_hdf):
                os.makedirs(save_path_hdf)

            file_save_path_hdf = os.path.join(save_path_hdf, f"{key}.h5")

            # Create a copy of the dataset to avoid fragmentation
            dataset_copy = dataset.copy()

            # Check if the key exists in the HDF5 file
            with tables.open_file(file_save_path_hdf, mode='a') as h5file:
                node_path = f'/{key}'
                if node_path in h5file:
                    try:
                        h5file.remove_node(node_path, recursive=True)
                    except tables.NoSuchNodeError:
                        print(f"Node {node_path} does not exist, no need to remove.")
                    except Exception as e:
                        print(f"Error removing node {node_path}: {e}")
                        return

            # Save the dataset using keyword arguments
            dataset_copy.to_hdf(path_or_buf=file_save_path_hdf, key=key, mode='a', data_columns=True, complib='zlib', format='table')

            save_path_csv = os.path.join(self.dataset_output_path, "crypto datasets", "csv")
            if not os.path.exists(save_path_csv):
                os.makedirs(save_path_csv)

            dataset_copy.to_csv(os.path.join(save_path_csv, f"{key}.csv"),sep=",")

        except Exception as e:
            print(f"An error occurred while saving the dataset: {e}")
    
    def get_indicator_list():
        
        list = ['ma','ema','short_ema', 'long_ema', 'high_ema', 'low_ema', 'close_ema', 'open_ema', 'volume_ema', 
                'macd', 'bollinger', 'bbandwith','hurst', 'vwap', 'supertrend', 'atr', 'adx', 'rsi', 'volume', 'ao', 'cti', 
                'dm', 'fisher', 'willr', "candlestick_patterns", "extrema","chop","kama", "rollvol"]
        
        return list 
    
    def get_pattern_list():

        patterns = [
                    "2crows", "3blackcrows", "3inside", "3linestrike", "3outside", "3starsinsouth",
                    "3whitesoldiers", "abandonedbaby", "advanceblock", "belthold", "breakaway",
                    "closingmarubozu", "concealbabyswall", "counterattack", "darkcloudcover", "doji",
                    "dojistar", "dragonflydoji", "engulfing", "eveningdojistar", "eveningstar",
                    "gapsidesidewhite", "gravestonedoji", "hammer", "hangingman", "harami",
                    "haramicross", "highwave", "hikkake", "hikkakemod", "homingpigeon",
                    "identical3crows", "inneck", "inside", "invertedhammer", "kicking", "kickingbylength",
                    "ladderbottom", "longleggeddoji", "longline", "marubozu", "matchinglow", "mathold",
                    "morningdojistar", "morningstar", "onneck", "piercing", "rickshawman",
                    "risefall3methods", "separatinglines", "shootingstar", "shortline", "spinningtop",
                    "stalledpattern", "sticksandwich", "takuri", "tasukigap", "thrusting", "tristar",
                    "unique3river", "upsidegap2crows", "xsidegap3methods"
                ]
        
        return patterns

    @staticmethod
    def generate_data_path_and_file_name(data_path, coin, fiat, metric, frequency, indicators, indicator_to_query):
        
        if metric in ['nvtRatioCirculation', 'nvtRatioTxVolume']:
            data_path = os.path.join(data_path, 'nvt_ratio', frequency, metric)
            file_name = f"nvt_ratio_{frequency}_{metric}.csv"
        elif metric in ["closePriceUsd", "highPriceUsd", "lowPriceUsd", "openPriceUsd"]:
            data_path = os.path.join(data_path, 'ohlc', frequency, metric)
            file_name = f"ohlc_{frequency}_{metric}.csv"
        elif indicator_to_query in "extrema" and indicator_to_query not in ["ema","ma","wma"]:
            data_path = os.path.join(data_path, metric, frequency)
            file_name = f"{metric}_{coin}_{fiat}_{frequency}.csv" 
        elif indicator_to_query in "kama" and indicator_to_query not in ["ema","ma","wma"]:
            data_path = os.path.join(data_path, metric, frequency)
            file_name = f"{metric}_{coin}_{fiat}_{frequency}.csv"
        elif indicator_to_query in indicators or metric == "price_usdt_kucoin":
            data_path = os.path.join(data_path, metric, frequency)
            file_name = f"{metric}_{coin}_{fiat}_{frequency}.csv"
        elif indicator_to_query in ["close", "high", "low", "open", "volume"]:
            data_path = os.path.join(data_path, "price_usdt_kucoin", frequency)
            file_name = f"price_usdt_kucoin_{coin}_{fiat}_{frequency}.csv"
        elif indicator_to_query in DataLoader.get_pattern_list():
            data_path = os.path.join(data_path, "candlestick_patterns", frequency)
            file_name = f"candlestick_patterns_{frequency}.csv"
        else:
            data_path = os.path.join(data_path, metric, frequency)
            file_name = f"{metric}_{frequency}.csv"
        return data_path, file_name

    @staticmethod
    def load_and_process_metric(data_path, coin, fiat, frequency, metric, dataset_index, asset_pool=None):
        indicators = DataLoader.get_indicator_list()
        candle_patterns = DataLoader.get_pattern_list()
        
        indicator_to_query = metric.split("_")[0]   #get only the indicator to adjust for e.g. ema_high_20 metrics to ema
        
        data_path, file_name = DataLoader.generate_data_path_and_file_name(data_path, coin, fiat, metric, frequency, indicators, indicator_to_query)
        full_path = os.path.join(data_path, file_name)
        
        if not os.path.exists(full_path):
            logger.error(f"FILE NON EXISTING: The data is not available for {metric} under {data_path} . Please investigate")
            return metric, None  # Indicating failure to load
        
        df = pd.read_csv(full_path)
        
        if asset_pool and indicator_to_query not in indicators and metric != "price_usdt_kucoin" and metric not in ["close", "high", "low", "open", "volume"]: #get metrics from consolidated files
            missing_assets = [asset for asset in asset_pool if asset not in df.columns]
            if missing_assets:
                logger.error(f"ASSET NON EXISTING: The assets {', '.join(missing_assets)} are not available for {metric}. Please investigate")
                asset_pool = [asset for asset in asset_pool if asset in df.columns]
                return metric, None  # Immediately return when the asset is not in the columns
            
            df.index = pd.to_datetime(df.index, utc=True)
            df = df[asset_pool]
            df.columns = [f"{metric}_{col}" for col in df.columns]

            return metric, df.reindex(dataset_index)
        elif indicator_to_query in indicators or metric == "price_usdt_kucoin" :    #get single indicators or price_usdt_kucoin
            df.set_index(df.columns[0],inplace=True)
            df.index = pd.to_datetime(df.index, utc=True)
            
            if metric != "price_usdt_kucoin" and not indicator_to_query in ["extrema"] and "bandwith" not in indicator_to_query:
                df = df.iloc[:,1:].copy()
            elif "bandwith" in metric:
                df = df.loc[:,["bb_bandwith"]].copy()
            else:
                df = df.copy()
                
            # Ensure dataset_index is a timezone-aware datetime index
            if not isinstance(dataset_index, pd.DatetimeIndex):
                dataset_index = pd.to_datetime(dataset_index, utc=True, format='%Y-%m-%d %H:%M:%S') #.tz_localize(None)
            
            df = df[df.index.isin(dataset_index)]
            
            if indicator_to_query in ["extrema"]:
                # df.ffill(inplace=True)   
                df = df.notna().astype(int)
            
            return metric, df
        elif indicator_to_query in candle_patterns:             #get candlestick patterns
            df.set_index(df.columns[0],inplace=True)
            df.index = pd.to_datetime(df.index, utc=True)
            
            df = df.copy()
            
            df = df[indicator_to_query].copy()  

            # Ensure dataset_index is a timezone-aware datetime index
            if not isinstance(dataset_index, pd.DatetimeIndex):
                dataset_index = pd.to_datetime(dataset_index, utc=True, format='%Y-%m-%d %H:%M:%S')
            
            df = df[df.index.isin(dataset_index)]

            return metric, df

        elif indicator_to_query in ["close", "high", "low", "open", "volume"]:        #get specific ohlc data
            df.set_index(df.columns[0],inplace=True)
            df.index = pd.to_datetime(df.index, utc=True)
            df = df[indicator_to_query].copy()

            # Ensure dataset_index is a timezone-aware datetime index
            if not isinstance(dataset_index, pd.DatetimeIndex): 
                dataset_index = pd.to_datetime(dataset_index, utc=True, format='%Y-%m-%d %H:%M:%S') #.tz_localize(None)
            
            df = df[df.index.isin(dataset_index)]
            
            return metric, df
      
        else:
            logger.error(f"No assets provided for metric {metric}.")
            return metric, None

    def replace_inf_and_fill_zeros(self, df, replace_0=False):
        """
        Replaces infinite values with 0, then forward fills and backward fills all 0 values.
        
        Parameters:
        - df: pandas DataFrame
        
        Returns:
        - df_processed: DataFrame with infinite values replaced and 0 values filled
        """
        df_processed = df.replace([np.inf, -np.inf], 0)  # Replace inf and -inf with 0
        
        # Forward fill then backward fill 0 values
        if replace_0:
            for col in df_processed.columns:
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    # Create masks for 0 values
                    zero_mask = df_processed[col] == 0
                    # Forward fill
                    df_processed[col] = df_processed[col].mask(zero_mask).ffill().fillna(0)
                    # Backward fill any remaining 0 values that were not filled by ffill
                    df_processed[col] = df_processed[col].mask(zero_mask).bfill().fillna(0)
        return df_processed


    def adjust_missing_values(self, df, method=None, window_size=None):
        """
        Fills missing values in a DataFrame using various methods suitable for time-series data.
        
        Parameters:
        - df: pandas DataFrame
        - method: string, method for filling missing values ('linear', 'ffill', 'bfill', 'rolling_mean', 'time')
        - window_size: int, size of the window used for the rolling mean method
        
        Returns:
        - df_filled: DataFrame with missing values filled
        """
        
        if window_size is None:
            window_size = 3
        else:
            window_size = int(window_size)
        # Make a copy of the DataFrame to avoid modifying the original data
        df_filled = df.copy()
        
        if method == "drop":

            if window_size == 0:
                df_filled.dropna(inplace=True)
            else:
                df_filled = df_filled.dropna(axis=1, how="all")
                
                first_valid_index = df_filled.dropna(how="any").index[0]
                
                df_filled = df_filled.loc[first_valid_index:]
                            
                # Identify rows with consecutive NaNs
                is_nan = df_filled.isna().all(axis=1)
                nan_groups = is_nan.astype(int).groupby((is_nan != is_nan.shift()).cumsum()).cumsum()
                
                # Drop rows where the number of consecutive NaNs is greater than or equal to 3
                df_filled = df_filled[~((nan_groups >= window_size) & is_nan)]
                
            return df_filled
        
        # Loop through each column in DataFrame
        for col in df_filled.columns:
            if method == 'linear':
                df_filled[col] = df_filled[col].interpolate(method='linear')
            elif method == 'ffill':
                df_filled[col] = df_filled[col].ffill()
            elif method == 'bfill':
                df_filled[col] = df_filled[col].bfill()
            elif method == 'rolling_mean':
                if pd.api.types.is_numeric_dtype(df_filled[col]):
                    df_filled[col] = self.rolling_mean_imputation(series=df_filled[col], window=window_size)
                else:
                    # Fallback for non-numeric columns
                    df_filled[col] = df_filled[col].ffill()
                    df_filled[col] = df_filled[col].bfill()
            elif method == 'time':
                if df_filled.index.is_all_dates:
                    df_filled[col] = df_filled[col].interpolate(method='time')
                else:
                    raise ValueError("DataFrame index must be datetime type for 'time' interpolation")
            else:
                raise ValueError(f"Unsupported method: {method}")
        
            # For any remaining missing values, use forward fill followed by backward fill as a final step
            df_filled[col] = df_filled[col].ffill()
            df_filled[col] = df_filled[col].bfill()
        
        df_filled.replace(np.nan, 0, inplace=True)
        
        return df_filled

    # Define a function for rolling mean imputation
    def rolling_mean_imputation(self, series, window):
        return series.fillna(series.rolling(window, min_periods=1, center=True).mean())


    
#################################################################################################################################################################
#
#                                                                              WINSORIZE
#         
#################################################################################################################################################################

    def winsorize(self, data_df=None, wins_all_cols=True, cols=None, limits=None, adjust_abnormal_values=False):
        
        if wins_all_cols:
            cols = data_df.columns
        else:
            cols = cols

        winsorized_df = data_df.copy(deep=True)
        # limit_dict = {limits, limits}

        for c in cols:
            if adjust_abnormal_values:
                if "price_" in c and "usdt" in c:
                    winsorized_df = self.adjust_abnormal_values(winsorized_df, cols=[c])

            if pd.api.types.is_numeric_dtype(winsorized_df[c]):
                goods = winsorized_df[c].notna()
                winsorized_df.loc[goods, c] = winsorize(winsorized_df.loc[goods, c], limits=limits)
            else:
                self.logger(f"Warning: Column '{c}' is categorical. Winsorization skipped.")

        return winsorized_df
    
    def adjust_abnormal_values(self, data_df=None, cols=None):
        
        if cols is None:
            cols = data_df.columns
        else:
            cols = cols
        
        df = data_df.copy(deep=True)
        
        for c in cols:
            std_dev = np.std(df[c])
            for i in range(0, len(df[c])):

                if i == 0:
                    continue

                value_t = df[c].iloc[i]
                value_tm1 = df[c].iloc[i-1]

                if value_tm1 != 0 and np.logical_or(abs((value_t - value_tm1) / value_tm1) > 3, ((value_t - value_tm1) / value_tm1) < -0.95):
                    df.loc[df.index[i], c] = value_tm1
                    self.logger.info(f"ADJUSTING ABNORMAL VALUES IN {c}: Value change is too high at {df.index[i]}")
                    
        return df
#################################################################################################################################################################
#
#                                                                              PREDICTING VARIABLE
#         
#################################################################################################################################################################
            
    def create_predicting_variable(self,df, method='price', periods=1, frequency=None):
        """
        Transforms a DataFrame column based on the specified method and shifts it by 'n' periods.

        Parameters:
        - df: pandas.DataFrame containing the data.
        - column: str, the name of the column to transform.
        - method: str, the method of transformation ('price', 'return', 'return quintile', 'return quartile').
        - periods: int, the number of periods to shift the transformed data.

        Returns:
        - pandas.Series: The transformed and shifted column.
        """
        pred_column = self.price_key
        frequency = self.frequency
        
        if method == 'price':
            transformed = df[pred_column].copy(deep=True)
        
        elif method == 'return':
            transformed = df[pred_column].copy(deep=True).pct_change()
            
        elif method =="4h_return":
            if frequency == "1h":
                shift_window = 4
            elif frequency == "5m":
                shift_window = 48
            elif frequency == "15m":
                shift_window = 16
            
            transformed = self.calculate_return(df[pred_column].copy(deep=True), shift_window)
            transformed = transformed.dropna()
        
        elif method == "12h_return":
            
            if frequency == "1h":
                shift_window = 12
            elif frequency == "5m":
                shift_window = 96
            elif frequency == "15m":
                shift_window = 32
            
            transformed = self.calculate_return(df[pred_column].copy(deep=True), shift_window)
            transformed = transformed.dropna()
        
        elif method =="1d_return":
            if frequency == "1h":
                shift_window = 24
            elif frequency == "5m":
                shift_window = 288
            elif frequency == "15m":
                shift_window = 96
            
            transformed = self.calculate_return(df[pred_column].copy(deep=True), shift_window)
            transformed = transformed.dropna()
        
        elif method == 'direction':
            returns = df[pred_column].copy(deep=True).pct_change()
            transformed = pd.cut(returns, bins=[-np.inf, -0.0075, 0.0075, np.inf], labels=[-1, 0, 1], include_lowest=True)
        
        elif method == "4h_return_direction":
            
            if frequency == "1h":
                shift_window = 4
            elif frequency == "5m":
                shift_window = 48
            elif frequency == "15m":
                shift_window = 16
            
            transformed = self.calculate_return(df[pred_column].copy(deep=True), shift_window)
            transformed = transformed.dropna()
            transformed = transformed.apply(lambda x: 1 if x > 0.01 else (-1 if x < -0.01 else 0))
            transformed = transformed.replace(np.nan, 0)
            #pd.cut(transformed, bins=[-np.inf, -0.01, 0.01, np.inf], labels=[-1, 0, 1], include_lowest=True)
        
        elif method == "12h_return_direction":
            
            if frequency == "1h":
                shift_window = 12
            elif frequency == "5m":
                shift_window = 96
            elif frequency == "15m":
                shift_window = 32
            
            transformed = self.calculate_return(df[pred_column].copy(deep=True), shift_window)
            transformed = transformed.dropna()
            transformed = transformed.apply(lambda x: 1 if x > 0.01 else (-1 if x < -0.01 else 0))
            transformed = transformed.replace(np.nan, 0)
            
        elif method == "1d_return_direction":
            
            if frequency == "1h":
                shift_window = 24
            elif frequency == "5m":
                shift_window = 288
            elif frequency == "15m":
                shift_window = 96
            
            transformed = self.calculate_return(df[pred_column].copy(deep=True), shift_window)
            transformed = transformed.dropna()
            transformed = transformed.apply(lambda x: 1 if x > 0.01 else (-1 if x < -0.01 else 0))
            transformed = transformed.replace(np.nan, 0)
        
        elif method in ['return_quintile', 'return_quartile']:
            returns = df[pred_column].copy(deep=True).pct_change()
            if method == 'return_quintile':
                transformed = pd.qcut(returns, 5, labels=False, duplicates='drop')
            else:  # return quartile
                transformed = pd.qcut(returns, 4, labels=False, duplicates='drop')
        else:
            raise ValueError("Method not recognized. Choose 'price', 'return', 'return quintile', or 'return quartile'.")

        # Shift the transformed data by 'n' periods. Use .shift(-periods) for forward-looking prediction.
        transformed = transformed.shift(-periods)
        transformed.ffill(inplace=True)
        transformed = transformed.astype(float)
        
        # df = df.dropna(subset=["Y_pred"]

        return transformed
    
    # def calculate_cumulative_return(self, df, periods):
    #     df_copy2 = df.copy(deep=True)
    #     df_copy2 = (df_copy2.shift(-periods) / df_copy2) -1
    
    #     return df_copy
    
    def calculate_cumulative_return(self, df, periods):
        df_copy = df.copy(deep=True)
        
        # Calculate individual step returns for each period
        df_step_returns = df_copy.pct_change().fillna(0) + 1
        
        # Use rolling window to calculate the cumulative product over the specified periods
        df_cumulative_return = df_step_returns.rolling(window=periods).apply(lambda x: x.prod(), raw=True) - 1
        
        return df_cumulative_return.shift(-periods)

    def calculate_return(self, df, periods):
        df_copy = df.copy(deep=True)
        
        df_copy = df.copy(deep=True)
        df_copy = (df_copy.shift(-periods) / df_copy) - 1
        return df_copy

#################################################################################################################################################################
#
#                                                                              MULTICOLLINEARITY
#         
#################################################################################################################################################################
    
    @staticmethod
    @lru_cache(maxsize=None)
    def calculate_vif_cached(values_tuple, index):
        values_array = np.array(values_tuple)
        return variance_inflation_factor(values_array, index)
    
    @staticmethod
    def calculate_vif_wrapper(args):
        return DataLoader.calculate_vif_cached(*args)
    

    def remove_multicollinearity(self, df, method='vif', threshold=5.0):
        # Create a copy of the DataFrame
        df_copy = df.copy(deep=True)
        
        # List to hold removed variables
        removed_variables = []
        
        if method == 'vif':
            df_values = df_copy.values
            columns = np.array(df_copy.columns)
            
            while True:
                vif = pd.DataFrame()
                vif["variables"] = columns
                
                # Convert df_values to a hashable type (tuple) for caching
                values_tuple = tuple(map(tuple, df_values))
                
                # Use multiprocessing to calculate VIF with increased timeout
                with mp.Pool(mp.cpu_count() - 4) as pool:
                    try:
                        vif["VIF"] = pool.map(DataLoader.calculate_vif_wrapper, [(values_tuple, i) for i in range(df_values.shape[1])], chunksize=1)
                    except mp.TimeoutError:
                        print("Multiprocessing timeout occurred. Increasing timeout and retrying...")
                        vif["VIF"] = pool.map_async(DataLoader.calculate_vif_wrapper, [(values_tuple, i) for i in range(df_values.shape[1])], chunksize=1).get(999999)
                
                # Sort variables by VIF in descending order
                vif = vif.sort_values("VIF", ascending=False)
                
                # If max VIF is below threshold, break the loop
                if np.isfinite(vif["VIF"].max()): 
                    if vif["VIF"].max() <= threshold:
                        break
                
                # Identify variable with highest VIF
                remove_idx = np.argmax(vif["VIF"].values)
                remove = columns[remove_idx]
                
                # Remove variable from DataFrame
                df_values = np.delete(df_values, remove_idx, axis=1)
                columns = np.delete(columns, remove_idx)
                self.logger.info(f"Removing variable '{remove}' with VIF of {vif['VIF'].iloc[remove_idx]:.2f}. Max VIF is: {max(vif['VIF'])}. Remaining len of columns: {len(columns)}")
                
                # Add removed variable to list
                removed_variables.append(remove)
            
                # Create a final DataFrame with remaining columns
                df_copy = pd.DataFrame(df_values, columns=columns)
                
                # Save the metrics that have not been removed due to high VIF
                dataset_config_file = self.dataset_config_file.replace('.xlsx', '')
                metrics_file_path = os.path.join(self.dataset_output_path, "crypto datasets", f"{dataset_config_file}_vif_metrics.csv")
                
                vif.to_csv(metrics_file_path, index=False)

            
            return df_copy, removed_variables
        
        elif method == 'stepwise':
            # Initialize list of included variables
            included = []
            
            # Initialize AIC for comparison purposes
            best_aic = np.inf
            
            # Continue while we can improve the model by adding a variable
            while True:
                # List to hold temporary results for each variable
                results = []
                
                # Loop through columns to identify the variable that will improve the model the most
                for col in df_copy.columns:
                    if col not in included:
                        # Fit OLS model with the current set of included variables and the new variable
                        try:
                            X = df_copy[included + [col]]
                            X = sm.add_constant(X)  # Add constant term for OLS
                            y = np.ones(len(X))  # Placeholder for the dependent variable
                            
                            model = OLS(y, X).fit()
                            
                            # Calculate the AIC of the model
                            aic = model.aic
                            
                            # Store the result (AIC and column name)
                            results.append((aic, col))
                        except np.linalg.LinAlgError:
                            # Handle singular matrix cases where the OLS model cannot be fit
                            self.logger.warning(f"Singular matrix encountered for column '{col}', skipping.")
                            continue
                
                # Check if results is empty before calling min()
                if not results:
                    self.logger.warning("No more variables to include, stopping stepwise selection.")
                    break
                
                # Identify the variable that improved the model the most (i.e., lowest AIC)
                best_aic_current, best_col = min(results, key=lambda x: x[0])
                
                # If the new variable improves the model (lower AIC), add it to the included list
                if best_aic_current < best_aic:
                    included.append(best_col)
                    best_aic = best_aic_current
                    self.logger.info(f"Adding variable '{best_col}' with AIC of {best_aic_current:.4f}.")
                else:
                    # If no improvement, break the loop
                    self.logger.info(f"No improvement in AIC. Stopping selection.")
                    break
            
            # Remove variables not included from the final DataFrame
            removed_variables = list(set(df_copy.columns) - set(included))
            df_copy = df_copy[included]
        
        return df_copy, removed_variables

#################################################################################################################################################################
#
#                                                                              ENCODING
#         
#################################################################################################################################################################

    def encode_dataframe(self, dataframe, method='onehot'):
        
        dataframe.columns = [str(col) for col in dataframe.columns]
        encoded_columns = []  # Initialize a list to store names of encoded columns
        
        if method not in ['label', 'onehot', 'sine_cosine']:
            raise ValueError("Method must be 'label', 'onehot', or 'sine_cosine'")
        
        categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        
        if not categorical_columns and method in ['label', 'onehot']:
            # No categorical columns to encode, return original DataFrame
            self.logger.info("No categorical features found for encoding.")
            return dataframe, encoded_columns
        
        if method == 'label':
            df_encoded = dataframe.copy()
            label_encoder = LabelEncoder()
            for column in categorical_columns:
                df_encoded[column] = label_encoder.fit_transform(df_encoded[column])
                encoded_columns.append(column)  # Add the encoded column name
            
        elif method == 'onehot':
            # Define the transformer only for categorical columns
            column_transformer = ColumnTransformer([
                ('cat', OneHotEncoder(drop='first'), categorical_columns)],
                remainder='passthrough')
            
            # Convert dataframe to a DataFrame with string columns to ensure compatibility
            dataframe = dataframe.astype({col: 'category' for col in categorical_columns})

            array_encoded = column_transformer.fit_transform(dataframe)
            array_encoded = array_encoded if isinstance(array_encoded, np.ndarray) else array_encoded.toarray()
            
            # Handling column names manually for OneHotEncoder
            new_columns = [f"{col}_{val}" for col in categorical_columns for val in sorted(dataframe[col].cat.categories)[1:]]
            all_columns = new_columns + numeric_columns
            
            df_encoded = pd.DataFrame(array_encoded, columns=all_columns, index=dataframe.index)
            encoded_columns.extend(new_columns)  # Update with new column names from one-hot encoding
            
        elif method == 'sine_cosine':
            df_encoded = dataframe.copy()
            for column in ['day_of_week', 'month']:
                if column in df_encoded.columns:
                    df_encoded[f'sin_{column}'] = np.sin(2 * np.pi * df_encoded[column]/df_encoded[column].max())
                    df_encoded[f'cos_{column}'] = np.cos(2 * np.pi * df_encoded[column]/df_encoded[column].max())
                    df_encoded.drop(column, axis=1, inplace=True)
                    encoded_columns.extend([f'sin_{column}', f'cos_{column}'])  # Add sine-cosine encoded column names
        else:
            raise ValueError("Invalid encoding method specified.")
        return df_encoded, encoded_columns

#################################################################################################################################################################
#
#                                                                              CLUSTERING
#         
#################################################################################################################################################################

    
    
    def clustering(self, data_df, variance_threshold=0.85, method='pca', scale_method='standardize', min_variance_contribution=0.0075):
        date_index = data_df.copy().index

        if method == 'vae':
            scaler = StandardScaler()
            data_df = scaler.fit_transform(data_df)
            
            vae, encoder = build_vae(input_dim=data_df.shape[1])
            vae.fit(data_df, epochs=50, batch_size=256, shuffle=True)
            transformed_data = encoder.predict(data_df)
            explained_variance = np.var(transformed_data, axis=0).sum() / np.var(data_df, axis=0).sum()
            n_components = transformed_data.shape[1]
        else:
            # Scaling the data
            if scale_method == 'standardize':
                scaler = StandardScaler()
                data_df = scaler.fit_transform(data_df)
            elif scale_method == 'yeo_johnson':
                scaler = PowerTransformer(method='yeo-johnson')
                data_df = scaler.fit_transform(data_df)
            else:
                transformer = PowerTransformer(method='yeo-johnson')
                transformed = transformer.fit_transform(data_df)
                standard_scaler = StandardScaler()
                data_df = standard_scaler.fit_transform(transformed)
            
            n_components = 0
            explained_variance = 0
            previous_explained_variance = 0

            while explained_variance <= variance_threshold:
                n_components += 1

                if method == 'pca':
                    transformer = PCA(n_components=n_components)
                elif method == 'svd':
                    transformer = TruncatedSVD(n_components=n_components)
                else:
                    raise ValueError("Method must be 'pca', 'svd', or 'vae'")

                transformed_data = transformer.fit_transform(data_df)
                explained_variance = np.cumsum(transformer.explained_variance_ratio_)[-1]

                # Check the contribution of the new component
                if explained_variance - previous_explained_variance < min_variance_contribution:
                    n_components -= 1
                    break

                previous_explained_variance = explained_variance

            transformer = TruncatedSVD(n_components=n_components)
            transformed_data = transformer.fit_transform(data_df)
            explained_variance = np.cumsum(transformer.explained_variance_ratio_)[-1]

        self.logger.info(f"Optimal number of components: {n_components}, Explained variance ratio: {explained_variance}")

        transformed_df = pd.DataFrame(data=transformed_data, columns=[f'{method}{i+1}' for i in range(n_components)])
        transformed_df.index = date_index
        return transformed_df
    
#################################################################################################################################################################
#
#                                                                              DATA TRANSFORMATION
#         
#################################################################################################################################################################

    
    
    def transform_data(self, data_df=None, method="scale", columns_to_transform=None, columns_to_detrend=None, rolling_window=None, use_roll_win=False, categorical_columns=None):
        
        lower_bound_min_max = -1
        if categorical_columns is None:
            categorical_columns = []
        
        if isinstance(data_df,pd.DataFrame):
            data_df.columns = [str(col) for col in data_df.columns]
            if columns_to_transform is None:
                columns_to_transform = [col for col in data_df.columns if col not in categorical_columns]
        elif isinstance(data_df, pd.Series):
            data_df = pd.DataFrame(data_df)
            data_df.columns = [data_df.columns[0]]
            columns_to_transform = [data_df.columns[0]]
        
        # Replace NaN, inf, and -inf values with 0
        data_df = data_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        transformed_df = data_df.copy(deep=True)

        if not os.path.exists(transformer_path):
            os.makedirs(transformer_path)

        if use_roll_win:
            # Using joblib's Parallel to parallelize column transformations
            results = Parallel(n_jobs=-4, backend='loky')(delayed(transform_column)(col, transformed_df[[col]], method, lower_bound_min_max, rolling_window, use_roll_win, categorical_columns) for col in columns_to_transform)

            for col, transformed_data, transformer in results:
                if transformed_data is not None:
                    transformed_df[col] = transformed_data
                    save_transformer(transformer, col, method, transformer_path)
                else:
                    print(f"No data returned for column {col}")

            return transformed_df

        else:
            for col in columns_to_transform:
                if col in categorical_columns:
                    continue

                transformer = initialize_transformer(method, lower_bound_min_max)
                if method in ['log_transform', 'log_then_scale', 'log_then_standardize', 'difference_transform', 'standardize_then_scale','yeo_johnson_then_scale', 'yeo_johnson_then_standardize', 'square','square_then_log','cube_root', 'cube_root_then_scale', 'cube_root_then_standardize']:
                    transformed_series = apply_custom_transformation(transformed_df[[col]].values, method, lower_bound_min_max)
                else:
                    transformed_series = transformer.fit_transform(transformed_df[[col]])
                    save_transformer(transformer, col, method, transformer_path)
                transformed_df[col] = transformed_series.flatten()

            return transformed_df

    @staticmethod
    def load_transformers_and_apply(s3utility, data_df, method, output_path=transformer_path):
        transformed_df = data_df.copy(deep=True)
        for col in data_df.columns:
            file_path = f"{output_path}/{col}_{method}_transformer.joblib'"
            if s3utility.check_if_path_exists(file_path):
                transformer = s3utility.load_file(file_path)
                transformed_series = transformer.transform(data_df[[col]])
                transformed_df[col] = transformed_series.flatten()
            else:
                print(f"No transformer found for column {col}")
        return transformed_df
    
    @staticmethod
    def reverse_load_transformers_and_apply(data_df, method, output_path=transformer_path):
        transformed_df = data_df.copy(deep=True)
        for col in data_df.columns:
            file_path = os.path.join(output_path, f'{col}_{method}_transformer.joblib')
            if os.path.exists(file_path):
                transformer = joblib.load(file_path)
                transformed_series = transformer.inverse_transform(data_df[[col]])
                transformed_df[col] = transformed_series.flatten()
            else:
                print(f"No transformer found for column {col}")
        return transformed_df

    @staticmethod
    def apply_single_timestep(data_df, method, output_path=transformer_path):
        transformed_df = data_df.copy(deep=True)
        for col in data_df.columns:
            try:
                transformer = load_transformer(col, method, output_path)
                transformed_series = transformer.transform(data_df[[col]])
                transformed_df[col] = transformed_series.flatten()
            except FileNotFoundError as e:
                print(e)
        return transformed_df
    
    @staticmethod
    def reverse_apply_single_timestep(data_df, method, output_path=transformer_path):
        transformed_df = data_df.copy(deep=True)
        for col in data_df.columns:
            try:
                transformer = load_transformer(col, method, output_path)
                transformed_series = transformer.inverse_transform(data_df[[col]])
                transformed_df[col] = transformed_series.flatten()
            except FileNotFoundError as e:
                print(e)
        return transformed_df
    
    @staticmethod
    def inverse_transform_data(self, y_pred, output_path=transformer_path):
        """
        Inversely transform the predicted values to the original scale.

        Parameters:
        - y_pred: pandas.Series or numpy array, predicted values to inverse transform.
        - output_path: str, path to the directory where transformers are saved.

        Returns:
        - y_pred_original: numpy array, inversely transformed predicted values.
        """
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        # Ensure y_pred is a 2D array for the transformers
        y_pred = y_pred.reshape(-1, 1)

        # Apply inverse transformations in reverse order
        if self.transform_method is not None:
            if self.pred_type in ["return", "4h_return", "1d_return", "price"]:
                for method in reversed(self.transform_method):
                    y_pred = DataLoader.reverse_load_transformers_and_apply(data_df=pd.DataFrame(y_pred), method=method, output_path=output_path)
                
        # Handle log transformation if applicable
        if self.use_log is not None:
            if self.pred_type in ["return", "4h_return", "1d_return", "price"]:
                y_pred = np.sqrt(np.expm1(y_pred))


        return y_pred.flatten()
    

#################################################################################################################################################################
#
#
#                                                                  MAIN
#
#################################################################################################################################################################

    def main_loader(self, current_datetime=None, save_dataset=False, update_dataset=True):
        
        self.logger.info("Data Loader initializing dataset...")
        
        if self.frequency in ["1M","1w","1d"]:
            format = "%Y-%m-%d"
            if current_datetime is not None:
                current_datetime = utils.convert_date_format(current_datetime, format_in = "%Y-%m-%d %H:%M:%S", format_out="%Y-%m-%d")
        else:
            format = "%Y-%m-%d %H:%M:%S"
        
        dataset = pd.DataFrame()
                
        for coin in self.coins:
            self.logger.info("Loading dataset for coin: " + coin)
            if not self.real_time:
                
                if update_dataset:
                    
                    price_data, minute_data = self.kucoin_price_loader.download_data(coin, self.fiat, self.interval, start_date=self.start_date, end_date=self.end_date, 
                                                                use_local_timezone=True, drop_not_complete_timestamps=True, simulate_live_data=False, overwrite_file=True)
                    
                    self.TA_creator.calculate_indicator_data(price_df=price_data, indicators_to_calc=self.dataset_main_metrics, coin=coin, fiat=self.fiat, interval=self.interval, end_date=self.end_date)
                    
                    # self.download_all_metrics(metrics=self.dataset_main_metrics, instrument_pool=self.main_asset, start_date=self.start_date, end_date=self.end_date, 
                    #                     frequency=self.frequency, use_last_available_date=True, format=format, pooled_metrics=self.dataset_pooled_metrics)
                    
                #load dataset from config file
                sub_dataset = self.load_dataset_from_config(coin=coin,start=self.start_date, end=self.end_date)
                
            else:
                
                start_t_1 = utils.get_offset_datetime(current_datetime, offset=1, unit=self.frequency)
                start_t_2 = utils.get_offset_datetime(current_datetime, offset=2, unit=self.frequency)
                start_rolling = utils.get_offset_datetime(current_datetime, offset=self.transform_window, unit=self.frequency)
                
                # end = utils.get_offset_datetime(start, offset=120, unit=self.frequency)
                
                self.download_all_metrics(metrics=self.dataset_main_metrics, instrument_pool=self.main_asset, start_date=start_t_2, end_date=current_datetime, 
                                        frequency=self.frequency, use_last_available_date=True, format=format, pooled_metrics=self.dataset_pooled_metrics)



                sub_dataset = self.load_dataset_from_config(coin=coin,start=start_rolling, end=current_datetime)
            
            # Combine the main dataset and sub_dataset along columns
            dataset = pd.concat([dataset, sub_dataset], axis=1)
            dataset = dataset.sort_index()
        
         # Rename columns to valid Python identifiers
        self.logger.info("Data Loader dataset loaded. Beginning processing...")
        dataset.columns = dataset.columns.str.replace(r'[^a-zA-Z0-9_]', '_')
        
        #fill missing values
        self.logger.info("Treat missing values...")
        dataset = self.adjust_missing_values(dataset, method="drop", window_size=self.nan_window)
        dataset = self.adjust_missing_values(dataset, method=self.nan_method, window_size=self.nan_window)
        dataset = self.adjust_missing_values(dataset, method="bfill", window_size=self.nan_window)

        #encode label features
        self.logger.info("Encoding categorical features...")
        dataset, encoded_cols = self.encode_dataframe(dataset, method=self.encode_method)

        #winsorize
        self.logger.info("Winsorizing data...")
        if self.winsorize_limits is not None:
            dataset = self.winsorize(data_df=dataset, limits=self.winsorize_limits, adjust_abnormal_values=True)
        
        price_usdt_columns = dataset.columns[dataset.columns.str.contains("price_usdt_")]
        if not price_usdt_columns.empty:
            self.price_key = price_usdt_columns[0]
        elif "close" in dataset.columns:
            self.price_key = "close"
        elif "Close" in dataset.columns:
            self.price_key = "Close"
        else:
            for col in dataset.columns:
                if any(key in col for key in ["close", "Close", "High", "high", "Low", "low", "Open", "open"]) and "lower" not in col and "upper" not in col:
                    self.price_key = col
                    break
            
        #getting price before transformations
        price_usdt = dataset[self.price_key].copy(deep=True)
        
        #create variables from data for later
        self.logger.info("Creating predicting and price usdt variable...")
        y_pred_data = self.create_predicting_variable(dataset, method=self.pred_type, periods=1)
        
        #fill missing values again 
        self.logger.info("Filling missing values...")
        dataset = self.replace_inf_and_fill_zeros(dataset, replace_0=False)
        dataset = self.adjust_missing_values(dataset, method=self.nan_method, window_size=self.nan_window)
        
        #plot correlation matrix    
        self.plot_correlation_matrix(dataset)
    
        #remove multicollinearity
        if self.multicollinearity_method is not None:
            self.logger.info("Removing multicollinearity...")
            dataset, removed_vars = self.remove_multicollinearity(dataset.copy(deep=True), method=self.multicollinearity_method, threshold=self.multicollinearity_threshold)

        # transform all values to relative values
        if self.use_relative_change:
            self.logger.info("Transforming data to relative change...")
            dataset = self.replace_inf_and_fill_zeros(dataset.copy(deep=True), replace_0=False)
            columns_to_transform = [col for col in dataset.columns if "change" not in col]
            dataset = self.transform_data(data_df=dataset.copy(deep=True), method="difference_transform",columns_to_transform=columns_to_transform, categorical_columns=encoded_cols)
            dataset = self.replace_inf_and_fill_zeros(dataset.copy(deep=True), replace_0=False)
            dataset = self.adjust_missing_values(dataset.copy(deep=True), method="bfill", window_size=self.nan_window)
            dataset = self.winsorize(data_df=dataset, limits=0.001)
            dataset = dataset.clip(lower=-0.99999, upper=10)
            
        # transform all values to log values
        if self.use_log:
            self.logger.info("Log transforming data...")
            dataset = self.transform_data(data_df=dataset.copy(deep=True), method="square_then_log", categorical_columns=encoded_cols)
            
        #visualize distribution    
        # self.visualize_distribution(dataset, combined=True)
        
        dataset = self.replace_inf_and_fill_zeros(dataset.copy(deep=True), replace_0=False)
        dataset = self.adjust_missing_values(dataset.copy(deep=True), method="bfill", window_size=self.nan_window)
        
        #Applying PCA or SVD, or vae (variational autoencoder) for dimensionality reduction
        if self.clustering_method is not None:
            self.logger.info("Applying clustering...")
            dataset = self.clustering(data_df=dataset, variance_threshold=self.explained_variance_threshold, method=self.clustering_method, scale_method=self.scale_method_pca)
        
        if self.transform_method is not None and self.clustering_method is None:
            self.logger.info("Transforming data...")
            for method in self.transform_method:
                dataset = self.transform_data(data_df=dataset.copy(deep=True), method=method, rolling_window=self.transform_window, 
                                        columns_to_transform=self.columns_to_transform, use_roll_win=self.use_roll_window, categorical_columns=encoded_cols)        
        
        # Create a 1-day forward-shifted normalized price column
        if self.add_shifted_price:
            name_shifted = "price_shifted"
            price_shifted = dataset[self.price_key].copy(deep=True).shift(1).fillna(0)  
            dataset[name_shifted] = price_shifted      
            
        if self.add_raw_price:
            # Add back the stored raw price column
            raw_price_name = f"price_raw"
            dataset[raw_price_name] = price_usdt.copy(deep=True)
            
        # Add y_pred data
        dataset["y_pred"] = y_pred_data.copy(deep=True)
        
        #visualize distribution
        # self.visualize_distribution(dataset, combined=True)
        # self.visualize_distribution(dataset)
        
        dataset = self.replace_inf_and_fill_zeros(dataset.copy(deep=True), replace_0=False)
        # Drop all rows up until the first value in column y_pred is not 0
        first_non_zero_index = dataset[dataset["y_pred"] != 0].index[0]
        dataset = dataset.loc[first_non_zero_index:]
        dataset = self.adjust_missing_values(dataset, method="drop", window_size=0)
        
        self.dataset = dataset        
        self.logger.info("Data Loader processing complete.")
        
        #reorder for primary info
        dataset = self.reorder_dataset_columns(dataset)
        
        if save_dataset:
            self.logger.info("Saving dataset...")
            self.save_dataset(dataset)
            self.logger.info("Dataset saved.")
        
        return dataset

def main():
    loader = DataLoader(filename="dataset_btc_only_price_and_volume_1hreturn.xlsx")        #dataset_base_btc_direction_minmax_relative
    dataset = loader.main_loader(save_dataset=True)

# if __name__ == "__main__":
#     from multiprocessing import freeze_support
#     freeze_support()
#     main()


