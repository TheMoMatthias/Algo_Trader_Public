import pandas as pd
import san 
import numpy as np 
import datetime as dt
import mo_utils as utils
import os
import time
from san import AsyncBatch
import json
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import seaborn 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from data_download_entire_history  import *
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sys

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
crypto_bot_path = os.path.dirname(base_path)
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



image_path = r"C:\Users\mauri\Documents\Trading Bot\Data\Cryptocurrencies\Data Analysis"

##########################################################################   Functions  #######################################################################
def calc_vif(X):
    
    #calculating VIF
    vif = pd.DataFrame()
    vif['variables'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return(vif)


##################################################################################################################################################################
## Create Dataset

# downloader_class = Data_Downloader_All()
# path = r"C:\Users\mauri\Documents\Trading Bot\Data\Cryptocurrencies\Datasets\Inputs\dataset_base_bitcoin.xlsx"
# btc_data = downloader_class.dataset_loader(dataset_config_path=path)

# btc_data = pd.read_hdf(r"C:\Users\mauri\Documents\Trading Bot\Data\Cryptocurrencies\Datasets\crypto datasets\hdf5\dataset_bitcoin_1d_20231128.h5", "dataset_bitcoin_1d_20231128", mode="r+")
btc_data = pd.read_csv(r"C:\Users\mauri\Documents\Trading Bot\Data\Cryptocurrencies\Datasets\crypto datasets\csv\dataset_bitcoin_1d_20231128.csv", index_col=0)


class DataAnalyser:
    def __init__(self, filename):
        self.data = pd.read_csv(os.path.join(csv_dataset_path, filename))
        self.image_path = r"C:\Users\mauri\Documents\Trading Bot\Data\Cryptocurrencies\Data Analysis"
    
    def calc_vif(self, X):
        # calculating VIF
        vif = pd.DataFrame()
        vif['variables'] = X.columns
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif
    
    def create_dataset(self):
        # downloader_class = Data_Downloader_All()
        # path = r"C:\Users\mauri\Documents\Trading Bot\Data\Cryptocurrencies\Datasets\Inputs\dataset_base_bitcoin.xlsx"
        # btc_data = downloader_class.dataset_loader(dataset_config_path=path)
        # btc_data = pd.read_hdf(r"C:\Users\mauri\Documents\Trading Bot\Data\Cryptocurrencies\Datasets\crypto datasets\hdf5\dataset_bitcoin_1d_20231128.h5", "dataset_bitcoin_1d_20231128", mode="r+")
        btc_data = pd.read_csv(self.filename, index_col=0)
        return btc_data
    
    def calculate_standard_correlation(self, btc_data):
        btc_data_corr = btc_data.corr(method='pearson')
        btc_data.applymap(lambda x: f'{x:.3f}')
        
        mask = np.zeros_like(btc_data)
        mask[np.triu_indices_from(mask)] = True
        
        fig = figure(figsize=(216, 216))
        fig = seaborn.heatmap(btc_data, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0, mask=mask, linewidths=0.2, square=False, xticklabels=True, yticklabels=True)
        plt.yticks(rotation=0, fontsize=6)
        plt.xticks(rotation=90, fontsize=6)
        plt.autoscale(enable=True, tight=True)
        
        image_path_corr = os.path.join(self.image_path, "correlation matrices")
        print("Saving factor correlation bitcoin.jpg")
        plt.savefig(os.path.join(image_path_corr, 'factor correlation bitcoin.jpg'), dpi=300)
        
        print("Saving factor correlation bitcoin.pdf")
        plt.savefig(os.path.join(image_path_corr, 'factor correlation bitcoin.pdf'), dpi=300)
        plt.show()
    
    def calculate_ranked_correlation(self, btc_data):
        btc_data_corr_ranked = btc_data.corr(method='spearman')
        btc_data_corr_ranked.applymap(lambda x: f'{x:.3f}')
        
        mask = np.zeros_like(btc_data_corr_ranked)
        mask[np.triu_indices_from(mask)] = True
        
        fig = figure(figsize=(300, 300))
        fig = seaborn.heatmap(btc_data_corr_ranked, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0, mask=mask, linewidths=0.2, square=False, xticklabels=True, yticklabels=True)
        plt.yticks(rotation=0, fontsize=10)
        plt.xticks(rotation=90, fontsize=10)
        plt.autoscale(enable=True, tight=True)
        
        image_path_corr = os.path.join(self.image_path, "correlation matrices")
        
        print("Saving factor correlation bitcoin ranked.jpg")
        plt.savefig(os.path.join(image_path_corr, 'factor correlation bitcoin ranked.jpg'), dpi=600)
        
        print("Saving factor correlation bitcoin ranked.pdf")
        plt.savefig(os.path.join(image_path_corr, 'factor correlation bitcoin ranked.pdf'), dpi=600)
        plt.show()
    
    def calculate_vif(self, btc_data_corr_ranked):
        # standard method
        vif = self.calc_vif(btc_data_corr_ranked).T
        vif_ranked = self.calc_vif(btc_data_corr_ranked).T
        print("please compare the data")
        
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
    
    def main():
        # Create an instance of DataAnalyser
        data_analyser = DataAnalyser("dataset_bitcoin_1d_20231128.csv")
        
        # Load the dataset
        btc_data = data_analyser.create_dataset()
        
        # Calculate standard correlation
        data_analyser.calculate_standard_correlation(btc_data)
        
        # Calculate ranked correlation
        data_analyser.calculate_ranked_correlation(btc_data)
        
        # Calculate VIF
        data_analyser.calculate_vif(btc_data)
        
        # Visualize distribution
        data_analyser.visualize_distribution(btc_data)
        
if __name__ == "__main__":
    analyser = DataAnalyser("dataset_btc_2019_2023_processed.csv")

        
    loader = analyser.main()
    
    