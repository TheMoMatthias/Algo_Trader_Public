
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
    crypto_bot_path_windows = "/mnt/c/Users/mauri/Documents/Trading Bot/Python/AlgoTrader"
else:
    crypto_bot_path = os.path.dirname(os.path.dirname(base_path))
    crypto_bot_path_windows = os.path.dirname(os.path.dirname(base_path))

Python_path = os.path.dirname(crypto_bot_path_windows)
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
from DataLoader import DataLoader
from StrategyEvaluator import StrategyEvaluator

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

import xgboost as xgb
from collections import Counter

def focal_loss(labels, preds, gamma=2.0):
    preds_prob = np.clip(preds, 1e-7, 1 - 1e-7)
    labels = np.eye(3)[labels.astype(int)]
    p_t = np.sum(labels * preds_prob, axis=1)
    loss = -np.sum(labels * ((1 - p_t) ** gamma) * np.log(p_t), axis=1)
    return loss


class XGBModelTrainer:
    def __init__(self, config_filename):
        self.logger = logger
        self.configure_logger()
        
        #config
        config_path = utils.find_config_path() 
        self.config = utils.read_config_file(os.path.join(config_path,"strategy config","XGB",config_filename))
        
        self.perform_classification = utils.get_config_value(self.config, "general", "perform_classification")
        self.perform_regression = utils.get_config_value(self.config, "general", "perform_regression")
        
        self.use_saved_model = utils.get_config_value(self.config, "general", "use_saved_model")
        self.model_filename = utils.get_config_value(self.config, "general", "model_filename")
        
        self.config_filename = config_filename

        self.model = None
        
        #trading pair
        self.coin = utils.get_config_value(self.config, "general", "coin")
        self.fiat = utils.get_config_value(self.config, "general", "currency")
        self.symbol = f"{self.coin}-{self.fiat}"
        
        #date input 
        start_date_str = utils.get_config_value(self.config, "general", "start_date")
        end_date_str = utils.get_config_value(self.config, "general", "end_date")

        self.start_datetime = pd.to_datetime(start_date_str, format="%Y-%m-%d %H:%M")
        self.end_datetime = pd.to_datetime(end_date_str, format="%Y-%m-%d %H:%M")
        
        filename = utils.get_config_value(self.config, "general", "dataset_filename")
        dataloader = DataLoader(logger_input=self.logger, filename=filename)
        
        csv_name = os.path.splitext(filename)[0] + "_processed" + ".csv"   
        hdf5_name = os.path.splitext(filename)[0] + "_processed" + ".h5"

        if not os.path.exists(os.path.join(csv_dataset_path, (csv_name))):
            self.data = dataloader.main_loader(save_dataset=True)
        else:
            self.data = pd.read_csv(os.path.join(csv_dataset_path, (csv_name)), index_col=0) #,sep=";")
        
        self.data.index = pd.to_datetime(self.data.index).strftime("%Y-%m-%d %H:%M:%S")
        self.data.index = pd.to_datetime(self.data.index, format="%Y-%m-%d %H:%M:%S")

        if self.data.index[1]> self.start_datetime:
            self.start_datetime = self.data.index[1]
            
        self.predicting_variable = self.data[self.data.columns[-1]].copy()
        
        if self.perform_classification:
            self.predicting_variable = self.predicting_variable.replace({-1: 0, 0: 1, 1: 2})
    
        if self.data.columns[-2] == "price_raw":    
            self.price_usdt = self.data[self.data.columns[-2]]
            self.dataset = self.data[self.data.columns[:-2]].copy()
        else:
            self.dataset = self.data[self.data.columns[:-1]].copy()

        self.use_grid_search = utils.get_config_value(self.config, "params", "use_grid_search")
        self.eval_metric = utils.get_config_value(self.config, "params", "eval_metric")
        self.objective = utils.get_config_value(self.config, "params", "objective")
        self.num_class = utils.get_config_value(self.config, "params", "num_class") if self.perform_classification else None
        self.use_class_weights = utils.get_config_value(self.config, "params", "use_class_weights")
#################################################################################################################################################################
#
#                                                                  Config
#
#################################################################################################################################################################
    
    def configure_logger(self):
        
        logger_path = utils.find_logging_path()

        #logger
        current_datetime = dt.datetime.now()
        timestamp = current_datetime.strftime('%d-%m-%Y_%H-%M')
        log_directory = "Model training logs"
        log_file_name = f"XGB_training_log_{timestamp}.txt"
        log_file_path = os.path.join(logger_path, log_directory, log_file_name)

        if not os.path.exists(os.path.join(logger_path, log_directory)):
            os.makedirs(os.path.join(logger_path, log_directory))

        self.logger.add(log_file_path, rotation="500 MB", level="INFO")


    def create_param_grid_from_config(self):
        param_grid = {}
        config_params = self.config["param_grid"]

        for param, values in config_params.items():
            values = eval(values)  # Convert string to list or values
            if isinstance(values[0], float):  # Continuous values
                if param == "learning_rate" or param in ["gamma", "subsample", "colsample_bytree"]:
                    param_grid[param] = Real(min(values), max(values), prior="log-uniform" if param == "learning_rate" else 'uniform')
                else:
                    param_grid[param] = Real(min(values), max(values))
            elif isinstance(values[0], int):  # Discrete integer values
                param_grid[param] = Integer(min(values), max(values))
            elif isinstance(values[0], str):  # Categorical values
                param_grid[param] = Categorical(values)

        return param_grid

#################################################################################################################################################################
#
#                                                                  Model Training
#
#################################################################################################################################################################
    def load_model(self):
        """Load an existing XGBoost model."""
        if not self.model_filename:
            self.logger.error("No model filename provided for loading!")
            return

        # Ensure model path exists
        model_path = os.path.join(os.path.dirname(__file__), 'models', self.model_filename)
        
        # Check if the model file exists before loading
        if not os.path.exists(model_path):
            self.logger.error(f"Model file {model_path} does not exist!")
            return

        # Load the model using the XGBoost Booster class
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.logger.info(f"Loaded model from {model_path}")

    def save_model(self):
        """Save the trained model."""
        if not self.model:
            self.logger.error("No model found to save!")
            return

        # Ensure the models directory exists
        directory = os.path.join(os.path.dirname(__file__), 'models')
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Use model_filename from the config file to save the model
        model_path = os.path.join(directory, self.model_filename)
        
        # Save the model using the XGBoost Booster's save_model function
        self.model.save_model(model_path)
        self.logger.info(f"Model saved to {model_path}")

    def save_best_params(self):
        """Save the best params found by GridSearch."""
        directory = os.path.join(os.path.dirname(__file__), 'params')
        if not os.path.exists(directory):
            os.makedirs(directory)

        params_filename = os.path.join(directory, 'best_params.json')
        with open(params_filename, 'w') as f:
            json.dump(self.best_params, f)
        self.logger.info(f"Best parameters saved to {params_filename}")
    
    def create_split_datasets(self):
        """Create train, validation, and test datasets."""
        train_split_length = utils.get_config_value(self.config, "params", "train_split_length")
        val_split_length = 1- utils.get_config_value(self.config, "params", "val_split_length")
        
        X, y = self.dataset, self.predicting_variable
        train_split_length, val_split_length = int(train_split_length * len(X)), int(val_split_length * len(X))
        self.X_train, self.X_val, self.X_test = X[:train_split_length], X[train_split_length:val_split_length], X[val_split_length:]
        self.y_train, self.y_val, self.y_test = y[:train_split_length], y[train_split_length:val_split_length], y[val_split_length:]
    
    
    def train(self, use_class_weights=False, use_grid_search=False):
        """Train the model."""
        
        self.create_split_datasets()

        sample_weights = np.ones_like(self.y_train)
        if self.perform_classification and use_class_weights:
            class_weights = self.compute_class_weights(self.y_train)
            sample_weights = np.array([class_weights[label] for label in self.y_train])

        if use_grid_search:
            self.grid_search()
        else:
            self.standard_train(sample_weights)

    # def load_param_grid(self):
    #     """Dynamically load the param_grid dictionary from the config file."""
    #     param_grid = {}
    #     if 'param_grid' in self.config:
    #         for key in self.config['param_grid']:
    #             # Split string values to list (for params like learning_rate, max_depth)
    #             param_grid[key] = eval(self.config['param_grid'][key])
    #     return param_grid

    def grid_search(self):
        """Perform BayesSearchCV for optimal parameters."""
        param_grid = self.create_param_grid_from_config()

        # Use BayesSearchCV instead of GridSearchCV
        if self.perform_classification:
            base_model = xgb.XGBClassifier(objective=self.objective,
                                        num_class=self.num_class,
                                        tree_method="hist", device="cuda")
        elif self.perform_regression:
            base_model = xgb.XGBRegressor(objective=self.objective,
                                        tree_method="hist", device="cuda")

        # BayesSearchCV replaces GridSearchCV, it also takes the same parameters
        bayes_search = BayesSearchCV(
            base_model,
            param_grid,
            n_iter=50,  # Set the number of parameter settings to sample
            scoring='accuracy' if self.perform_classification else 'neg_mean_squared_error',
            cv=20,
            verbose=3
        )
        
        bayes_search.fit(self.X_train, self.y_train)

        self.best_params = bayes_search.best_params_
        self.logger.info(f"Best Params Found: {self.best_params}")

        # Saving the best params
        self.save_best_params()

        # Train the model with best parameters
        self.params = {**self.best_params, "tree_method": "hist", "device": "cuda"}
        self.standard_train(sample_weights=None, params=self.params)  # Train using the best params

   

    def standard_train(self, sample_weights, params = None):
        """Standard training without grid search."""
        learning_rate = utils.get_config_value(self.config, "params", "learning_rate")
        max_depth = utils.get_config_value(self.config, "params", "max_depth")
        n_estimators = utils.get_config_value(self.config, "params", "n_estimators")
        gamma = utils.get_config_value(self.config, "params", "gamma")
        colsample_bytree = utils.get_config_value(self.config, "params", "colsample_bytree")
        subsample = utils.get_config_value(self.config, "params", "subsample")
        
        if params is None:
            self.params = {
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "n_estimators": n_estimators,
                "gamma": gamma,
                "colsample_bytree": colsample_bytree,
                "subsample": subsample,
                "tree_method": "hist",
                "device": "cuda",
                "objective": self.objective,
                "eval_metric": self.eval_metric
            }
        else:
            self.params = params

        if self.perform_classification:
            self.params["num_class"] = self.num_class

        dtrain = xgb.DMatrix(self.X_train, label=self.y_train, weight=sample_weights)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)

        evals = [(dtrain, 'train'), (dval, 'eval')]
        evals_result = {}

        if self.use_saved_model and not self.use_grid_search:
            self.load_model()
        else:
            self.model = xgb.train(self.params, dtrain, num_boost_round=self.params['n_estimators'], evals=evals, evals_result=evals_result, verbose_eval=True, early_stopping_rounds=2500)

        self.log_training_metrics(evals_result, dval)

        self.save_model()

    def log_training_metrics(self, evals_result, dval):
        """Log specific training metrics depending on task."""
        if self.perform_classification:
            # Log classification metrics
            train_metric = evals_result['train'][self.params['eval_metric']][-1]
            val_metric = evals_result['eval'][self.params['eval_metric']][-1]
            self.logger.info(f"Final Train {self.params['eval_metric']}: {train_metric}")
            self.logger.info(f"Final Validation {self.params['eval_metric']}: {val_metric}")
        elif self.perform_regression:
            # For regression, log RMSE
            train_rmse = np.sqrt(mean_squared_error(self.y_train, self.predict(self.X_train)))
            val_rmse = np.sqrt(mean_squared_error(self.y_val, self.predict(self.X_val)))
            self.logger.info(f"Final Train RMSE: {train_rmse}")
            self.logger.info(f"Final Validation RMSE: {val_rmse}")



    def predict(self, X):
        """Predict using the trained model or a loaded model."""
        dmatrix = xgb.DMatrix(X)
        predictions = self.model.predict(dmatrix)
        return predictions

    def evaluate(self):
        """Evaluate model performance on the test set."""
        if self.perform_classification:
            accuracy = self.evaluate_classification()
            self.logger.info(f"Test Accuracy: {accuracy}")
        elif self.perform_regression:
            self.evaluate_regression()

    def evaluate_classification(self):
        """Evaluate classification metrics."""
        preds = self.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, preds)
        return accuracy

    def evaluate_regression(self):
        """Evaluate regression metrics."""
        preds = self.predict(self.X_test)
        preds = pd.DataFrame(preds, index=self.y_test.index, columns=["y_pred"])
        df = pd.concat([self.y_test, preds], axis=1)
        rmse = np.sqrt(mean_squared_error(self.y_test, preds))
        r2 = r2_score(self.y_test, preds)
        self.logger.info(f"Test RMSE: {rmse}, Test R^2: {r2}")

    def compute_class_weights(self, y):
        counter = Counter(y)
        total_samples = len(y)
        class_weights = {cls: total_samples / count for cls, count in counter.items()}
        return class_weights

    def main(self):
        self.train(use_class_weights=self.use_class_weights, use_grid_search=self.use_grid_search)
        
        #evaluate model
        self.create_split_datasets()
        self.load_model()
        self.evaluate()
        self.logger.info("Model training completed.")
        
if __name__ == "__main__":
    config_filename = "config_XGB_BTC_1h_indicator_raw.ini"
    model_trainer = XGBModelTrainer(config_filename)
    model_trainer.main()