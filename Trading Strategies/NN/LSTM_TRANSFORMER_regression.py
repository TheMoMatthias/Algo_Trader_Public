import gc
gc.collect()

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
from sklearn.model_selection import train_test_split, ParameterGrid
import platform
import psutil

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, LeakyReLU, Conv1D, MaxPooling1D, Flatten, Concatenate, RepeatVector, MultiHeadAttention, Add, LayerNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy, Loss, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.utils.class_weight import compute_class_weight
from keras.saving import register_keras_serializable
from sklearn.metrics import r2_score
from keras_tuner import HyperModel, RandomSearch, Hyperband

from loguru import logger
import psutil


# Reset TensorFlow session
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

test = tf.config.list_physical_devices('GPU')
print(test)

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

@register_keras_serializable(package="Custom", name="wasserstein_loss")
def wasserstein_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.reduce_mean(y_true * y_pred)

def custom_mse_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    loss = tf.reduce_mean(tf.square(y_pred - y_true))
    return loss

def gradient_penalty(real_data, generated_data, batch_size, critic):
    real_data = tf.reshape(real_data, generated_data.shape)

    alpha = tf.random.uniform(shape=[batch_size, 1], minval=0., maxval=1.)
    interpolated = alpha * real_data + (1 - alpha) * generated_data

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic(interpolated, training=True)

    grads = tape.gradient(pred, [interpolated])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    return gradient_penalty

# Path configuration
def convert_path_to_wsl(path):
    return path.replace('C:\\', '/mnt/c/').replace('\\', '/')

def convert_path_to_windows(path):
    return path.replace('/mnt/c/', 'C:\\').replace('/', '\\')

def convert_path(path, env):
    if env == 'wsl':
        return path.replace('C:\\', '/mnt/c/').replace('\\', '/')
    elif env == 'windows':
        return path.replace('/mnt/c/', 'C:\\').replace('/', '\\')
    else:
        return path

def get_running_environment():
    if 'microsoft-standard' in platform.uname().release:
        return 'wsl'
    elif platform.system() == 'Windows':
        return 'windows'
    else:
        return 'unknown'

def get_converted_path(path):
    return convert_path(path, env)

env = get_running_environment()

# crypto_bot_path = r"C:\Users\mauri\Documents\Trading Bot\Python\AlgoTrader"
crypto_bot_path_windows = "/mnt/c/Users/mauri/Documents/Trading Bot/Python/AlgoTrader"
crypto_bot_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

Python_path = os.path.dirname(crypto_bot_path_windows)
Python_path = os.path.dirname(crypto_bot_path)
Trading_bot_path = os.path.dirname(Python_path)
Trading_path = os.path.join(Trading_bot_path, "Trading")

data_path_crypto = os.path.join(Trading_bot_path, "Data", "Cryptocurrencies")
datasets_path = os.path.join(data_path_crypto, "Datasets")
csv_dataset_path = os.path.join(datasets_path, "crypto datasets", "csv")
hdf_dataset_path = os.path.join(datasets_path, "crypto datasets", "hdf5")
hist_data_download_path = os.path.join(crypto_bot_path, "Hist Data Download")
san_api_data_path = os.path.join(hist_data_download_path, "SanApi Data")
main_data_files_path = os.path.join(san_api_data_path, "Main data files")

strategy_path = os.path.join(crypto_bot_path, "Trading Strategies")
gan_path = os.path.join(strategy_path, "NN")
trade_api_path = os.path.join(crypto_bot_path, "API Trader")
backtest_path = os.path.join(crypto_bot_path, "Backtesting")
kucoin_api = os.path.join(crypto_bot_path, "Kucoin API")

utils_path = os.path.join(Python_path, "Tools")
logging_path = os.path.join(Trading_bot_path, "Logging")

if env == "windows":
    data_loader = os.path.join(crypto_bot_path, "Data Loader")
    config_path = os.path.join(crypto_bot_path, "Config")
else:
    data_loader = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"Data Loader")
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"Config")

paths_to_validate = [
    crypto_bot_path, Python_path, Trading_bot_path, Trading_path,
    data_path_crypto, datasets_path, csv_dataset_path, hdf_dataset_path,
    hist_data_download_path, san_api_data_path, main_data_files_path,
    strategy_path, gan_path, trade_api_path, backtest_path, kucoin_api,
    config_path, utils_path, logging_path, data_loader]

for path in paths_to_validate:
    sys.path.append(path)
    
import mo_utils as utils
from DataLoader import DataLoader
from StrategyEvaluator import StrategyEvaluator
from Base_NN_skeletons.lstm_autotransformer import build_sophisticated_lstm_transformer_model as nn_architecture

class LSTMAutoencoderHyperModel(HyperModel):
    def __init__(self, window_size, X_train_shape):
        self.window_size = window_size
        self.X_train_shape = X_train_shape

    def build(self, hp):
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.05)
        kernel_regularizer = hp.Float('kernel_regularizer', min_value=1e-6, max_value=1e-2, sampling='log')
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        
        input_shape = (self.window_size, self.X_train_shape[-1])
        model = nn_architecture(input_shape, 64, dropout_rate, kernel_regularizer)
        
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError())
        return model

class LSTMAutoencoderModelTrainer:
    def __init__(self, config_filename):
        self.logger = logger
        self.configure_logger()

        config_path = utils.find_config_path()
        self.config = utils.read_config_file(os.path.join(config_path, "strategy config", "NN", config_filename))

        self.config_filename = config_filename
        self.model = None

        self.coin = utils.get_config_value(self.config, "general", "coin")
        self.fiat = utils.get_config_value(self.config, "general", "currency")
        self.symbol = f"{self.coin}-{self.fiat}"
        self.slug = utils.get_config_value(self.config, "general", "slug")

        start_date_str = utils.get_config_value(self.config, "general", "start_date")
        end_date_str = utils.get_config_value(self.config, "general", "end_date")

        self.start_datetime = pd.to_datetime(start_date_str, format="%Y-%m-%d %H:%M:%S")
        self.end_datetime = pd.to_datetime(end_date_str, format="%Y-%m-%d %H:%M:%S")

        filename = utils.get_config_value(self.config, "general", "dataset_filename")
        dataloader = DataLoader(logger_input=self.logger, filename=filename)

        csv_name = os.path.splitext(filename)[0] + "_processed" + ".csv"
        hdf5_name = os.path.splitext(filename)[0] + "_processed" + ".h5"

        if not os.path.exists(os.path.join(csv_dataset_path, (csv_name))):
            self.data = dataloader.main_loader(save_dataset=True)
        else:
            self.logger.info(f"Loading dataset from {csv_name}")
            self.data = pd.read_csv(os.path.join(csv_dataset_path, (csv_name)), index_col=0)

        self.data.fillna(0, inplace=True)

        self.data.index = pd.to_datetime(self.data.index).strftime("%Y-%m-%d %H:%M:%S")
        self.data.index = pd.to_datetime(self.data.index, format="%Y-%m-%d %H:%M:%S")

        if self.data.index[1] > self.start_datetime:
            self.start_datetime = self.data.index[1]

        self.predicting_variable = self.data[self.data.columns[-1]].copy()
        self.predicting_variable.replace(np.nan,0, inplace=True)
        self.predicting_variable.replace(np.inf, 0, inplace=True)
        self.predicting_variable.replace(-np.inf, 0, inplace=True)   
        self.predicting_variable.replace({r'[^a-zA-Z0-9 ]+':''}, regex=True, inplace=True) 

        if self.data.columns[-2] == "price_raw":    
            self.price_usdt = self.data[self.data.columns[-2]]
            self.dataset = self.data[self.data.columns[:-2]].copy()
        else:
            self.dataset = self.data[self.data.columns[:-1]].copy()
            
        self.dataset = self.dataset.clip(lower=-1, upper=1)
        self.dataset.replace(np.nan, 0, inplace=True)
        self.dataset.replace(np.inf, 0, inplace=True)
        self.dataset.replace(-np.inf, 0, inplace=True)
        
        self.checkpoint_filepath = (os.path.join(gan_path, "model", "lstm_autoencoder_checkpoint.h5"))
        
        self.window_size = int(utils.get_config_value(self.config, "params", 'window_size'))
        self.epochs = int(utils.get_config_value(self.config, "params", "epochs"))
        self.batch_size = int(utils.get_config_value(self.config, "params", "batch_size"))
        self.learning_rate = float(utils.get_config_value(self.config, "params", "learning_rate"))
        self.lstm_units = int(utils.get_config_value(self.config, "params", "lstm_units"))
        self.dropout_rate = float(utils.get_config_value(self.config, "params", "dropout_rate"))
        self.kernel_regularizer = float(utils.get_config_value(self.config, "params", "kernel_regularizer"))
        self.early_stopping_patience = int(utils.get_config_value(self.config, "params", "early_stopping_patience"))
        self.seed = int(utils.get_config_value(self.config, "params", "seed"))
        self.val_size = float(utils.get_config_value(self.config, "params", "val_size"))
        self.test_size = float(utils.get_config_value(self.config, "params", "test_size"))
        
        self.use_grid_search = utils.get_config_value(self.config, "general", "use_grid_search")
        
        self.loss_fn = MeanSquaredError()

    def configure_logger(self):
        logger_path = utils.find_logging_path()
        current_datetime = datetime.now()
        timestamp = current_datetime.strftime('%d-%m-%Y_%H-%M')
        log_directory = "Model training logs"
        log_file_name = f"Model_training_log_{timestamp}.txt"
        log_file_path = os.path.join(logger_path, log_directory, log_file_name)

        if not os.path.exists(os.path.join(logger_path, log_directory)):
            os.makedirs(os.path.join(logger_path, log_directory))

        self.logger.add(log_file_path, rotation="500 MB", level="INFO")

    def create_sequences(self, data, labels, window_size):
        sequences = []
        sequence_labels = []

        for i in range(window_size, len(data)):
            sequences.append(data[i - window_size:i])
            sequence_labels.append(labels[i])

        sequences = np.array(sequences, dtype=np.float64)
        sequence_labels = np.array(sequence_labels, dtype=np.float64).reshape(-1, 1)

        assert not np.isnan(sequences).any(), "NaN values found in sequences."
        assert not np.isnan(sequence_labels).any(), "NaN values found in sequence labels."

        return sequences, sequence_labels

    def prepare_data(self, data, labels, window_size, batch_size):
        data_sequences, data_labels = self.create_sequences(data, labels, window_size)
        num_sequences = len(data_sequences)

        if num_sequences % batch_size != 0:
            num_sequences = (num_sequences // batch_size) * batch_size
            data_sequences = data_sequences[:num_sequences]
            data_labels = data_labels[:num_sequences]

        assert not np.isnan(data_sequences).any(), "NaN values found in data sequences."
        assert not np.isnan(data_labels).any(), "NaN values found in data labels."

        return data_sequences, data_labels

    def build_model(self):
        input_shape = (self.window_size, self.dataset.shape[-1])
        self.model = nn_architecture(input_shape, self.lstm_units, self.dropout_rate, self.kernel_regularizer)
        
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss_fn, metrics=[tf.keras.metrics.MeanAbsoluteError()])

    def train_model(self):
        X = self.dataset.values
        y = self.predicting_variable.values.astype(np.float64)
        y = y.flatten()

        test_split_idx = int(len(X) * (1 - self.test_size))
        val_split_idx = int(test_split_idx * (1 - self.val_size))

        X_train, X_val, X_test = X[:val_split_idx], X[val_split_idx:test_split_idx], X[test_split_idx:]
        y_train, y_val, y_test = y[:val_split_idx], y[val_split_idx:test_split_idx], y[test_split_idx:]

        self.X_train, self.y_train = self.prepare_data(X_train, y_train, self.window_size, self.batch_size)
        self.X_val, self.y_val = self.prepare_data(X_val, y_val, self.window_size, self.batch_size)
        self.X_test, self.y_test = self.prepare_data(X_test, y_test, self.window_size, self.batch_size)

        early_stopping = EarlyStopping(monitor='val_loss', patience=self.early_stopping_patience, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(self.checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min')

        if self.use_grid_search:
            self.logger.info("Performing Hyperparameter Tuning with Keras Tuner")

            hypermodel = LSTMAutoencoderHyperModel(self.window_size, self.X_train.shape)
            tuner = RandomSearch(
                hypermodel,
                objective='val_loss',
                max_trials=20,  # Adjust the number of trials as needed
                executions_per_trial=1,
                directory='keras_tuner_dir',
                project_name='lstm_autoencoder_tuning'
            )

            tuner.search_space_summary()

            tuner.search(self.X_train, self.y_train, epochs=self.epochs, validation_data=(self.X_val, self.y_val), batch_size=self.batch_size)

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            self.logger.info(f"Best hyperparameters: {best_hps.values}")

            best_params = {
                'learning_rate': best_hps.get('learning_rate'),
                'dropout_rate': best_hps.get('dropout_rate'),
                'kernel_regularizer': best_hps.get('kernel_regularizer')
            }

            self.build_model()
            self.model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss=self.loss_fn, metrics=[tf.keras.metrics.MeanAbsoluteError()])
            self.save_best_params(best_params)
        else:
            self.build_model()
            history = self.model.fit(self.X_train, self.y_train, 
                                    epochs=self.epochs, 
                                    batch_size=self.batch_size, 
                                    validation_data=(self.X_val, self.y_val), 
                                    callbacks=[early_stopping, model_checkpoint])
            return history

    def evaluate_model(self):
        self.model = load_model(self.checkpoint_filepath)
        loss, mae = self.model.evaluate(self.X_test, self.y_test)
        y_pred = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        self.logger.info(f"Test Loss: {loss}")
        self.logger.info(f"Test MAE: {mae}")
        self.logger.info(f"Test R2 Score: {r2}")
        return loss, mae, r2

    def predict(self, data):
        data_sequences, _ = self.create_sequences(data, np.zeros((data.shape[0],)), self.window_size)
        predictions = self.model.predict(data_sequences)
        return predictions

    def preprocess_real_time_data(self, real_time_data):
        processed_data = real_time_data
        processed_data = np.expand_dims(processed_data, axis=2)
        return processed_data

    def predict_real_time(self, real_time_data):
        real_time_data = self.preprocess_real_time_data(real_time_data)
        predictions = self.predict(real_time_data)
        return predictions

    def save_model(self):
        model_dir = os.path.join(gan_path, "model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_filename = os.path.join(model_dir, f'lstm_autoencoder_{self.model_identifier}.keras')
        self.model.save(model_filename)

        self.logger.info(f"Model saved to {model_dir} with identifier {self.model_identifier}")
        
        with open(os.path.join(model_dir, 'latest_model_id.txt'), 'w') as f:
            f.write(str(self.model_identifier))

    def save_best_params(self, best_params):
        params_dir = os.path.join(gan_path, "model")
        if not os.path.exists(params_dir):
            os.makedirs(params_dir)
        params_filename = os.path.join(params_dir, 'best_params.json')
        with open(params_filename, 'w') as f:
            json.dump(best_params, f)
        self.logger.info(f"Best parameters saved to {params_filename}")

    def load_model(self):
        model_dir = os.path.join(gan_path, "model")
        if not self.model_identifier:
            try:
                with open(os.path.join(model_dir, 'latest_model_id.txt'), 'r') as f:
                    self.model_identifier = f.read().strip()
            except FileNotFoundError:
                self.logger.error("No model identifier found. Please specify an identifier.")
                return

        model_path = os.path.join(model_dir, f'lstm_autoencoder_{self.model_identifier}.keras')
        self.model = tf.keras.models.load_model(model_path)

        self.logger.info(f"Model loaded from {model_dir} with identifier {self.model_identifier}")

    def main(self):
        self.train_model()
        self.evaluate_model()
        
        y_pred = self.predict_real_time(self.X_test)
        real_time_data = self.X_test[0]  # Example real-time data
        real_time_prediction = self.predict_real_time(real_time_data)
        print(f"Real-time prediction: {real_time_prediction}")


if __name__ == "__main__":
    config_filename = "config_LSTM_transformer.ini"
    model_trainer = LSTMAutoencoderModelTrainer(config_filename)
    model_trainer.main()