import gc
# Collect garbage and release memory
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
import tensorflow as tf
import platform
from tensorflow.keras import backend as K
# import tf_keras

import psutil
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, LeakyReLU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tf_keras.optimizers.legacy import Adam
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy, Loss, MeanSquaredError
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.metrics import Precision, Recall, AUC
from loguru import logger

# Reset TensorFlow session
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

test = tf.config.list_physical_devices('GPU')
print(test)

# Ensure TensorFlow uses the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# Convert Windows paths to WSL paths
def convert_path_to_wsl(path):
    return path.replace('C:\\', '/mnt/c/').replace('\\', '/')

# Convert WSL paths to Windows paths
def convert_path_to_windows(path):
    return path.replace('/mnt/c/', 'C:\\').replace('/', '\\')

def convert_path(path, env):
    if env == 'wsl':
        # Convert Windows path to WSL path
        return path.replace('C:\\', '/mnt/c/').replace('\\', '/')
    elif env == 'windows':
        # Convert WSL path to Windows path
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

def check_df(df):
    if df.isnull().values.any():
        print("DataFrame contains NaN values.")
    if df.isin([np.inf, -np.inf]).values.any():
        print("DataFrame contains Inf or -Inf values.")

# Convert path based on environment
def get_converted_path(path):
    return convert_path(path, env)

# Detect environment
env = get_running_environment()

# crypto_bot_path = r"C:\Users\mauri\Documents\Trading Bot\Python\AlgoTrader"
crypto_bot_path = "/mnt/c/Users/mauri/Documents/Trading Bot/Python/AlgoTrader"
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
gan_path = os.path.join(strategy_path, "NN")
trade_api_path = os.path.join(crypto_bot_path, "API Trader")
backtest_path = os.path.join(crypto_bot_path, "Backtesting")
kucoin_api = os.path.join(crypto_bot_path, "Kucoin API")

# Config and Utility Paths
config_path = os.path.join(crypto_bot_path, "Config")
utils_path = os.path.join(Python_path, "Tools")
logging_path = os.path.join(Trading_bot_path, "Logging")

if env == "windows":
    data_loader = os.path.join(crypto_bot_path, "Data Loader")
else:
    data_loader = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"Data Loader")

# List of paths to validate
paths_to_validate = [
    crypto_bot_path, Python_path, Trading_bot_path, Trading_path,
    data_path_crypto, datasets_path, csv_dataset_path, hdf_dataset_path,
    hist_data_download_path, san_api_data_path, main_data_files_path,
    strategy_path, gan_path, trade_api_path, backtest_path, kucoin_api,
    config_path, utils_path, logging_path, data_loader
]

# # Validate paths
# for path in paths_to_validate:
#     if not os.path.exists(path):
#         print(f"Path does not exist: {path}")
#     else:
#         print(f"Path exists: {path}")

# Add valid paths to sys.path
for path in paths_to_validate:
    sys.path.append(path)

# Import custom modules
import mo_utils as utils
from DataLoader import DataLoader
from StrategyEvaluator import StrategyEvaluator
from Base_NN_skeletons import convolutional_lstm_gan_regression as nn_architecture

class LSTMGANModelTrainer:
    def __init__(self, config_filename):
        self.logger = logger
        self.configure_logger()

        # Config
        config_path = utils.find_config_path()
        self.config = utils.read_config_file(os.path.join(config_path, "strategy config", "NN", config_filename))

        self.config_filename = config_filename
        self.generator = None
        self.discriminator = None
        self.gan = None

        # Trading pair
        self.coin = utils.get_config_value(self.config, "general", "coin")
        self.fiat = utils.get_config_value(self.config, "general", "currency")
        self.symbol = f"{self.coin}-{self.fiat}"
        self.slug = utils.get_config_value(self.config, "general", "slug")

        # Date input
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
        self.predicting_variable_adj = self.predicting_variable 
        self.predicting_variable_adj.replace(np.nan,0, inplace=True)
        self.predicting_variable_adj.replace(np.inf, 0, inplace=True)
        self.predicting_variable_adj.replace(-np.inf, 0, inplace=True)   
        self.predicting_variable_adj.replace({r'[^a-zA-Z0-9 ]+':''}, regex=True, inplace=True) 

        self.price_usdt = self.data[self.data.columns[-2]].copy()
        self.dataset = self.data[self.data.columns[:-2]].copy()
        self.dataset = self.data[self.data.columns[:-2]].copy()
        self.dataset = self.dataset.clip(lower=-1, upper=1)
        self.dataset.replace(np.nan, 0, inplace=True)
        self.dataset.replace(np.inf, 0, inplace=True)
        self.dataset.replace(-np.inf, 0, inplace=True)
        
        self.checkpoint_filepath = (os.path.join(gan_path, "model", "gan_checkpoint.h5"))
        
        # Model structure params
        self.window_size = int(utils.get_config_value(self.config, "params", 'window_size'))
        self.epochs = int(utils.get_config_value(self.config, "params", "epochs"))
        self.batch_size = int(utils.get_config_value(self.config, "params", "batch_size"))
        # Initialize the loss function
        self.loss_fn = MeanSquaredError()
        self.early_stopping_patience = int(utils.get_config_value(self.config, "params", "early_stopping_patience"))

        # Model parameters
        self.lstm_units = int(utils.get_config_value(self.config, "params", "lstm_units"))
        self.dropout_rate = float(utils.get_config_value(self.config, "params", "dropout_rate"))
        self.kernel_regularizer = float(utils.get_config_value(self.config, "params", "kernel_regularizer"))
        self.learning_rate = float(utils.get_config_value(self.config, "params", "learning_rate"))
        self.beta_1 = float(utils.get_config_value(self.config, "params", "beta_1"))
        self.test_size = float(utils.get_config_value(self.config, "params", "test_size"))
        self.val_size = float(utils.get_config_value(self.config, "params", "val_size"))
        self.seed = int(utils.get_config_value(self.config, "params", "seed"))

        # Grid search parameters
        self.use_grid_search = utils.get_config_value(self.config, "general", "use_grid_search")
        self.grid_learning_rate = utils.get_config_value(self.config, "params", "grid_learning_rate")
        self.grid_beta_1 = utils.get_config_value(self.config, "params", "grid_beta_1")
        self.grid_lstm_units = utils.get_config_value(self.config, "params", "grid_lstm_units")
        self.grid_dropout_rate = utils.get_config_value(self.config, "params", "grid_dropout_rate")
        self.grid_kernel_regularizer = utils.get_config_value(self.config, "params", "grid_kernel_regularizer")

    def configure_logger(self):
        logger_path = utils.find_logging_path()
        current_datetime = datetime.now()
        timestamp = current_datetime.strftime('%d-%m-%Y_%H-%M')
        log_directory = "Algo Trader Backtest"
        log_file_name = f"Algo_trader_backtest_log_{timestamp}.txt"
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
    

    @tf.function
    def train_discriminator(self, real_data, real_labels):
        real_data = tf.cast(real_data, tf.float32)
        real_labels = tf.cast(real_labels, tf.float32)

        with tf.GradientTape() as tape:
            generated_data = self.generator(real_data, training=True)
            d_real_pred = self.discriminator(real_data, training=True)
            d_fake_pred = self.discriminator(generated_data, training=True)

            d_real_loss = self.loss_fn(real_labels, d_real_pred)
            d_fake_loss = self.loss_fn(tf.zeros_like(real_labels), d_fake_pred)
            d_loss = d_real_loss + d_fake_loss

        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        return d_loss

    @tf.function
    def train_generator(self, real_data, real_labels):
        real_data = tf.cast(real_data, tf.float32)
        real_labels = tf.cast(real_labels, tf.float32)

        with tf.GradientTape() as tape:
            generated_data = self.generator(real_data, training=True)
            g_loss = self.loss_fn(real_labels, self.discriminator(generated_data, training=True))

        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        return g_loss

    def compile_gan(self, params):
        dropout_rate = params['dropout_rate']
        kernel_regularizer = params['kernel_regularizer']
        learning_rate = params['learning_rate']
        beta_1 = params['beta_1']

        input_shape = (self.window_size, self.X_train.shape[-1])
        self.generator = nn_architecture.build_generator(input_shape, dropout_rate, kernel_regularizer)
        discriminator_input_shape = (self.window_size, 1)
        self.discriminator = nn_architecture.build_discriminator(discriminator_input_shape, dropout_rate, kernel_regularizer)

        print(self.generator.summary())
        print(self.discriminator.summary())

        self.generator_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, clipvalue=1.0)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, clipvalue=1.0)

        self.discriminator.compile(optimizer=self.discriminator_optimizer, loss=self.loss_fn, metrics=['mse'])

        self.discriminator.trainable = False
        gan_input = Input(shape=input_shape)
        generated = self.generator(gan_input)
        gan_output = self.discriminator(generated)

        self.gan = Model(gan_input, gan_output)
        self.gan.compile(optimizer=self.generator_optimizer, loss=self.loss_fn, metrics=['mse'])

        self.discriminator.trainable = True
    
    def train_gan(self, epochs, batch_size):
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.early_stopping_patience, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(self.checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min')

        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(epochs):
            d_losses = []
            g_losses = []

            for start_idx in range(0, self.X_train.shape[0], batch_size):
                end_idx = start_idx + batch_size
                real_data = self.X_train[start_idx:end_idx]
                real_labels = self.y_train[start_idx:end_idx]

                g_loss = self.train_generator(real_data, real_labels)
                d_loss = self.train_discriminator(real_data, real_labels)

                d_losses.append(d_loss)
                g_losses.append(g_loss)

            avg_d_loss = np.mean(d_losses)
            avg_g_loss = np.mean(g_losses)

            val_loss, mse = self.evaluate(self.X_val, self.y_val)

            self.logger.info(f"Epoch {epoch}, D Loss: {avg_d_loss}, G Loss: {avg_g_loss}, Validation Loss: {val_loss}, MSE: {mse}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                self.save_model()
            elif epoch - best_epoch >= self.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
    
    def train(self):
        
        # Add early stopping and model checkpoint parameters to the configuration
        self.early_stopping_patience = int(utils.get_config_value(self.config, "params", "early_stopping_patience"))
        
        X = self.dataset.values
        y = self.predicting_variable_adj.values.astype(np.int32)
        y = y.flatten()  # Ensure y is a 1D array of integers
        
        test_split_idx = int(len(X) * (1 - self.test_size))
        val_split_idx = int(test_split_idx * (1 - self.val_size))

        X_train, X_val, X_test = X[:val_split_idx], X[val_split_idx:test_split_idx], X[test_split_idx:]
        y_train, y_val, y_test = y[:val_split_idx], y[val_split_idx:test_split_idx], y[test_split_idx:]

         # Calculate class weights
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
        self.class_weights = {i: class_weights[i] for i in range(len(unique_classes))}
        
        # Ensure sets do not overlap
        X_train_tuple = tuple(map(tuple, X_train))
        X_val_tuple = tuple(map(tuple, X_val))
        X_test_tuple = tuple(map(tuple, X_test))

        assert len(set(X_train_tuple).intersection(set(X_val_tuple))) == 0, "Train and validation sets overlap"
        assert len(set(X_train_tuple).intersection(set(X_test_tuple))) == 0, "Train and test sets overlap"
        assert len(set(X_val_tuple).intersection(set(X_test_tuple))) == 0, "Validation and test sets overlap"

        self.X_train, self.y_train = self.prepare_data(X_train, y_train, self.window_size, self.batch_size)
        self.X_val, self.y_val = self.prepare_data(X_val, y_val, self.window_size, self.batch_size)
        self.X_test, self.y_test = self.prepare_data(X_test, y_test, self.window_size, self.batch_size)
        
        self.y_train = self.y_train.astype(np.int32).flatten()
        self.y_val = self.y_val.astype(np.int32).flatten()
        self.y_test = self.y_test.astype(np.int32).flatten()
        
        if self.use_grid_search:
            self.logger.info("Performing Grid Search")
            param_grid = {
                'learning_rate': self.grid_learning_rate,
                'beta_1': self.grid_beta_1,
                'lstm_units': self.grid_lstm_units,
                'dropout_rate': self.grid_dropout_rate,
                'kernel_regularizer': self.grid_kernel_regularizer,
            }
            best_params, best_val_loss = None, float('inf')
            for params in ParameterGrid(param_grid):
                self.compile_gan(params)
                self.train_gan(epochs=self.epochs, batch_size=self.batch_size)
                val_loss, accuracy = self.evaluate(self.X_val, self.y_val)
                if val_loss < best_val_loss:
                    best_params, best_val_loss = params, val_loss
                self.logger.info(f"Params: {params}, Validation Loss: {val_loss}")
            self.logger.info(f"Best Params: {best_params}, Best Validation Loss: {best_val_loss}")
            self.compile_gan(best_params)
            self.save_best_params(best_params)
        else:
            params = {
                'learning_rate': self.learning_rate,
                'beta_1': self.beta_1,
                'lstm_units': self.lstm_units,
                'dropout_rate': self.dropout_rate,
                'kernel_regularizer': self.kernel_regularizer,
            }
            self.compile_gan(params)
            self.train_gan(epochs=self.epochs, batch_size=self.batch_size)

        self.save_model()  # Save model

    def evaluate(self, X, y):
        generated_data = self.generator.predict(X)
        y_pred = self.discriminator.predict(generated_data)

        mse = tf.keras.losses.MeanSquaredError()
        d_loss = mse(y, y_pred)
        
        return tf.reduce_mean(d_loss).numpy(), mse(y, y_pred).numpy()
    
    def predict(self, data):
        return self.generator.predict(data)

    def save_model(self):
        model_dir = os.path.join(gan_path, "model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_filename = os.path.join(model_dir, 'gan_generator.h5')
        self.generator.save(model_filename)
        discriminator_filename = os.path.join(model_dir, 'gan_discriminator.h5')
        self.discriminator.save(discriminator_filename)
        self.logger.info(f"Models saved to {model_dir}")

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
        generator_path = os.path.join(model_dir, 'gan_generator.h5')
        self.generator = load_model(generator_path)
        discriminator_path = os.path.join(model_dir,'gan_discriminator.h5')
        self.discriminator = load_model(discriminator_path, custom_objects={'custom_loss': self.custom_loss})

    def get_variable_importance(self):
        input_tensors = [self.generator.input] + [K.learning_phase()]
        gradients = self.generator.optimizer.get_gradients(self.generator.total_loss, self.generator.input)
        compute_gradients = K.function(inputs=input_tensors, outputs=gradients)

        input_data = self.X_train[:100]
        outputs = compute_gradients([input_data, 0])
        gradients_values = np.mean(np.abs(outputs[0]), axis=0)

        variable_importance = np.mean(gradients_values, axis=0)
        return variable_importance

    def predict_real_time(self, real_time_data):
        predictions = self.generator.predict(real_time_data)
        return predictions
    
    def convert_predictions_to_class(self, predictions):
        return np.argmax(predictions, axis=1)

    def calculate_metrics(self, y_true, y_pred):
        y_pred = y_pred[:,-1,:]  # Flatten y_pred to match the shape of y_true
        y_true = y_true[self.window_size:]  # Ensure y_true matches the length of y_pred
        mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred).numpy()
        mse = tf.keras.metrics.mean_squared_error(y_true, y_pred).numpy()
        self.logger.info(f"Mean Absolute Error: {mae}")
        self.logger.info(f"Mean Squared Error: {mse}")
        return mae, mse

    def main(self):
        self.train()
        self.load_model() 
        
        y_pred = self.predict_real_time(self.X_test)
        # Example usage of conversion method
        # Calculate metrics on the test set
        mae, mse = self.calculate_metrics(self.y_test, y_pred)
        print(f"Test MAE: {mae}, Test MSE: {mse}")

        # Example usage of real-time prediction
        real_time_data = self.X_test[0]  # Example real-time data
        real_time_prediction = self.predict_real_time(real_time_data)
        print(f"Real-time prediction: {real_time_prediction}")

if __name__ == "__main__":
    config_filename = "config_GAN_regression.ini"
    model_trainer = LSTMGANModelTrainer(config_filename)
    model_trainer.main()