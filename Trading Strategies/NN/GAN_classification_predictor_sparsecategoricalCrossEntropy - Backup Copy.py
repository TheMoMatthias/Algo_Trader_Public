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

# Reset TensorFlow session
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

test = tf.config.list_physical_devices('GPU')
print(test)

from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, LeakyReLU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers.legacy import Adam
# from tf_keras.optimizers.legacy import Adam
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
from sklearn.model_selection import train_test_split, ParameterGrid
from loguru import logger


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
data_loader = os.path.join(crypto_bot_path, "Data Loader")

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
from Base_NN_skeletons import lstm_gan as nn_architecture

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
            self.data = pd.read_csv(os.path.join(csv_dataset_path, (csv_name)), index_col=0)

        self.data.fillna(0, inplace=True)

        self.data.index = pd.to_datetime(self.data.index).strftime("%Y-%m-%d %H:%M:%S")
        self.data.index = pd.to_datetime(self.data.index, format="%Y-%m-%d %H:%M:%S")

        if self.data.index[1] > self.start_datetime:
            self.start_datetime = self.data.index[1]

        self.predicting_variable = self.data[self.data.columns[-1]].copy()
        self.predicting_variable_adj = self.predicting_variable.replace({-1: 0, 0: 1, 1: 2})
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
        
        

        # Model structure params
        self.window_size = int(utils.get_config_value(self.config, "params", 'window_size'))
        self.epochs = int(utils.get_config_value(self.config, "params", "epochs"))
        self.batch_size = int(utils.get_config_value(self.config, "params", "batch_size"))
        # Initialize the loss function
        self.loss_fn = SparseCategoricalCrossentropy()

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

    # def build_generator(self, input_shape, lstm_units, dropout_rate, kernel_regularizer):
    #     initializer = HeNormal()
    #     inputs = Input(shape=input_shape)
    #     x = LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(inputs)
    #     x = Dropout(dropout_rate)(x)
    #     x = LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
    #     x = Dropout(dropout_rate)(x)
    #     x = LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(kernel_regularizer))(x)
    #     x = Dropout(dropout_rate)(x)
    #     x = Dense(input_shape[-1], activation='tanh', kernel_initializer=initializer)(x)
    #     generator = Model(inputs, x)
    #     return generator

    # def build_discriminator(self, input_shape, dropout_rate, kernel_regularizer):
    #     initializer = HeNormal()
    #     inputs = Input(shape=input_shape)
    #     x = Conv1D(64, kernel_size=3, activation='relu', kernel_initializer=initializer)(inputs)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Dropout(dropout_rate)(x)
    #     x = Conv1D(128, kernel_size=3, activation='relu', kernel_initializer=initializer)(x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Dropout(dropout_rate)(x)
    #     x = Flatten()(x)
    #     x = Dense(50, kernel_initializer=initializer, kernel_regularizer=l2(kernel_regularizer))(x)
    #     x = LeakyReLU(alpha=0.2)(x)
    #     x = Dense(3, activation='softmax', kernel_initializer=initializer)(x)  # Output three logits
    #     discriminator = Model(inputs, x)
    #     return discriminator

    def compile_gan(self, params):
        lstm_units = params['lstm_units']
        dropout_rate = params['dropout_rate']
        kernel_regularizer = params['kernel_regularizer']
        learning_rate = params['learning_rate']
        beta_1 = params['beta_1']

        input_shape = (self.window_size, self.X_train.shape[-1])
        self.generator = nn_architecture.build_generator(input_shape, dropout_rate, kernel_regularizer)
        self.discriminator = nn_architecture.build_discriminator(input_shape, dropout_rate, kernel_regularizer)

        print(self.generator.summary())
        print(self.discriminator.summary())

        # Re-initialize the optimizers
        self.generator_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, clipvalue=1.0)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, clipvalue=1.0)

        self.discriminator.compile(optimizer=self.discriminator_optimizer, loss=self.loss_fn, metrics=['accuracy'])

        self.discriminator.trainable = False
        gan_input = Input(shape=input_shape)
        generated = self.generator(gan_input)
        gan_output = self.discriminator(generated)

        self.gan = Model(gan_input, gan_output)
        self.gan.compile(optimizer=self.generator_optimizer, loss=self.loss_fn, metrics=['accuracy'])

        self.discriminator.trainable = True


    @tf.function
    def train_discriminator(self, real_data, real_labels):
        real_data = tf.cast(real_data, tf.float32)
        real_labels = tf.cast(real_labels, tf.int32)  # Ensure labels are integers

        # Debugging prints
        # tf.print("Real Data Shape:", tf.shape(real_data))
        # tf.print("Real Labels Shape:", tf.shape(real_labels))

        with tf.GradientTape() as tape:
            generated_data = self.generator(real_data, training=True)
            d_real_pred = self.discriminator(real_data, training=True)
            d_fake_pred = self.discriminator(generated_data, training=True)

            # Ensure real_labels are compatible with the shape (batch_size,)
            real_labels = tf.squeeze(real_labels, axis=-1)  # Remove the last dimension if needed

            # Create fake labels with the correct shape (batch_size,)
            fake_labels = tf.zeros(shape=(real_labels.shape[0],), dtype=tf.int32)

            d_real_loss = self.loss_fn(real_labels, d_real_pred)  # Labels should be integers
            d_fake_loss = self.loss_fn(fake_labels, d_fake_pred)
            d_loss = d_real_loss + d_fake_loss

        # NaN check and debugging prints
        # tf.print("d_real_loss:", d_real_loss)
        # tf.print("d_fake_loss:", d_fake_loss)
        # tf.print("d_loss:", d_loss)
        # tf.debugging.assert_all_finite(d_loss, 'd_loss contains NaN values')

        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        return d_loss


    @tf.function
    def train_generator(self, real_data, real_labels):
        real_data = tf.cast(real_data, tf.float32)
        real_labels = tf.cast(real_labels, tf.int32)  # Ensure labels are integers

        with tf.GradientTape() as tape:
            generated_data = self.generator(real_data, training=True)
            # Ensure real_labels are compatible with the shape (batch_size,)
            real_labels = tf.squeeze(real_labels, axis=-1)  # Remove the last dimension if needed

            # Use real_labels for training the generator (trying to trick the discriminator)
            g_loss = self.loss_fn(real_labels, self.discriminator(generated_data, training=True))

        tf.debugging.assert_all_finite(g_loss, 'g_loss contains NaN values')

        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        return g_loss

    def train_gan(self, epochs, batch_size):
        for epoch in range(epochs):
            for start_idx in range(0, self.X_train.shape[0], batch_size):
                end_idx = start_idx + batch_size
                real_data = self.X_train[start_idx:end_idx]
                real_labels = self.y_train[start_idx:end_idx]

                # Check for NaNs in real_data and real_labels
                assert not np.isnan(real_data).any(), 'real_data contains NaN values'
                assert not np.isnan(real_labels).any(), 'real_labels contains NaN values'
                
                g_loss = self.train_generator(real_data, real_labels)
                d_loss = self.train_discriminator(real_data, real_labels)
                
                # print(f"Epoch {epoch + 1}, Batch {start_idx // batch_size + 1}: D Loss = {d_loss}, G Loss = {g_loss}")

            if epoch % 10 == 0:
                val_loss, accuracy = self.evaluate(self.X_val, self.y_val)
                self.logger.info(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}, Validation Loss: {val_loss}, Accuracy: {accuracy}")

    def train(self):
        X = self.dataset.values
        y = self.predicting_variable_adj.values.astype(np.int32)
        y = y.flatten()
        test_split_idx = int(len(X) * (1 - self.test_size))
        val_split_idx = int(test_split_idx * (1 - self.val_size))

        X_train, X_val, X_test = X[:val_split_idx], X[val_split_idx:test_split_idx], X[test_split_idx:]
        y_train, y_val, y_test = y[:val_split_idx], y[val_split_idx:test_split_idx], y[test_split_idx:]

        # Convert numpy arrays to tuples to ensure set operations work correctly
        X_train_tuple = tuple(map(tuple, X_train))
        X_val_tuple = tuple(map(tuple, X_val))
        X_test_tuple = tuple(map(tuple, X_test))

        assert len(set(X_train_tuple).intersection(set(X_val_tuple))) == 0, "Train and validation sets overlap"
        assert len(set(X_train_tuple).intersection(set(X_test_tuple))) == 0, "Train and test sets overlap"
        assert len(set(X_val_tuple).intersection(set(X_test_tuple))) == 0, "Validation and test sets overlap"

        print(f"Original Shapes - X_train: {X_train.shape}, X_test: {X_test.shape}, X_val: {X_val.shape}")

        self.X_train, self.y_train = self.prepare_data(X_train, y_train, self.window_size, self.batch_size)
        self.X_val, self.y_val = self.prepare_data(X_val, y_val, self.window_size, self.batch_size)
        self.X_test, self.y_test = self.prepare_data(X_test, y_test, self.window_size, self.batch_size)
        self.y_train = self.y_train.astype(np.int32)
        self.y_val = self.y_val.astype(np.int32)
        self.y_test = self.y_test.astype(np.int32)
        
        print(f"Reshaped Shapes - X_train: {self.X_train.shape}, X_test: {self.X_test.shape}, X_val: {self.X_val.shape}")

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
            self.save_best_params(best_params)  # Save best parameters
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
        d_loss, accuracy = self.discriminator.evaluate(generated_data, y, verbose=0)
        
        return d_loss, accuracy

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
        generator_path = os.path.join(os.path.dirname(__file__), 'models', 'gan_generator.h5')
        self.generator = load_model(generator_path, custom_objects={'custom_loss': self.custom_loss})
        discriminator_path = os.path.join(os.path.dirname(__file__), 'models', 'gan_discriminator.h5')
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
        processed_data = self.preprocess_real_time_data(real_time_data)
        predictions = self.predict(processed_data)
        return predictions

    def preprocess_real_time_data(self, real_time_data):
        processed_data = real_time_data
        processed_data = np.expand_dims(processed_data, axis=2)
        return processed_data
    
    def convert_predictions_to_class(self, predictions):
        return np.argmax(predictions, axis=1)

    def main(self):
        self.train()
        variable_importances = self.get_variable_importance()
        print("Variable Importances:", variable_importances)

        # Example usage of conversion method
        predictions = self.predict(self.X_test)
        predicted_classes = self.convert_predictions_to_class(predictions)
        print("Predicted Classes:", predicted_classes)

if __name__ == "__main__":
    config_filename = "config_GAN_btc_PCA.ini"
    model_trainer = LSTMGANModelTrainer(config_filename)
    model_trainer.main()