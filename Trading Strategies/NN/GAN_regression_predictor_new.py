import gc
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
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, LeakyReLU, Conv1D, MaxPooling1D, Flatten, Concatenate, RepeatVector
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy, Loss,MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.utils.class_weight import compute_class_weight
from keras.saving import register_keras_serializable
from keras_tuner import HyperModel, RandomSearch, Hyperband

from loguru import logger
import psutil


# Reset TensorFlow session
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

test = tf.config.list_physical_devices('GPU')
print(test)

###########################    GPU CONFIGURATION    ############################

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

###########################    CUSTOM TRAIN FUNCTIONS  ############################

@register_keras_serializable(package="Custom", name="wasserstein_loss")
def wasserstein_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.reduce_mean(y_true * y_pred)

def custom_mse_loss(y_true, y_pred):
    """
    Custom mean squared error loss function for regression.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    loss = tf.reduce_mean(tf.square(y_pred - y_true))
    return loss

def gradient_penalty(real_data, generated_data, batch_size, critic):
    # real_data = tf.reshape(real_data, generated_data.shape)

    alpha = tf.random.uniform(shape=[batch_size, 1], minval=0., maxval=1.)
    interpolated = alpha * real_data + (1 - alpha) * generated_data

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic(interpolated, training=True)

    grads = tape.gradient(pred, [interpolated])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    return gradient_penalty



###########################    PATH CONFIGURATION    ############################

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

utils_path = os.path.join(Python_path, "Tools")
logging_path = os.path.join(Trading_bot_path, "Logging")

if env == "windows":
    data_loader = os.path.join(crypto_bot_path, "Data Loader")
    config_path = os.path.join(crypto_bot_path, "Config")
else:
    data_loader = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"Data Loader")
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"Config")

# List of paths to validate
paths_to_validate = [
    crypto_bot_path, Python_path, Trading_bot_path, Trading_path,
    data_path_crypto, datasets_path, csv_dataset_path, hdf_dataset_path,
    hist_data_download_path, san_api_data_path, main_data_files_path,
    strategy_path, gan_path, trade_api_path, backtest_path, kucoin_api,
    config_path, utils_path, logging_path, data_loader
]

# Add valid paths to sys.path
for path in paths_to_validate:
    sys.path.append(path)
    
# Import custom modules
import mo_utils as utils
from DataLoader import DataLoader
from StrategyEvaluator import StrategyEvaluator
from Base_NN_skeletons import convolutional_lstm_gan_regression_with_noise as nn_architecture

#################################################################################################################################################################
#
#
#                                                                  KERAS TUNER CLASS
#
#################################################################################################################################################################


class MyHyperModel(HyperModel):
    def __init__(self, window_size, noise_dim, X_train_shape):
        self.window_size = window_size
        self.noise_dim = noise_dim
        self.X_train_shape = X_train_shape

    def build(self, hp):
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.05)
        kernel_regularizer = hp.Float('kernel_regularizer', min_value=1e-6, max_value=1e-2, sampling='log')
        learning_rate_gen = hp.Float('learning_rate_gen', min_value=1e-5, max_value=1e-2, sampling='log')
        learning_rate_disc = hp.Float('learning_rate_disc', min_value=1e-5, max_value=1e-2, sampling='log')
        beta_1 = hp.Float('beta_1', min_value=0.5, max_value=0.99)

        input_shape = (self.window_size, self.X_train_shape[-1])
        critic_input_shape = (self.window_size, 1)  # Adjust as needed

        generator = nn_architecture.build_generator(input_shape, self.noise_dim, dropout_rate, kernel_regularizer)
        critic = nn_architecture.build_discriminator(critic_input_shape, dropout_rate, kernel_regularizer)

        lr_gen_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate_gen, decay_steps=100000, decay_rate=0.96, staircase=True)
        lr_critic_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate_disc, decay_steps=100000, decay_rate=0.96, staircase=True)

        generator_optimizer = Adam(learning_rate=lr_gen_schedule, beta_1=beta_1, clipvalue=1.0)
        critic_optimizer = Adam(learning_rate=lr_critic_schedule, beta_1=beta_1, clipvalue=1.0)

        critic.compile(optimizer=critic_optimizer, loss=wasserstein_loss)

        critic.trainable = False
        real_input = Input(shape=input_shape)
        noise_input = Input(shape=(self.noise_dim,))
        generated = generator([real_input, noise_input])
        gan_output = critic(generated)

        gan = Model([real_input, noise_input], gan_output)
        gan.compile(optimizer=generator_optimizer, loss=wasserstein_loss)

        critic.trainable = True

        return gan

#################################################################################################################################################################
#
#
#                                                                  INIT CLASS
#
#################################################################################################################################################################


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
        self.predicting_variable.replace(np.nan,0, inplace=True)
        self.predicting_variable.replace(np.inf, 0, inplace=True)
        self.predicting_variable.replace(-np.inf, 0, inplace=True)   
        self.predicting_variable.replace({r'[^a-zA-Z0-9 ]+':''}, regex=True, inplace=True) 

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
        self.use_balanced_features = int(utils.get_config_value(self.config, "general", "use_balanced_features"))
        self.generator_updates = int(utils.get_config_value(self.config, "general", "generator_updates")) # New line for generator_updates
        self.discriminator_updates = int(utils.get_config_value(self.config, "general", "discriminator_updates"))
        self.noise_dim = int(utils.get_config_value(self.config, "params", "noise_dim"))
        self.model_identifier = utils.get_config_value(self.config, "general", "model_identifier")
        self.train_existing_model = utils.get_config_value(self.config, "general", "train_existing_model")
        self.use_balanced_training = utils.get_config_value(self.config, "general", "use_balanced_training")
        
        # Initialize the loss function
        self.loss_fn = MeanSquaredError()
        self.early_stopping_patience = int(utils.get_config_value(self.config, "params", "early_stopping_patience"))
        
        # Model parameters
        self.lstm_units = int(utils.get_config_value(self.config, "params", "lstm_units"))
        self.dropout_rate = float(utils.get_config_value(self.config, "params", "dropout_rate"))
        self.kernel_regularizer = float(utils.get_config_value(self.config, "params", "kernel_regularizer"))
        self.learning_rate_gen = float(utils.get_config_value(self.config, "params", "learning_rate_gen"))
        self.learning_rate_disc = float(utils.get_config_value(self.config, "params", "learning_rate_disc"))
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

#################################################################################################################################################################
#
#
#                                                                  Logger
#
#################################################################################################################################################################

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

#################################################################################################################################################################
#
#
#                                                                  DATA PREPARATION
#
#################################################################################################################################################################

    def create_sequences(self, data, labels, window_size):
        sequences = []
        sequence_labels = []
        sequence_batched_labels = []
        sequence_batched_labels_oh = []
        
        for i in range(window_size, len(data)):
            sequences.append(data[i - window_size:i])
            sequence_labels.append(labels[i])
            sequence_batched_labels.append(labels[i - window_size+1:i+1])
            
            if "convolutional" in str(nn_architecture):
                # Convert label to one-hot encoded vector
                one_hot_label = tf.one_hot(labels[i - window_size+1:i+1], 1).numpy()  # Convert to NumPy array
            else:
                # Use this function for running non convolutional networks as LSTM does not require 3d shape as input
                one_hot_label = tf.one_hot(labels[i], 1).numpy()
            
            sequence_batched_labels_oh.append(one_hot_label)
            
        sequences = np.array(sequences, dtype=np.float64)
        sequence_labels = np.array(sequence_labels, dtype=np.float64).reshape(-1, 1)
        sequence_batched_labels = np.array(sequence_batched_labels, dtype=np.float64)
        sequence_batched_labels_oh = np.array(sequence_batched_labels_oh, dtype=np.float64)
        
        assert not np.isnan(sequences).any(), "NaN values found in sequences."
        assert not np.isnan(sequence_labels).any(), "NaN values found in sequence labels."
        assert not np.isnan(sequence_batched_labels).any(), "NaN values found in sequence batched labels."

        return sequences, sequence_labels, sequence_batched_labels, sequence_batched_labels_oh

    def prepare_data(self, data, labels, window_size, batch_size):
        data_sequences, data_labels, sequence_batched_labels, sequence_batched_labels_oh = self.create_sequences(data, labels, window_size)
        num_sequences = len(data_sequences)

        if num_sequences % batch_size != 0:
            num_sequences = (num_sequences // batch_size) * batch_size
            data_sequences = data_sequences[:num_sequences]
            data_labels = data_labels[:num_sequences]
            sequence_batched_labels = sequence_batched_labels[:num_sequences]
            sequence_batched_labels_oh[:num_sequences]
            
        assert not np.isnan(data_sequences).any(), "NaN values found in data sequences."
        assert not np.isnan(data_labels).any(), "NaN values found in data labels."
        assert not np.isnan(sequence_batched_labels).any(), "NaN values found in sequence batched labels."
        
        return data_sequences, data_labels, sequence_batched_labels, sequence_batched_labels_oh

#################################################################################################################################################################
#
#
#                                                                  MODEL FUNCTIONS
#
#################################################################################################################################################################


    def build_models(self, input_shape=None, critic_input_shape=None, dropout_rate=None, kernel_regularizer=None):
        if self.train_existing_model and os.path.exists(os.path.join(gan_path, "model")):
            self.load_model()
        else:
            self.generator = nn_architecture.build_generator(input_shape, self.noise_dim, dropout_rate, kernel_regularizer)
            self.critic = nn_architecture.build_discriminator(critic_input_shape, dropout_rate, kernel_regularizer)
            self.save_model()  # Save the initial state of the models

    def compile_gan(self, params):
        dropout_rate = params['dropout_rate']
        kernel_regularizer = params['kernel_regularizer']
        learning_rate_gen = params["learning_rate_gen"]
        learning_rate_disc = params["learning_rate_disc"]
        beta_1 = params['beta_1']

        
        input_shape = (self.window_size, self.X_train.shape[-1])
        critic_input_shape = (self.window_size, 1)  # The generator's output shape
        self.build_models(input_shape, critic_input_shape, dropout_rate, kernel_regularizer)
        
        print(self.generator.summary())
        print(self.critic.summary())

        # Exponential decay for learning rates
        lr_gen_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate_gen, decay_steps=100000, decay_rate=0.96, staircase=True)
        lr_critic_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate_disc, decay_steps=100000, decay_rate=0.96, staircase=True)

        self.generator_optimizer = Adam(learning_rate=lr_gen_schedule, beta_1=beta_1, clipvalue=1.0)
        self.critic_optimizer = Adam(learning_rate=lr_critic_schedule, beta_1=beta_1, clipvalue=1.0)

        self.critic.compile(optimizer=self.critic_optimizer, loss=wasserstein_loss)

        self.critic.trainable = False
        real_input = Input(shape=input_shape)
        noise_input = Input(shape=(self.noise_dim,))
        generated = self.generator([real_input, noise_input])
        gan_output = self.critic(generated)

        self.gan = Model([real_input, noise_input], gan_output)
        self.gan.compile(optimizer=self.generator_optimizer, loss=wasserstein_loss)

        self.critic.trainable = True

#################################################################################################################################################################
#
#
#                                                                  TRAIN FUNCTIONS
#
#################################################################################################################################################################  
    
    @tf.function
    def train_generator(self, real_data, real_labels):
        real_data = tf.cast(real_data, tf.float32)
        real_labels = tf.cast(real_labels, tf.float32)
        batch_size = real_data.shape[0]
        noise = tf.random.uniform((batch_size, self.noise_dim))

        with tf.GradientTape() as tape:
            generated_data = self.generator([real_data, noise], training=True)
            d_fake = self.critic(generated_data, training=True)

            # Adversarial loss
            g_loss_adv = -tf.reduce_mean(d_fake)
            
            # Regression loss
            real_labels = tf.reshape(real_labels, [batch_size, 1])
            generated_data = tf.reshape(generated_data, [batch_size, 1])
            reg_loss = custom_mse_loss(real_labels, generated_data)
            
            # Total generator loss
            g_loss = g_loss_adv + reg_loss

        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        return g_loss, g_loss_adv, reg_loss
    
    
    @tf.function
    def train_critic(self, real_data, real_labels):
        batch_size = real_data.shape[0]
        real_data = tf.cast(real_data, tf.float32)
        real_labels = tf.cast(real_labels, tf.float32)
        noise = tf.random.uniform((batch_size, self.noise_dim))

        with tf.GradientTape() as tape:
            generated_data = self.generator([real_data, noise], training=True)

            # Ensure real_labels and generated_data have the correct shapes
            real_labels = tf.reshape(real_labels, [batch_size, 1])
            generated_data = tf.reshape(generated_data, [batch_size, 1])

            d_real = self.critic(real_labels, training=True)
            d_fake = self.critic(generated_data, training=True)

            # Compute critic loss
            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
            gp = gradient_penalty(real_labels, generated_data, batch_size, self.critic)
            d_loss += 10 * gp
            
        grads = tape.gradient(d_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        return d_loss
    
    def train_gan(self, epochs, batch_size):
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.early_stopping_patience, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(self.checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min')

        best_val_loss = float('inf')
        best_epoch = 0
        leading_network = "critic"
        last_d_loss = None
        last_g_loss = None  
        
        self.logger.info("Training GAN")
        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch}/{epochs}")
            d_losses = []
            g_losses = []
            g_losses_adv =  []
            reg_losses = []

            for start_idx in range(0, self.X_train.shape[0], batch_size):
                end_idx = start_idx + batch_size
                real_data = self.X_train[start_idx:end_idx]
                real_labels = self.y_train[start_idx:end_idx]
                
                if self.use_balanced_training:
                    if leading_network == "critic":
                        for _ in range(3):
                            g_loss, g_loss_adv, reg_loss = self.train_generator(real_data, real_labels)
                            g_losses.append(g_loss)
                            g_losses_adv.append(g_loss_adv.numpy())
                            reg_losses.append(reg_loss.numpy())
                            
                        d_loss = self.train_critic(real_data, real_labels)
                        d_losses.append(d_loss)
                        
                    else:
                        for _ in range(3):
                            d_loss = self.train_critic(real_data, real_labels)
                            d_losses.append(d_loss)
                            
                        g_loss, g_loss_adv, reg_loss = self.train_generator(real_data, real_labels)
                        g_losses.append(g_loss) 
                        g_losses_adv.append(g_loss_adv.numpy())
                        reg_losses.append(reg_loss.numpy())      
                else:
                    # Train critic multiple times for each generator update
                    for _ in range(self.discriminator_updates):
                        d_loss = self.train_critic(real_data, real_labels)
                        d_losses.append(d_loss)
                        
                    for _ in range(self.generator_updates):
                        g_loss, g_loss_adv, reg_loss = self.train_generator(real_data, real_labels)
                        g_losses.append(g_loss)
                        g_losses_adv.append(g_loss_adv.numpy())
                        reg_losses.append(reg_loss.numpy())

            avg_d_loss = np.mean(d_losses) if d_losses else 0
            avg_g_loss = np.mean(g_losses) if g_losses else 0
            avg_g_loss_adv = np.mean(g_losses_adv) if g_losses_adv else 0
            avg_reg_loss = np.mean(reg_losses) if reg_losses else 0
            
            if epoch == 0:
                    continue
            elif last_d_loss is not None and last_g_loss is not None:
                if abs(avg_d_loss-last_d_loss)/last_d_loss > abs(avg_g_loss-last_g_loss)/last_g_loss:
                    leading_network = "critic"
                else:
                    leading_network = "generator"   

            last_d_loss = avg_d_loss
            last_g_loss = avg_g_loss
            
            val_loss, mae, accuracy = self.evaluate(self.X_val, self.y_val)
            self.logger.info(f"Epoch {epoch}, D Loss: {avg_d_loss}, G Loss: {avg_g_loss}, G Loss Adv: {avg_g_loss_adv}, Reg Loss: {avg_reg_loss}")
            self.logger.info(f"Validation Loss: {val_loss}, Mean Absolute Error: {mae}, Accuracy: {accuracy}")

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
        y = self.predicting_variable.values.astype(np.float64)
        y = y.flatten()  # Ensure y is a 1D array of floats
        
        test_split_idx = int(len(X) * (1 - self.test_size))
        val_split_idx = int(test_split_idx * (1 - self.val_size))

        X_train, X_val, X_test = X[:val_split_idx], X[val_split_idx:test_split_idx], X[test_split_idx:]
        y_train, y_val, y_test = y[:val_split_idx], y[val_split_idx:test_split_idx], y[test_split_idx:]
        
        X_train_tuple = tuple(map(tuple, X_train))
        X_val_tuple = tuple(map(tuple, X_val))
        X_test_tuple = tuple(map(tuple, X_test))

        assert len(set(X_train_tuple).intersection(set(X_val_tuple))) == 0, "Train and validation sets overlap"
        assert len(set(X_train_tuple).intersection(set(X_test_tuple))) == 0, "Train and test sets overlap"
        assert len(set(X_val_tuple).intersection(set(X_test_tuple))) == 0, "Validation and test sets overlap"

        self.logger.info("Creating batches and formatted Y labels")
        self.X_train, self.y_train, self.y_train_batched, self.y_train_vector = self.prepare_data(X_train, y_train, self.window_size, self.batch_size)
        self.X_val, self.y_val, self.y_val_batched, self.y_val_vector = self.prepare_data(X_val, y_val, self.window_size, self.batch_size)
        self.X_test, self.y_test, self.y_test_batched, self.y_test_vector = self.prepare_data(X_test, y_test, self.window_size, self.batch_size)
        
        self.y_train = self.y_train.astype(np.float64).flatten()
        self.y_val = self.y_val.astype(np.float64).flatten()
        self.y_test = self.y_test.astype(np.float64).flatten()
        
        if self.use_grid_search:
            self.logger.info("Performing Hyperparameter Tuning with Keras Tuner")

            hypermodel = MyHyperModel(self.window_size, self.noise_dim, self.X_train.shape)
            tuner = RandomSearch(
                hypermodel,
                objective='val_loss',
                max_trials=40,  # Adjust the number of trials as needed
                executions_per_trial=1,
                directory='keras_tuner_dir',
                project_name='gan_tuning'
            )

            tuner.search_space_summary()

            noise_train = np.random.uniform(size=(self.X_train.shape[0], self.noise_dim))
            noise_val = np.random.uniform(size=(self.X_val.shape[0], self.noise_dim))

            tuner.search([self.X_train, noise_train], self.y_train, epochs=1000, validation_data=([self.X_val, noise_val], self.y_val), batch_size=self.batch_size)

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            self.logger.info(f"Best hyperparameters: {best_hps.values}")

            best_params = {
                'learning_rate_gen': best_hps.get('learning_rate_gen'),
                'learning_rate_disc': best_hps.get('learning_rate_disc'),
                'beta_1': best_hps.get('beta_1'),
                'dropout_rate': best_hps.get('dropout_rate'),
                'kernel_regularizer': best_hps.get('kernel_regularizer')
            }

            self.compile_gan(best_params)
            self.save_best_params(best_params)
        else:
            params = {
                'learning_rate_gen': self.learning_rate_gen,
                'learning_rate_disc': self.learning_rate_disc,
                'beta_1': self.beta_1,
                'dropout_rate': self.dropout_rate,
                'kernel_regularizer': self.kernel_regularizer,
            }
            self.compile_gan(params)
            self.train_gan(epochs=self.epochs, batch_size=self.batch_size)
        # self.save_model()  # Save model after training

    def evaluate(self, X, y_true):
        noise = np.random.uniform(size=(X.shape[0], self.noise_dim))
        y_pred = self.generator.predict([X, noise])
        y_true = y_true.reshape(-1, 1)  # Ensure y_true matches the shape of y_pred

        loss_fn = tf.keras.losses.MeanSquaredError()
        mae_fn = tf.keras.losses.MeanAbsoluteError()
        
        reg_loss = loss_fn(y_true, y_pred)
        mae = mae_fn(y_true, y_pred)
        
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, tf.round(y_pred)), tf.float32))

        print(f"Validation MSE: {reg_loss.numpy()}")
        print(f"Validation MAE: {mae.numpy()}")
        print(f"Validation Accuracy: {accuracy.numpy()}")

        return reg_loss.numpy(), mae.numpy(), accuracy.numpy()
    

    def custom_accuracy_metric(self, y_true, y_pred):
        """
        Custom accuracy metric to measure the exact match accuracy between predicted and true vectors.
        
        Parameters:
        - y_true: tensor of shape (batch_size, sequence_length, num_classes)
        - y_pred: tensor of shape (batch_size, sequence_length, num_classes)
        
        Returns:
        - accuracy: mean accuracy over the batch
        """
        if y_true.shape[0] > y_pred.shape[0]:
            y_true = y_true[:y_pred.shape[0]]
        
        matches = tf.reduce_all(tf.equal(y_true, y_pred), axis=-1)
        
        accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
        
        return accuracy
    
    def custom_loss_function(self, y_true, y_pred):
        """
        Custom loss function that computes the mean squared error between predicted and true vectors.
        
        Parameters:
        - y_true: tensor of shape (batch_size, sequence_length, num_classes)
        - y_pred: tensor of shape (batch_size, sequence_length, num_classes)
        
        Returns:
        - loss: mean squared error
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        if y_true.shape[0] > y_pred.shape[0]:
            y_true = y_true[:y_pred.shape[0]]
        
        loss = tf.reduce_mean(tf.square(y_pred - y_true))
        
        return loss

#################################################################################################################################################################
#
#
#                                                                  PREDICT FUNCTIONS
#
#################################################################################################################################################################

    def predict(self, data):
        noise = np.random.uniform(size=(data.shape[0], self.noise_dim))
        return self.generator.predict([data, noise])

    def preprocess_real_time_data(self, real_time_data):
        processed_data = real_time_data
        processed_data = np.expand_dims(processed_data, axis=2)
        return processed_data

    def predict_real_time(self, real_time_data):
        # Generate random noise
        noise = np.random.uniform(size=(real_time_data.shape[0], self.noise_dim))
        # Ensure both inputs are provided to the generator
        gen_predictions = self.generator.predict([real_time_data, noise])
        dis_predictions = self.critic.predict(gen_predictions)
        predictions = self.convert_predictions_to_class(dis_predictions)
        return predictions
    
    def convert_predictions_to_class(self, predictions):
        return np.argmax(predictions, axis=1)

    def calculate_metrics(self, y_true, y_pred):
        y_pred = y_pred  # Flatten y_pred to match the shape of y_true
        y_true = y_true[self.window_size:]  # Ensure y_true matches the length of y_pred
        acc = tf.keras.metrics.Accuracy(y_true, y_pred).numpy()
        prec = tf.keras.metrics.Precision(y_true, y_pred).numpy()
        self.logger.info(f"Accuracy: {acc}")
        self.logger.info(f"Precision: {prec}")
        return acc, prec

#################################################################################################################################################################
#
#
#                                                                  SAVE AND LOAD FUNCTIONS
#
#################################################################################################################################################################

    def save_model(self):
        model_dir = os.path.join(gan_path, "model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_filename = os.path.join(model_dir, f'gan_generator_{self.model_identifier}.keras')
        self.generator.save(model_filename)

        critic_filename = os.path.join(model_dir, f'gan_critic_{self.model_identifier}.keras')
        self.critic.save(critic_filename)

        self.logger.info(f"Models saved to {model_dir} with identifier {self.model_identifier}")
        
        # Save the model identifier for future reference
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

        generator_path = os.path.join(model_dir, f'gan_generator_{self.model_identifier}.keras')
        critic_path = os.path.join(model_dir, f'gan_critic_{self.model_identifier}.keras')

        custom_objects = {'wasserstein_loss': wasserstein_loss}

        # Load models with custom objects
        self.generator = load_model(generator_path, custom_objects=custom_objects)
        self.critic = load_model(critic_path, custom_objects=custom_objects)

        # Recompile models
        self.generator.compile(optimizer=Adam(learning_rate=self.learning_rate_gen), loss=wasserstein_loss)
        self.critic.compile(optimizer=Adam(learning_rate=self.learning_rate_disc), loss=wasserstein_loss)

        self.logger.info(f"Models loaded from {model_dir} with identifier {self.model_identifier}")

#################################################################################################################################################################
#
#
#                                                                  MAIN
#
#################################################################################################################################################################

    def main(self):
        self.train()
        self.load_model() 
        
        y_pred = self.predict_real_time(self.X_test)
        real_time_data = self.X_test[0]  # Example real-time data
        real_time_prediction = self.predict_real_time(real_time_data)
        print(f"Real-time prediction: {real_time_prediction}")


if __name__ == "__main__":
    config_filename = "config_GAN_regression.ini"
    model_trainer = LSTMGANModelTrainer(config_filename)
    model_trainer.main()
