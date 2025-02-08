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
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, ParameterGrid
import platform
import psutil
import joblib

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, LeakyReLU, Conv1D, MaxPooling1D, Flatten, Concatenate, RepeatVector, MultiHeadAttention, Add, LayerNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy, Loss, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.metrics import AUC, Precision, Recall, Accuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import custom_object_scope, get_registered_name, get_custom_objects
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.saving import register_keras_serializable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, confusion_matrix, classification_report
from keras_tuner import HyperModel, RandomSearch, Hyperband, BayesianOptimization, Objective
from sklearn.model_selection import KFold
from loguru import logger
import psutil
import shutil

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    
import mo_utils as utils
from DataLoader import DataLoader
from StrategyEvaluator import StrategyEvaluator
from Base_NN_skeletons import lstm_autotransformer_classification as nn_architecture, LearnablePositionalEncoding, feed_forward_network, PositionalEncoding
from HiddenMarkovModel import HiddenMarkovModel

precision_metric = tf.keras.metrics.Precision()
recall_metric = tf.keras.metrics.Recall()

def f1_score(y_true, y_pred):
    # Convert probabilities to predicted class labels
    y_pred_classes = tf.argmax(y_pred, axis=-1)

    # If y_true is one-hot encoded, convert it to class indices
    if len(tf.shape(y_true)) == len(tf.shape(y_pred)):
        y_true = tf.argmax(y_true, axis=-1)

    # Update precision and recall metrics
    precision_metric.update_state(y_true, y_pred_classes)
    recall_metric.update_state(y_true, y_pred_classes)

    # Retrieve precision and recall values
    precision = precision_metric.result()
    recall = recall_metric.result()

    # Compute F1 score
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    # Reset the state of the metrics to avoid accumulation
    precision_metric.reset_states()
    recall_metric.reset_states()

    return f1

@register_keras_serializable(name="CustomWeightedLoss")
class CustomWeightedLoss(tf.keras.losses.Loss):
    def __init__(self, class_weight_dict, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = tf.constant(list(class_weight_dict.values()), dtype=tf.float32)
        self.class_weight_dict = class_weight_dict

    def call(self, y_true, y_pred):
        # Flatten labels for consistent weighting
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])

        # Compute sparse categorical crossentropy
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        unweighted_loss = scce(y_true_flat, y_pred_flat)

        # Apply class weights
        weight = tf.gather(self.class_weights, tf.cast(y_true_flat, tf.int32))
        weighted_loss = unweighted_loss * weight

        return tf.reduce_mean(weighted_loss)

    def get_config(self):
        # Serialize class_weight_dict and other necessary attributes
        config = super().get_config()
        config.update({"class_weight_dict": self.class_weight_dict})
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize to create a new instance
        class_weight_dict = config.pop("class_weight_dict")
        return cls(class_weight_dict, **config)

@register_keras_serializable(name="CustomFocalLoss")
class CustomFocalLoss(Loss):
    def __init__(self, gamma=2.0, alpha=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = tf.constant(alpha, dtype=tf.float32) if alpha is not None else None

    def call(self, y_true, y_pred):
        # Clip predictions to avoid numerical issues
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # Convert y_true to one-hot if necessary
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])

        # Compute the focal loss
        cross_entropy = -y_true_one_hot * K.log(y_pred)
        focal_term = K.pow(1 - y_pred, self.gamma)
        loss = focal_term * cross_entropy

        # Apply class weights if alpha is provided
        if self.alpha is not None:
            loss *= self.alpha

        return K.mean(K.sum(loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha.numpy().tolist() if self.alpha is not None else None})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class CustomModelSaver(tf.keras.callbacks.Callback):
    def __init__(self, save_path):
        super(CustomModelSaver, self).__init__()
        self.save_path = save_path
        self.best_val_loss = np.inf  # Start with an infinitely large loss
        
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_f1_score')
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print(f" Validation f1 score improved to {val_loss:.4f}. Saving model...")
            self.model.save(self.save_path, save_format='keras')

# HyperModel for Keras Tuner
class LSTMAutoencoderHyperModel(HyperModel):
    def __init__(self, window_size, input_shape, loss_func, num_classes, weight_dict):
        self.window_size = window_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.loss_func_hyper = loss_func
        
    def build(self, hp):
        dropout_rate_lstm = hp.Float('dropout_rate_lstm', min_value=0, max_value=0.5, step=0.025)
        dropout_rate_dense = hp.Float('dropout_rate_dense', min_value=0, max_value=0.6, step=0.025)
        dropout_rate_multihead = hp.Float('dropout_rate_multihead', min_value=0, max_value=0.6, step=0.025)
        learning_rate = hp.Float('learning_rate', min_value=1e-7, max_value=1e-2, sampling='log')
        regularization_rate = hp.Float('regularization_rate', min_value=1e-7, max_value=1e-2, sampling='log')
        num_heads = hp.Choice('num_heads', values=[2, 4, 8,12,16])
        size_heads = hp.Choice('size_heads', values=[32, 64, 128, 256])
        d_dff_value = hp.Choice('d_ff', values=[128, 256, 512])
        
        if self.loss_func_hyper == "focal_loss":
            alpha = [self.weight_dict[i] for i in sorted(self.weight_dict.keys())]
            
        
        model = nn_architecture.build_sophisticated_lstm_transformer_model(
            input_shape=(self.window_size, self.input_shape[-1]),
            dropout_rate_lstm=dropout_rate_lstm,
            dropout_rate_dense=dropout_rate_dense,
            dropout_rate_multihead=dropout_rate_multihead,
            regularization_rate=regularization_rate,
            num_heads=num_heads,
            size_heads=size_heads,
            num_classes=self.num_classes,
            d_ff=d_dff_value
        )
        
        if self.loss_func_hyper == "sparse_categorical_crossentropy":
            model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=CustomWeightedLoss(self.weight_dict),
            metrics=['accuracy', f1_score])
        elif self.loss_func_hyper == "focal_loss":
            model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=CustomFocalLoss(alpha=alpha),
            metrics=['accuracy', f1_score])
        
        print(f"Trial Hyperparameters: {hp.values}")
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
        self.interval = utils.get_config_value(self.config, "general", "interval")

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
        self.predicting_variable += 1
        self.predicting_variable = self.predicting_variable.astype(np.int32) 

        if self.data.columns[-2] == "price_raw":    
            self.price_usdt = self.data[self.data.columns[-2]]
            self.dataset = self.data[self.data.columns[:-2]].copy()
        else:
            self.dataset = self.data[self.data.columns[:-1]].copy()
            
        self.dataset = self.dataset.clip(lower=-1, upper=1)
        self.dataset.replace(np.nan, 0, inplace=True)
        self.dataset.replace(np.inf, 0, inplace=True)
        self.dataset.replace(-np.inf, 0, inplace=True)
        
        self.model_identifier = utils.get_config_value(self.config, "general", "model_identifier")
        self.checkpoint_filepath = (os.path.join(gan_path, "model", "autotransformer", f"lstm_autotransformer_{self.model_identifier}.hdf5"))
        self.model_save_path = os.path.join(gan_path, "model", "autotransformer",  f"lstm_autotransformer_{self.model_identifier}.keras")
        self.best_model_save_path = os.path.join(gan_path, "model","autotransformer",  f"best_lstm_autotransformer_{self.model_identifier}.keras")
        
        if not os.path.exists(os.path.join(gan_path, "model", "autotransformer")):
            os.makedirs(os.path.join(gan_path, "model", "autotransformer"))
        
        self.window_size = int(utils.get_config_value(self.config, "params", 'window_size'))
        self.epochs = int(utils.get_config_value(self.config, "params", "epochs"))
        self.batch_size = int(utils.get_config_value(self.config, "params", "batch_size"))
        self.learning_rate = float(utils.get_config_value(self.config, "params", "learning_rate"))
        self.lstm_units = int(utils.get_config_value(self.config, "params", "lstm_units"))
        self.dropout_rate_lstm = float(utils.get_config_value(self.config, "params", "dropout_rate_lstm"))
        self.dropout_rate_dense = float(utils.get_config_value(self.config, "params", "dropout_rate_dense"))
        self.dropout_rate_multihead = float(utils.get_config_value(self.config, "params", "dropout_rate_multihead"))
        self.kernel_regularizer = float(utils.get_config_value(self.config, "params", "kernel_regularizer"))
        self.size_heads = int(utils.get_config_value(self.config, "params", "size_heads"))
        self.num_heads = int(utils.get_config_value(self.config, "params", "num_heads"))
        self.d_ff = int(utils.get_config_value(self.config, "params", "d_ff"))
        self.early_stopping_patience = int(utils.get_config_value(self.config, "params", "early_stopping_patience"))
        self.seed = int(utils.get_config_value(self.config, "params", "seed"))
        self.val_size = float(utils.get_config_value(self.config, "params", "val_size"))
        self.test_size = float(utils.get_config_value(self.config, "params", "test_size"))
        
        self.use_grid_search = utils.get_config_value(self.config, "general", "use_grid_search")
        self.use_balanced_features = utils.get_config_value(self.config, "general", "use_balanced_features")
        self.used_loss_func = utils.get_config_value(self.config, "general", "used_loss_func")
        self.use_actual_class_weights = utils.get_config_value(self.config, "general", "use_actual_class_weights")
        
        self.num_classes = len(np.unique(self.predicting_variable))
        
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

        for i in range(len(data) - window_size):
            sequences.append(data[i:i + window_size])  # Input features
            sequence_labels.append(labels[i:i + window_size])  # Label for each timestep

        return np.array(sequences, dtype=np.float64), np.array(sequence_labels, dtype=np.int32)

    def prepare_data(self, data, labels, window_size, batch_size):
        data_sequences, data_labels = self.create_sequences(data, labels, window_size)

        # Truncate to fit batch size
        num_batches = len(data_sequences) // batch_size
        data_sequences = data_sequences[:num_batches * batch_size]
        data_labels = data_labels[:num_batches * batch_size]

        return data_sequences, data_labels

    def prepare_data_with_hmm(self):

        HMM = HiddenMarkovModel("config_hmm.ini")
        HMM.train_hmm(self.dataset)
        model_name = HMM.hmm_model_name
        market_states = HMM.predict_states(self.dataset, load_model_name=model_name)
        # HMM.save_hmm_model("autotransformer_model")    
        # HMM.load_hmm_model(os.path.join(gan_path,"model","hmm_params.pt"))

        # Plotting the data
        price_data_path = os.path.join(data_path_crypto,"Historical data","price_usdt_kucoin",self.interval,f"price_usdt_kucoin_{self.coin}_{self.fiat}_{self.interval}.csv")
        price_data = pd.read_csv(price_data_path, index_col=0)
        price_data.index = pd.to_datetime(price_data.index, format="%Y-%m-%d %H:%M:%S")
        price_series = price_data["close"]
        price_series = price_series.reindex(self.dataset.index, method="ffill")
        
        plot_save_path = os.path.join(data_path_crypto, "Data Analysis", "HMM market states")   #f"market_states_{self.model_identifier}.png"
        
        HMM.plot_market_states(market_states, self.dataset.index, price_series, plot_save_path)
        
        scaler = MinMaxScaler()
        self.dataset['market_state'] = scaler.fit_transform(market_states.reshape(-1, 1))
    
        return
    
    
    def build_model(self):
        input_shape = (self.window_size, self.dataset.shape[-1])
        self.model = nn_architecture.build_sophisticated_lstm_transformer_model(input_shape, 
                                                                                dropout_rate_lstm=self.dropout_rate_lstm,
                                                                                dropout_rate_dense=self.dropout_rate_dense,
                                                                                dropout_rate_multihead=self.dropout_rate_multihead,
                                                                                regularization_rate = self.kernel_regularizer, 
                                                                                num_heads=self.num_heads,
                                                                                size_heads=self.size_heads,
                                                                                num_classes=len(np.unique(self.predicting_variable)),
                                                                                d_ff=self.d_ff)

    def save_model(self):
        model_dir = os.path.join(gan_path, "model","autotransformer")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_filename = os.path.join(model_dir, f'lstm_autotransformer_{self.model_identifier}.keras')
        self.model.save(model_filename)

        self.logger.info(f"Model saved to {model_dir} with identifier {self.model_identifier}")
        
        with open(os.path.join(model_dir, 'latest_model_id.txt'), 'w') as f:
            f.write(str(self.model_identifier))

    def save_best_params(self, best_params):
        params_dir = os.path.join(gan_path, "model","autotransformer")
        if not os.path.exists(params_dir):
            os.makedirs(params_dir)
        params_filename = os.path.join(params_dir, 'best_params.json')
        with open(params_filename, 'w') as f:
            json.dump(best_params.values, f)
        self.logger.info(f"Best parameters saved to {params_filename}")

    def load_model(self):
        model_dir = os.path.join(gan_path, "model","autotransformer")
        self.class_weight_dict = self.load_custom_weights(gan_path, self.model_identifier)
        
        custom_objects = get_custom_objects()
        for name, obj in custom_objects.items():
            print(f"{name}: {obj}")
        
        # Ensure the `class_weight_dict` is defined before loading the model
        if hasattr(self, 'class_weight_dict') and self.class_weight_dict is not None and self.used_loss_func == "sparse_categorical_crossentropy":
            loss_func = CustomWeightedLoss
        elif hasattr(self, 'class_weight_dict') and self.class_weight_dict is not None and self.used_loss_func == "focal_loss":
            loss_func = CustomFocalLoss
        else:
            loss_func = SparseCategoricalCrossentropy(from_logits=False)
            
        if not os.path.exists(self.model_save_path):
            self.logger.error("No model found. Please train a model first.")
            pass
        else:
        #     with tf.keras.utils.custom_object_scope({'custom_weighted_loss': custom_loss}):
        #         self.model = tf.keras.models.load_model(
        #     self.model_save_path,
        #     custom_objects={'custom_weighted_loss': custom_loss}
        # )
            # Load the model with the custom loss function
            if self.used_loss_func == "sparse_categorical_crossentropy":    
                self.model = tf.keras.models.load_model(self.model_save_path,
                                                        custom_objects={'CustomWeightedLoss': loss_func, 
                                                                        'LearnablePositionalEncoding': LearnablePositionalEncoding})
            elif self.used_loss_func == "focal_loss":
                self.model = tf.keras.models.load_model(self.model_save_path,
                                                        custom_objects={'CustomFocalLoss': loss_func, 
                                                                        'LearnablePositionalEncoding': LearnablePositionalEncoding})
                
            self.logger.info(f"Model loaded from {model_dir} with identifier {self.model_identifier}")
            return self.model
        
        if os.path.exists(os.path.join(model_dir, 'latest_model_id.txt')):
            try:
                with open(os.path.join(model_dir, 'latest_model_id.txt'), 'r') as f:
                    self.model_identifier = f.read().strip()
                    self.model = tf.keras.models.load_model(os.path.join(model_dir, f'lstm_autotransformer_{self.model_identifier}'))
            except FileNotFoundError:
                self.logger.error("No model identifier found. Please specify an identifier.")
                return

    def save_custom_weights(self, weights, save_path, identifier):
        """
        Save custom weights to a JSON file.
        
        Args:
            weights (dict): The class weights to save.
            save_path (str): Path to the directory where the weights will be saved.
            identifier (str): Identifier for the model, used to name the file.
        """
        weights_filename = os.path.join(save_path, f'class_weights_{identifier}.json')
        with open(weights_filename, 'w') as f:
            json.dump(weights, f)
        print(f"Custom weights saved to {weights_filename}")
    
    
    def load_custom_weights(self, save_path, identifier):
        """
        Load custom weights from a JSON file.
        
        Args:
            save_path (str): Path to the directory where the weights are saved.
            identifier (str): Identifier for the model, used to locate the file.
        
        Returns:
            dict: Loaded class weights.
        """
        weights_filename = os.path.join(save_path, f'class_weights_{identifier}.json')
        if not os.path.exists(weights_filename):
            raise FileNotFoundError(f"Custom weights file {weights_filename} not found.")
        with open(weights_filename, 'r') as f:
            weights = json.load(f)
        print(f"Custom weights loaded from {weights_filename}")
        return weights
    
    def setup_training(self):
        X = self.dataset.values
        y = self.predicting_variable.values
        # y = y.flatten()

        test_split_idx = int(len(X) * (1 - self.test_size))
        val_split_idx = int(test_split_idx * (1 - self.val_size))

        X_train, X_val, X_test = X[:val_split_idx], X[val_split_idx:test_split_idx], X[test_split_idx:]
        y_train, y_val, y_test = y[:val_split_idx], y[val_split_idx:test_split_idx], y[test_split_idx:]

        self.X_train, self.y_train = self.prepare_data(X_train, y_train, self.window_size, self.batch_size)
        self.X_val, self.y_val = self.prepare_data(X_val, y_val, self.window_size, self.batch_size)
        self.X_test, self.y_test = self.prepare_data(X_test, y_test, self.window_size, self.batch_size)

        print("Unique classes in y_train:", np.unique(self.y_train))
        print("Unique classes in y_val:", np.unique(self.y_val))
        
        if self.use_balanced_features:
            
            if self.use_actual_class_weights:
                self.class_weights = compute_class_weight(
                    class_weight='balanced',  # Use keyword argument for 'balanced'
                    classes=np.unique(self.predicting_variable),  # Unique classes
                    y=self.predicting_variable.values)
                self.class_weight_dict = dict(enumerate(self.class_weights))
            else:
                self.class_weight_dict = {0:3.0, 1:1.0, 2:3.0}
            
            path = os.path.join(gan_path, "model", "autotransformer")
            self.save_custom_weights(self.class_weight_dict, path, self.model_identifier)
        
    def train_model(self):    
        early_stopping = EarlyStopping(
                         monitor='val_f1_score', 
                         patience=self.early_stopping_patience if not self.use_grid_search else 10, 
                         restore_best_weights=True)
        
        
        reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',  # Reduce LR if validation loss stops improving
                    factor=0.5,          # Reduce LR by half
                    patience=5,          # Wait 5 epochs before reducing
                    min_lr=1e-6          # Minimum learning rate
                )
        
        # with tf.keras.utils.custom_object_scope({'custom_weighted_loss': custom_weighted_loss}):
        model_checkpoint = ModelCheckpoint(
                    self.checkpoint_filepath, 
                    save_best_only=True, 
                    monitor='val_loss', 
                    mode='min', 
                    save_format='keras')
        
        
        custom_saver = CustomModelSaver(self.model_save_path)

        if self.use_grid_search:
            self.logger.info("Performing Hyperparameter Tuning with Keras Tuner")

            print("Class Weights for Hyperparameter Search:", self.class_weight_dict)
            
            if self.use_balanced_features and self.used_loss_func == "sparse_categorical_crossentropy":
                loss_func = CustomWeightedLoss(self.class_weight_dict)
            elif self.used_loss_func == "focal_loss":
                
                alpha = [self.class_weight_dict[i] for i in sorted(self.class_weight_dict.keys())]
                loss_func = CustomFocalLoss(gamma=2.0, alpha=alpha)
            else:
                loss_func = SparseCategoricalCrossentropy(from_logits=False)
            
            # hypermodel = LSTMAutoencoderHyperModel(self.window_size,(self.window_size, self.dataset.shape[-1]), len(np.unique(self.predicting_variable)), self.class_weight_dict)
            
            # tuner = RandomSearch(
            #     hypermodel,
            #     objective='val_loss',
            #     max_trials=20,
            #     executions_per_trial=1,
            #     directory='keras_tuner_dir',
            #     project_name='lstm_autoencoder_tuning'
            # )
            
            # Remove any existing tuner directory to ensure a fresh search
            tuner_dir = f'tuning_dir/lstm_autotransformer/{self.model_identifier}'
            if os.path.exists(tuner_dir):
                shutil.rmtree(tuner_dir)
                
                # Create the tuner directory if it doesn't exist
            os.makedirs(tuner_dir, exist_ok=True)
            
            tuner = BayesianOptimization(
                hypermodel=LSTMAutoencoderHyperModel(self.window_size, (self.window_size, self.dataset.shape[-1]), self.used_loss_func, len(np.unique(self.predicting_variable)), self.class_weight_dict),
                objective=Objective('val_f1_score', direction='max'),  # Primary objective
                max_trials=50,  # Number of hyperparameter combinations to try
                directory='tuner_dir',
                project_name='lstm_autotransformer_tuning'
            )
            
            tuner.search(self.X_train, self.y_train, epochs=50, validation_data=(self.X_val, self.y_val), batch_size=self.batch_size, callbacks=[early_stopping, reduce_lr])

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            self.logger.info(f"Best Hyperparameters: {best_hps.values}")
            
            # Build and evaluate the best model
            best_model = tuner.get_best_models(num_models=1)[0]
            best_model.evaluate(self.X_test, self.y_test)
            best_model.save(self.best_model_save_path)

            
            # Rebuild Model with Best Hyperparameters
            self.build_model()
            self.model.compile(
                optimizer=Adam(learning_rate=best_hps.get('learning_rate')),
                loss=loss_func,
                metrics=['accuracy', f1_score]
            )
            self.save_best_params(best_hps)
            
            history = self.model.fit(self.X_train, self.y_train,
                                    epochs=self.epochs,
                                    batch_size=self.batch_size,
                                    validation_data=(self.X_val, self.y_val),
                                    callbacks=[early_stopping, model_checkpoint, custom_saver, reduce_lr])
            
            
        else:
            
            self.build_model()
                        
            if self.use_balanced_features and self.used_loss_func == "sparse_categorical_crossentropy":
                loss_func = CustomWeightedLoss(self.class_weight_dict)
            elif self.used_loss_func == "focal_loss":
                alpha = [self.class_weight_dict[i] for i in sorted(self.class_weight_dict.keys())]
                loss_func = CustomFocalLoss(gamma=2.0, alpha=alpha)
            else:
                loss_func = SparseCategoricalCrossentropy(from_logits=False)
            
            self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss_func,
            metrics=['accuracy', f1_score] 
            )
            
            history = self.model.fit(self.X_train, self.y_train, 
                                    epochs=self.epochs, 
                                    batch_size=self.batch_size, 
                                    validation_data=(self.X_val, self.y_val),
                                    callbacks=[early_stopping, custom_saver, reduce_lr])   
            
            return history

    def evaluate_model(self):
        self.model = self.load_model()
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        y_pred = self.model.predict(self.X_test)
        predicted_classes = np.argmax(y_pred, axis=-1)
        final_predictions = predicted_classes[:, -1]
        y_test_last_timestep = self.y_test[:, -1]  # Shape: (6176,)
        # label_mapping = {0: -1, 1: 0, 2: 1}
        # decoded_predictions = np.vectorize(label_mapping.get)(final_predictions)    
        conf_matrix = confusion_matrix(y_test_last_timestep, final_predictions)
        class_report = classification_report(y_test_last_timestep, final_predictions, target_names=["Class -1", "Class 0", "Class 1"])    
        self.logger.info(f"Test Loss: {loss}")
        self.logger.info(f"Test Accuracy: {accuracy}")
        return loss, accuracy, conf_matrix, class_report

    def predict(self, data):
        data_sequences, _ = self.create_sequences(data, np.zeros((data.shape[0],)), self.window_size)
        predictions = self.model.predict(data_sequences)
        predicted_classes = np.argmax(predictions, axis=1)
        return predicted_classes
    
    
    def visualize_results(self, loss, accuracy, conf_matrix, class_report):
        # Plot the confusion matrix
        plt.figure(figsize=(10, 7))
        seaborn.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Class -1", "Class 0", "Class 1"], yticklabels=["Class -1", "Class 0", "Class 1"])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        # Print the classification report
        print("Classification Report:\n", class_report)

        # Plot the loss and accuracy
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss over Epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Accuracy over Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.show()
    
    def main(self):
        self.prepare_data_with_hmm() 
        self.setup_training()
        self.train_model()
        loss, accuracy, conf_matrix, class_report = self.evaluate_model()
        self.visualize_results(loss, accuracy, conf_matrix, class_report)
        y_pred = self.predict_real_time(self.X_test)
        real_time_data = self.X_test[0]  # Example real-time data
        real_time_prediction = self.predict_real_time(real_time_data)
        print(f"Real-time prediction: {real_time_prediction}")



if __name__ == "__main__":
    config_filename = "config_LSTM_transformer.ini"
    model_trainer = LSTMAutoencoderModelTrainer(config_filename)
    model_trainer.main()