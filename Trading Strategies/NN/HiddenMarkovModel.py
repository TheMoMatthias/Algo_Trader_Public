#############################################
# File: pyro_hmm_advanced.py
#############################################
import os
import sys
import platform
import json
import numpy as np
import torch
import optuna
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from pyro.optim import Adam as PyroAdam

from pyro.infer import SVI, TraceEnum_ELBO, Trace_ELBO
from pyro.infer.autoguide import AutoDelta, AutoNormal
from pyro.infer import config_enumerate
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
from datetime import datetime
from torch.distributions import constraints
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import gc
import joblib

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --------------
# Path configuration
# (Mirroring your approach; tweak as needed)
# --------------
def get_running_environment():
    if 'microsoft-standard' in platform.uname().release:
        return 'wsl'
    elif platform.system() == 'Windows':
        return 'windows'
    else:
        return 'unknown'

def convert_path(path, env):
    """Convert path from Windows to WSL or vice versa."""
    if env == 'wsl':
        return path.replace('C:\\', '/mnt/c/').replace('\\', '/')
    elif env == 'windows':
        return path.replace('/mnt/c/', 'C:\\').replace('/', '\\')
    else:
        return path

env = get_running_environment()

def get_converted_path(path):
    return convert_path(path, env)

# Example of reading a config for the HMM
def read_hmm_config_file(config_path):
    """
    Reads a JSON or INI config for HMM parameters.
    For demonstration, we assume JSON. Adjust as needed.
    Returns a dict with keys: 'n_states', 'n_iter', etc.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

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

class HiddenMarkovModel:
    """
    A fully Bayesian HMM using Pyro with enumerated discrete states,
    diagonal Gaussian emissions, and advanced time-series modeling.
    Maintains the same 'train_hmm', 'predict_states', 'save_hmm_model',
    'load_hmm_model', and 'plot_market_states' signatures as your
    hmmlearn-based HiddenMarkovModel.
    """

    def __init__(self, config_filename="hmm_config.json"):
        """
        We read the config file (path can be absolute or relative)
        and set up Pyro-based HMM parameters.
        
        Your model function includes several regularization terms:

        L2 Regularization (l2_reg_weight):

        Applies to transition matrix A, emission means, and scales.
        
        Transition Regularization (transition_reg_weight):

        - Encourages the transition matrix to approach a uniform distribution.
        
        Emission Regularization (emission_reg_weight):

        - Penalizes large emission parameters.
        
        Sparse Transitions Regularization (sparse_transition_weight):

        Introduces L1 regularization to promote sparsity in the transition matrix.
        Hierarchical Priors (hierarchical_prior):       
        
        """
        # ~~~~~ PATHS ~~~~~
        # (mirroring your script's style)
        # You can customize or remove as needed.
        self.env = get_running_environment()
        
        config_path = utils.find_config_path()
        self.config = utils.read_config_file(os.path.join(config_path, "strategy config", "NN", config_filename))
        
        self.use_optuna = utils.get_config_value(self.config, "HMM_SETTINGS", "use_optuna")
        self.n_optuna_trials = utils.get_config_value(self.config, "HMM_SETTINGS","n_optuna_trials")
        
        # Basic HMM parameters
        self.n_states = utils.get_config_value(self.config, "HMM_SETTINGS", "n_states")
        self.n_iter = utils.get_config_value(self.config, "HMM_SETTINGS", "n_iter") 
        self.tol = utils.get_config_value(self.config, "HMM_SETTINGS", "tol")
        self.init_params = utils.get_config_value(self.config, "HMM_SETTINGS", "init_params")
        self.init_model_randomness = utils.get_config_value(self.config, "HMM_SETTINGS", "init_model_randomness")
        self.load_existing_model = utils.get_config_value(self.config, "HMM_SETTINGS", "load_existing_model")
        self.used_autoguide = utils.get_config_value(self.config, "HMM_SETTINGS", "used_autoguide")
        
        # Optional PCA for HMM
        self.use_pca_for_hmm = utils.get_config_value(self.config, "HMM_SETTINGS", "use_pca_for_hmm")
        self.hmm_pca_components = utils.get_config_value(self.config, "HMM_SETTINGS", "hmm_pca_components")
        self.load_pca_transformer = utils.get_config_value(self.config, "HMM_SETTINGS", "load_pca_transformer")
        
        # Additional advanced parameters
        self.time_lag = utils.get_config_value(self.config, "HMM_SETTINGS", "time_lag")
        self.learning_rate = utils.get_config_value(self.config, "HMM_SETTINGS", "learning_rate")
        self.dirichlet_alpha = utils.get_config_value(self.config, "HMM_SETTINGS", "dirichlet_alpha")
        self.emission_cov_scale = utils.get_config_value(self.config, "HMM_SETTINGS", "emission_cov_scale")
        self.transition_reg_weight = utils.get_config_value(self.config, "HMM_SETTINGS", "transition_reg_weight")
        self.emission_reg_weight = utils.get_config_value(self.config, "HMM_SETTINGS", "emission_reg_weight")
        self.means_init_scale = utils.get_config_value(self.config, "HMM_SETTINGS", "means_init_scale")
        self.l2_reg_weight = utils.get_config_value(self.config, "HMM_SETTINGS", "l2_reg_weight") 
        self.sparse_transition_weight = utils.get_config_value(self.config, "HMM_SETTINGS", "sparse_transition_weight")
        self.hierarchical_prior = utils.get_config_value(self.config, "HMM_SETTINGS", "hierarchical_prior")
        
        #hmm model paths 
        self.hmm_model_name = utils.get_config_value(self.config, "HMM_SETTINGS", "hmm_model_save_path")
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[PyroHMMAdvanced] Using device: {self.device}")

        if self.used_autoguide == "AutoNormal":
            self.auto_guide = AutoNormal(self.model)
        else:
            self.auto_guide = AutoDelta(self.model)
        
        # Use a more expressive guide
        # self.auto_guide = AutoNormal(self.model)
        
        # Flag to know if we've fit
        self._fitted = False
        self.svi = None
        self.writer = SummaryWriter(log_dir=utils.find_logging_path())

    ################################################
    # Optional: Preprocess features with time_lag
    ################################################
    def _build_lagged_features(self, raw_features):
        """
        Enhance feature engineering by ensuring appropriate time lag and normalization.
        """
        if self.time_lag <= 1:
            return raw_features

        T, D = raw_features.shape
        out_length = T - self.time_lag + 1
        if out_length <= 0:
            raise ValueError(f"Not enough timesteps ({T}) for time_lag={self.time_lag}.")

        # Create the lagged matrix
        lagged_data = []
        for t in range(out_length):
            segment = raw_features[t : t + self.time_lag].flatten()
            lagged_data.append(segment)

        lagged_array = np.array(lagged_data, dtype=np.float32)

        # Normalize features
        mean = lagged_array.mean(axis=0)
        std = lagged_array.std(axis=0).clip(min=1e-6)
        lagged_array = (lagged_array - mean) / std

        return lagged_array

    ################################################
    # The enumerated HMM model & guide
    ################################################
    
    def model(self, observations):
        """
        Defines the HMM model using Pyro's DiscreteHMM with optional initialization randomness
        and sparse transitions.
        """
        T, D = observations.shape

        # Calculate data statistics
        data_mean = observations.mean(dim=0)
        data_std = observations.std(dim=0).clamp(min=1e-6)

        # pi: Initial state probabilities
        pi = pyro.sample(
            "pi",
            dist.Dirichlet(
                self.dirichlet_alpha * torch.ones(self.n_states, device=self.device)
            )
        )
        pi = pi.clamp(min=1e-15)
        pi = pi / pi.sum()

        # Transition matrix A
        with pyro.plate("transitions", self.n_states):
            if self.init_model_randomness:
                # Introduce slight randomness per row
                A = pyro.sample(
                    "A",
                    dist.Dirichlet(
                        (self.dirichlet_alpha + 0.1) * torch.ones(self.n_states, device=self.device) + 
                        0.05 * torch.rand(self.n_states, device=self.device)
                    )
                )
            else:
                # Deterministic initialization per row
                A = pyro.sample(
                    "A",
                    dist.Dirichlet(
                        self.dirichlet_alpha * torch.ones(self.n_states, device=self.device)
                    )
                )
        A = A.clamp(min=1e-15)
        A = A / A.sum(dim=1, keepdim=True)

        # Emission parameters: means & scales
        with pyro.plate("emission_states", self.n_states):
            if self.hierarchical_prior:
                # Hierarchical Priors Implementation
                # Shared hyperparameters for means and scales
                global_mean = pyro.sample(
                    "global_mean",
                    dist.Normal(torch.zeros(D, device=self.device), torch.ones(D, device=self.device)).to_event(1)
                )
                global_scale = pyro.sample(
                    "global_scale",
                    dist.HalfCauchy(scale=torch.ones(D, device=self.device)).to_event(1)
                )

                means = pyro.sample(
                    "means",
                    dist.Normal(
                        global_mean,
                        global_scale * self.means_init_scale * torch.ones(D, device=self.device)
                    ).to_event(1)
                )
            else:
                # Non-hierarchical Emission Means
                if self.init_model_randomness:
                    # Introduce slight randomness in means
                    means = pyro.sample(
                        "means",
                        dist.Normal(
                            data_mean + self.means_init_scale * torch.randn(D, device=self.device),
                            self.means_init_scale * torch.ones(D, device=self.device),
                        ).to_event(1)
                    )
                else:
                    # Deterministic initialization without added randomness
                    means = pyro.sample(
                        "means",
                        dist.Normal(
                            data_mean,
                            self.means_init_scale * torch.ones(D, device=self.device),
                        ).to_event(1)
                    )
            
            # Scales remain non-hierarchical; modify if hierarchical scales are desired
            raw_scales = pyro.sample(
                "scales",
                dist.HalfCauchy(self.emission_cov_scale * torch.ones(D, device=self.device)).to_event(1)
            )
            # Clamp to ensure positivity
            scales = raw_scales.clamp(min=1e-6)

        # Regularization terms
        if self.l2_reg_weight > 0:
            l2_reg = (A ** 2).sum() + (means ** 2).sum() + (scales ** 2).sum()
            pyro.factor("l2_reg", -self.l2_reg_weight * l2_reg)

        if self.transition_reg_weight > 0:
            uniform_A = (1.0 / self.n_states) * torch.ones_like(A)
            diff_A = (A - uniform_A).pow(2).sum()
            pyro.factor("transition_reg", -self.transition_reg_weight * diff_A)

        if self.emission_reg_weight > 0:
            emission_penalty = means.pow(2).sum() + scales.pow(2).sum()
            pyro.factor("emission_reg", -self.emission_reg_weight * emission_penalty)

        # **New: Sparse Transitions Regularization**
        if self.sparse_transition_weight > 0:
            # L1 regularization to encourage sparsity in the transition matrix
            l1_reg = A.abs().sum()
            pyro.factor("sparse_transition_reg", -self.sparse_transition_weight * l1_reg)

        # Build the DiscreteHMM
        emission_dist = dist.Independent(
            dist.Normal(means, scales),
            reinterpreted_batch_ndims=1
        )
        hmm_dist = dist.DiscreteHMM(
            initial_logits=torch.log(pi + 1e-15),
            transition_logits=torch.log(A + 1e-15),
            observation_dist=emission_dist
        )

        with pyro.plate("sequences", 1):
            pyro.sample("obs", hmm_dist, obs=observations)
    
    def reset_environment(self):
        """
        Resets the Python environment by clearing Pyro's parameter store and reinitializing the guide and optimizer.
        """
        import gc
        import torch.nn.utils as nn_utils
        import pyro
        from pyro.infer import SVI, Trace_ELBO
        from pyro.contrib.autoguide import AutoNormal, AutoDelta

        # Clear Pyro's parameter store
        pyro.clear_param_store()
        print("[PyroHMMAdvanced] Pyro parameter store cleared.")

        # Reinitialize the guide
        if self.used_autoguide == "AutoNormal":
            self.auto_guide = AutoNormal(self.model)
        else:
            self.auto_guide = AutoDelta(self.model)
        
        print("[PyroHMMAdvanced] AutoNormal guide reinitialized.")

        # Reinitialize the optimizer
        optimizer = PyroAdam({"lr": self.learning_rate})
        print(f"[PyroHMMAdvanced] Optimizer reinitialized with learning rate: {self.learning_rate}")

        # Reinitialize SVI
        self.svi = SVI(
            model=self.model,
            guide=self.auto_guide,
            optim=optimizer,
            loss=Trace_ELBO()
        )
        print("[PyroHMMAdvanced] SVI reinitialized.")

        # Collect garbage to free up memory
        gc.collect()
        torch.cuda.empty_cache()
        print("[PyroHMMAdvanced] Garbage collection completed.")

    ################################################
    # TRAINING
    ################################################
    def train_hmm(
        self,
        features,
        max_retries=5,
        tolerance=1e-12,
        n_components_range=(2, 15),
        n_splits=5,
        n_restarts=5
    ):
        """
        If self.use_optuna == True, we run an Optuna study to find 
        the best set of hyperparameters. Otherwise, single-run enumerated SVI.
        """
        # Possibly build lagged features below, but let's handle it in objective() 
        # if needed in the single-run approach as well.

        if self.use_optuna:
            print("[PyroHMMAdvanced] Starting Optuna hyperparameter search.")
            self._run_optuna_search(features)
        else:
            print("[PyroHMMAdvanced] Single-run enumerated discrete HMM training.")
            self._single_run_train(features)

        # end train_hmm

    ###############################
    # SINGLE-RUN APPROACH
    ###############################

    def _single_run_train(self, features, debug=True):
        """
        Single-run training with:
        - Early Stopping
        - Gradient Clipping (only for leaf tensors)
        - ELBO optimization with manual learning rate decay and clamping of scale parameters
        - Optional PCA transformation (if use_pca_for_hmm is True)
        - Optional feature selection (using the hyperparameter n_selected_features)
    
        """
        import pyro, torch, gc, os, joblib
        import torch.nn.utils as nn_utils
        from sklearn.decomposition import PCA

        # --- Scenario 1: Load Existing Model ---
        if self.load_existing_model:
            print("[PyroHMMAdvanced] load_existing_model flag is True. Loading saved model and hyperparameters...")
            self.load_hmm_model(self.hmm_model_name)
            if self.use_pca_for_hmm:
                if self.load_pca_transformer:
                    # Try to load the saved PCA transformer.
                    pca_path = os.path.join(gan_path, "model", "hmm", self.hmm_model_name + "_best_pca.joblib")
                    if os.path.exists(pca_path):
                        self.pca_transformer = joblib.load(pca_path)
                        features = self.pca_transformer.transform(features)
                        print("[PyroHMMAdvanced] PCA transformer loaded from file and applied.")
                    else:
                        print("[PyroHMMAdvanced] PCA transformer file not found. Fitting a new PCA transformer.")
                        self.pca_transformer = PCA(n_components=self.hmm_pca_components)
                        features = self.pca_transformer.fit_transform(features)
                        pca_dir = os.path.join(gan_path, "model", "hmm")
                        os.makedirs(pca_dir, exist_ok=True)
                        best_pca_name = self.hmm_model_name + "_best_pca.joblib"
                        joblib.dump(self.pca_transformer, os.path.join(pca_dir, best_pca_name))
                else:
                    # load_existing_model True but load_pca_transformer flag is False:
                    print("[PyroHMMAdvanced] load_existing_model is True but load_pca_transformer flag is False. Fitting a new PCA transformer.")
                    self.pca_transformer = PCA(n_components=self.hmm_pca_components)
                    features = self.pca_transformer.fit_transform(features)
                    pca_dir = os.path.join(gan_path, "model", "hmm")
                    os.makedirs(pca_dir, exist_ok=True)
                    best_pca_name = self.hmm_model_name + "_best_pca.joblib"
                    joblib.dump(self.pca_transformer, os.path.join(pca_dir, best_pca_name))
            self._fitted = True
            return
        else:
            # If PCA is enabled:
            if self.use_pca_for_hmm:
                if self.load_pca_transformer:
                    # Try to load an existing PCA transformer.
                    pca_path = os.path.join(gan_path, "model", "hmm", self.hmm_model_name + "_best_pca.joblib")
                    if os.path.exists(pca_path):
                        self.pca_transformer = joblib.load(pca_path)
                        features = self.pca_transformer.transform(features)
                        print("[PyroHMMAdvanced] PCA transformer loaded from file and applied.")
                    else:
                        print("[PyroHMMAdvanced] PCA transformer file not found. Fitting new PCA transformer.")
                        self.pca_transformer = PCA(n_components=self.hmm_pca_components)
                        features = self.pca_transformer.fit_transform(features)
                        pca_dir = os.path.join(gan_path, "model", "hmm")
                        os.makedirs(pca_dir, exist_ok=True)
                        best_pca_name = self.hmm_model_name + "_best_pca.joblib"
                        joblib.dump(self.pca_transformer, os.path.join(pca_dir, best_pca_name))
                        print(f"[PyroHMMAdvanced] New PCA transformer saved as {best_pca_name}")
                else:
                    # Always fit a new PCA transformer.
                    print("[PyroHMMAdvanced] load_pca_transformer flag is False. Fitting new PCA transformer.")
                    self.pca_transformer = PCA(n_components=self.hmm_pca_components)
                    features = self.pca_transformer.fit_transform(features)
                    pca_dir = os.path.join(gan_path, "model", "hmm")
                    os.makedirs(pca_dir, exist_ok=True)
                    best_pca_name = self.hmm_model_name + "_best_pca.joblib"
                    joblib.dump(self.pca_transformer, os.path.join(pca_dir, best_pca_name))
                    print(f"[PyroHMMAdvanced] New PCA transformer saved as {best_pca_name}")
            else:
                if hasattr(features, "values"):
                    features = features.values
        # --- Scenario 2: Train a New Model ---
        self.reset_environment()

        # Apply feature selection if specified.
        if hasattr(self, "n_selected_features") and self.n_selected_features < features.shape[1]:
            features = features[:, self.selected_columns]
        
        # Build lagged features from the (possibly transformed) full dataset.
        lagged = self._build_lagged_features(features)
        train_torch = torch.tensor(lagged, dtype=torch.float, device=self.device)

        # Initialize optimizer and SVI.
        from pyro.optim import Adam as PyroAdam
        optimizer = PyroAdam({"lr": self.learning_rate})
        from pyro.infer import SVI, Trace_ELBO
        self.svi = SVI(model=self.model, guide=self.auto_guide, optim=optimizer, loss=Trace_ELBO())

        best_loss = float("inf")
        best_params = None
        no_improve_steps = 0
        early_stop_steps = 1000
        lr_decay_factor = 0.5
        lr_patience = 250
        lr_no_improve = 0
        steps_since_lr_reduction = 0

        print(f"[PyroHMMAdvanced] Starting final training with tol={self.tol}")
        for step in range(1, self.n_iter + 1):
            loss = self.svi.step(train_torch)
            steps_since_lr_reduction += 1

            for name, param in pyro.get_param_store().named_parameters():
                if param.requires_grad and param.grad is not None and param.is_leaf:
                    nn_utils.clip_grad_norm_(param, max_norm=1.0)
            for name, param in pyro.get_param_store().named_parameters():
                if "scales" in name:
                    param.data.clamp_(min=0.01)

            if loss < best_loss - self.tol:
                best_loss = loss
                best_params = pyro.get_param_store().get_state()
                no_improve_steps = 0
                lr_no_improve = 0
            else:
                no_improve_steps += 1
                lr_no_improve += 1

            if lr_no_improve >= lr_patience and steps_since_lr_reduction >= 150:
                current_lr = self.learning_rate
                new_lr = current_lr * lr_decay_factor
                if new_lr < 1e-8:
                    print(f"[PyroHMMAdvanced] Learning rate below threshold ({new_lr:.8f}). Stopping training.")
                    break
                print(f"[PyroHMMAdvanced] Reducing learning rate from {current_lr} to {new_lr}")
                self.learning_rate = new_lr

                if hasattr(self, "svi"):
                    del self.svi
                if 'optimizer' in locals():
                    del optimizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                optimizer = PyroAdam({"lr": self.learning_rate})
                self.svi = SVI(model=self.model, guide=self.auto_guide, optim=optimizer, loss=Trace_ELBO())
                lr_no_improve = 0
                steps_since_lr_reduction = 0

            if step % 500 == 0:
                print(f"[PyroHMMAdvanced] step {step}/{self.n_iter} loss={loss:.4f}, best={best_loss:.4f}")

            if no_improve_steps >= early_stop_steps:
                print(f"[PyroHMMAdvanced] Early stopping triggered at step {step}. Best loss={best_loss:.4f}")
                break

        if best_params is not None:
            pyro.clear_param_store()
            pyro.get_param_store().set_state(best_params)

        self._fitted = True  # Mark final model as fitted.
        print(f"[PyroHMMAdvanced] Final training complete. Best loss={best_loss:.4f}")
        self.writer.close()

        self.save_hmm_model(self.hmm_model_name)

    def _reduce_learning_rate(self, factor=0.5, min_lr=1e-6):
        """
        Manually reduce the learning rate for all unique torch optimizers by a given factor.
        Ensures that learning rates do not go below min_lr.
        """
        optimizers = set()
        for name, param in pyro.get_param_store().named_parameters():
            optimizer = self.svi.optim._get_optim(param)
            if optimizer not in optimizers:
                optimizers.add(optimizer)
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * factor, min_lr)
                    param_group['lr'] = new_lr
                    print(f"[PyroHMMAdvanced] Reduced learning rate from {old_lr:.6f} to {new_lr:.6f} for optimizer handling param '{name}'.")
    
    def _run_optuna_search(self, features, debug=True):
        """
        Perform hyperparameter search via Optuna.
        For each trial:
        - Optionally apply PCA (if pca_apply is true) so that the emission parameters
            (means and scales) are learned in the PCA-reduced space.
        - Additionally, a random feature transformation is chosen: either a random subset
            of features is selected or PCA is applied. The chosen feature selection (or PCA)
            configuration is saved permanently.
        - Each trial's model is saved with a unique name (self.hmm_model_name_{trial.number}),
            and if PCA is applied, its transformer is saved as well.
        After the study, the best trial’s hyperparameters are applied locally and:
        - The best model is saved as self.hmm_model_name_best.
        - If PCA was used in the best trial, its PCA transformer is saved as self.hmm_model_name_best_pca.joblib.
        """
        import optuna
        import numpy as np
        import pandas as pd
        import pyro

        def objective(trial):
            # Clear the parameter store for a fresh start.
            pyro.clear_param_store()

            # --- Hyperparameter Suggestions ---
            n_states = trial.suggest_int("n_states", 2, 10)
            time_lag = trial.suggest_int("time_lag", 1, 5)
            trans_reg = trial.suggest_float("transition_reg_weight", 1e-6, 1e-2, log=True)
            emiss_reg = trial.suggest_float("emission_reg_weight", 1e-6, 1e-2, log=True)
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            sparse_trans = trial.suggest_float("sparse_transition_weight", 1e-6, 1e-2, log=True)
            hierarchical = trial.suggest_categorical("hierarchical_prior", [True, False])
            dirichlet_alpha = trial.suggest_float("dirichlet_alpha", 0.01, 5, log=True)
            emission_cov_scale = trial.suggest_float("emission_cov_scale", 0.5, 5.0, log=True)
            means_init_scale = trial.suggest_float("means_init_scale", 1.0, 10.0, log=True)
            l2_reg_weight = trial.suggest_float("l2_reg_weight", 1e-6, 1e-2, log=True)
            
            # --- New: Feature Transformation Choice ---
            # Let the optimizer choose whether to use a random subset of features or PCA.
            feature_transform = trial.suggest_categorical("feature_transform", ["subset", "pca","no_transform"])
            
            # --- Save Old Hyperparameter Values ---
            old_n_states = self.n_states
            old_time_lag = self.time_lag
            old_tr_weight = self.transition_reg_weight
            old_er_weight = self.emission_reg_weight
            old_lr = self.learning_rate
            old_sparse_trans = self.sparse_transition_weight
            old_hierarchical = self.hierarchical_prior
            old_dirichlet_alpha = self.dirichlet_alpha
            old_emission_cov_scale = self.emission_cov_scale
            old_means_init_scale = self.means_init_scale
            old_l2_reg_weight = self.l2_reg_weight

            # --- Update Model Attributes for This Trial ---
            self.n_states = n_states
            self.time_lag = time_lag
            self.transition_reg_weight = trans_reg
            self.emission_reg_weight = emiss_reg
            self.learning_rate = lr
            self.sparse_transition_weight = sparse_trans
            self.hierarchical_prior = hierarchical
            self.dirichlet_alpha = dirichlet_alpha
            self.emission_cov_scale = emission_cov_scale
            self.means_init_scale = means_init_scale
            self.l2_reg_weight = l2_reg_weight

            # --- Reinitialize the AutoGuide ---
            if self.used_autoguide == "AutoNormal":
                from pyro.contrib.autoguide import AutoNormal
                self.auto_guide = AutoNormal(self.model)
            else:
                from pyro.contrib.autoguide import AutoDelta
                self.auto_guide = AutoDelta(self.model)

            pyro.clear_param_store()

            # --- Data Preprocessing with Feature Transformation ---
            from sklearn.model_selection import train_test_split
            train_features, val_features = train_test_split(features, test_size=0.4, shuffle=False, random_state=42)
            
            # Determine total number of features.
            n_all_features = train_features.shape[1]
            selected_columns = None
            pca_transformer = None
            
            if feature_transform == "subset":
                # Random subset selection.
                pca_apply = False
                trial.set_user_attr("pca_apply", pca_apply)
                n_selected_features = trial.suggest_int("n_selected_features", 1, n_all_features)
                if isinstance(train_features, pd.DataFrame):
                    all_columns = train_features.columns.tolist()
                    selected_columns = np.random.choice(all_columns, size=n_selected_features, replace=False)
                    train_features = train_features.loc[:, selected_columns]
                    val_features = val_features.loc[:, selected_columns]
                else:
                    all_idx = np.arange(n_all_features)
                    selected_columns = np.random.choice(all_idx, size=n_selected_features, replace=False)
                    train_features = train_features[:, selected_columns]
                    val_features = val_features[:, selected_columns]
                # Save the selected columns (or indices) permanently.
                if isinstance(selected_columns[0], str):
                    trial.set_user_attr("selected_columns", list(selected_columns))
                else:
                    trial.set_user_attr("selected_columns", selected_columns.tolist())
                self.selected_columns = trial.user_attrs.get("selected_columns")
            elif feature_transform == "pca":
                # PCA transformation.
                if self.use_pca_for_hmm:
                    pca_apply = True
                    trial.set_user_attr("pca_apply", pca_apply)
                    if pca_apply:
                        from sklearn.decomposition import PCA
                        n_original_features = train_features.shape[1]
                        hmm_pca_components = trial.suggest_int("hmm_pca_components", 1, n_original_features)
                        pca_transformer = PCA(n_components=hmm_pca_components)
                        train_features = pca_transformer.fit_transform(train_features)
                        val_features = pca_transformer.transform(val_features)
                else:
                    pca_apply = False
                    trial.set_user_attr("pca_apply", pca_apply)
                    # If not applying PCA, convert to numpy.
                    if isinstance(train_features, pd.DataFrame):
                        train_features = train_features.values
                        val_features = val_features.values
            elif feature_transform == "no_transform":
                # No transformation.
                pca_apply = False
                trial.set_user_attr("pca_apply", pca_apply)
                if isinstance(train_features, pd.DataFrame):
                    train_features = train_features.values
                    val_features = val_features.values 

            # Convert to numpy array if still a DataFrame.
            if isinstance(train_features, pd.DataFrame):
                train_features = train_features.values
                val_features = val_features.values

            # --- Build Lagged Features ---
            train_lagged = self._build_lagged_features(train_features)
            val_lagged = self._build_lagged_features(val_features)
            import torch
            train_torch = torch.tensor(train_lagged, dtype=torch.float, device=self.device)
            val_torch = torch.tensor(val_lagged, dtype=torch.float, device=self.device)

            # --- Save the PCA Transformer for This Trial (if applied) ---
            if self.use_pca_for_hmm and pca_transformer is not None:
                import os, joblib
                pca_dir = os.path.join(gan_path, "model", "hmm")
                os.makedirs(pca_dir, exist_ok=True)
                trial_pca_name = f"{self.hmm_model_name}_{trial.number}_pca.joblib"
                joblib.dump(pca_transformer, os.path.join(pca_dir, trial_pca_name))
                print(f"[Optuna] Trial {trial.number} PCA transformer saved as {trial_pca_name}")
                trial.set_user_attr("pca_transformer", pca_transformer)

            # --- Initialize Optimizer and SVI ---
            from pyro.optim import Adam as PyroAdam
            optimizer = PyroAdam({"lr": self.learning_rate})
            from pyro.infer import SVI, Trace_ELBO
            self.svi = SVI(model=self.model, guide=self.auto_guide, optim=optimizer, loss=Trace_ELBO())

            best_val_loss_for_run = float("inf")
            best_params_for_run = None
            no_improve_steps = 0
            early_stop_steps = 1000
            lr_decay_factor = 0.5
            lr_patience = 500
            lr_no_improve = 0
            steps_since_lr_reduction = 0

            # --- Training Loop for This Trial ---
            for step in range(1, self.n_iter + 1):
                loss = self.svi.step(train_torch)
                steps_since_lr_reduction += 1

                import torch.nn.utils as nn_utils
                for name, param in pyro.get_param_store().named_parameters():
                    if param.requires_grad and param.grad is not None and param.is_leaf:
                        nn_utils.clip_grad_norm_(param, max_norm=1.0)
                for name, param in pyro.get_param_store().named_parameters():
                    if "scales" in name:
                        param.data.clamp_(min=0.01)

                val_loss = self.evaluate(val_torch)

                # --- Compute a Penalty for Scales Near the Lower Bound ---
                penalty = 0.0
                for name, param in pyro.get_param_store().named_parameters():
                    if "scales" in name:
                        penalty += torch.sum(torch.clamp(0.01 - param.data, min=0.0)).item()
                lambda_penalty = 1e-3
                composite_loss = val_loss + lambda_penalty * penalty

                if composite_loss < best_val_loss_for_run - self.tol:
                    best_val_loss_for_run = composite_loss
                    best_params_for_run = pyro.get_param_store().get_state()
                    no_improve_steps = 0
                    lr_no_improve = 0
                else:
                    no_improve_steps += 1
                    lr_no_improve += 1

                if lr_no_improve >= lr_patience and steps_since_lr_reduction >= 150:
                    current_lr = self.learning_rate
                    new_lr = current_lr * lr_decay_factor
                    if new_lr < 1e-8:
                        print(f"[Optuna] Learning rate below threshold ({new_lr:.8f}). Stopping training.")
                        break
                    print(f"[Optuna] Reducing learning rate from {current_lr} to {new_lr}")
                    self.learning_rate = new_lr

                    if hasattr(self, "svi"):
                        del self.svi
                    if 'optimizer' in locals():
                        del optimizer
                    import gc
                    gc.collect()
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    optimizer = PyroAdam({"lr": self.learning_rate})
                    self.svi = SVI(model=self.model, guide=self.auto_guide, optim=optimizer, loss=Trace_ELBO())
                    lr_no_improve = 0
                    steps_since_lr_reduction = 0

                if step % 500 == 0:
                    print(f"[Optuna Trial] step {step}/{self.n_iter} train loss={loss:.4f}, "
                        f"val loss={val_loss:.4f}, composite loss={composite_loss:.4f}, best composite loss={best_val_loss_for_run:.4f}")

                if no_improve_steps >= early_stop_steps:
                    print(f"[Optuna] Early stopping triggered at step {step}. Best composite loss={best_val_loss_for_run:.4f}")
                    break

                if debug and step % 1000 == 0:
                    for name, param in pyro.get_param_store().named_parameters():
                        if torch.isnan(param).any():
                            print(f"[Optuna] Parameter {name} contains NaNs.")
                        if torch.isinf(param).any():
                            print(f"[Optuna] Parameter {name} contains Infs.")

            if best_params_for_run is not None:
                pyro.clear_param_store()
                pyro.get_param_store().set_state(best_params_for_run)

            # --- Save This Trial's Model with a Unique Name ---
            run_model_name = f"{self.hmm_model_name}_{trial.number}"
            self.save_hmm_model(run_model_name)
            print(f"[Optuna] Trial {trial.number} model saved as {run_model_name}")

            if best_params_for_run is not None:
                trial.set_user_attr("best_params_for_run", best_params_for_run)

            # Restore old hyperparameters.
            self.n_states = old_n_states
            self.time_lag = old_time_lag
            self.transition_reg_weight = old_tr_weight
            self.emission_reg_weight = old_er_weight
            self.learning_rate = old_lr
            self.sparse_transition_weight = old_sparse_trans
            self.hierarchical_prior = old_hierarchical
            self.dirichlet_alpha = old_dirichlet_alpha
            self.emission_cov_scale = old_emission_cov_scale
            self.means_init_scale = old_means_init_scale
            self.l2_reg_weight = old_l2_reg_weight

            return best_val_loss_for_run

        # --- End of Objective Function ---

        # Create and run the study (only once)
        study = optuna.create_study(direction="minimize")
        print(f"[Optuna] Running Optuna with {self.n_optuna_trials} trials...")
        study.optimize(objective, n_trials=self.n_optuna_trials)

        # Retrieve the best trial.
        best_trial = study.best_trial
        print("[Optuna] Best trial found:")
        for k, v in best_trial.params.items():
            print(f"   {k}: {v}")
        print(f"   -> Best Composite Loss: {best_trial.value:.4f}")

        best_param_store = best_trial.user_attrs.get("best_params_for_run", None)
        if best_param_store is not None:
            pyro.clear_param_store()
            pyro.get_param_store().set_state(best_param_store)

        # Update local hyperparameters to reflect the best trial.
        self.n_states = best_trial.params["n_states"]
        self.time_lag = best_trial.params["time_lag"]
        self.transition_reg_weight = best_trial.params["transition_reg_weight"]
        self.emission_reg_weight = best_trial.params["emission_reg_weight"]
        self.learning_rate = best_trial.params["learning_rate"]
        self.sparse_transition_weight = best_trial.params["sparse_transition_weight"]
        self.hierarchical_prior = best_trial.params["hierarchical_prior"]
        self.dirichlet_alpha = best_trial.params["dirichlet_alpha"]
        self.emission_cov_scale = best_trial.params["emission_cov_scale"]
        self.means_init_scale = best_trial.params["means_init_scale"]
        self.l2_reg_weight = best_trial.params["l2_reg_weight"]
        self.n_selected_features = best_trial.params["n_selected_features"]

        if self.use_pca_for_hmm:
            self.hmm_pca_components = best_trial.params.get("hmm_pca_components", self.hmm_pca_components)
            best_pca = best_trial.user_attrs.get("pca_transformer", None)
            if best_pca is not None:
                import os, joblib
                pca_dir = os.path.join(gan_path, "model", "hmm")
                os.makedirs(pca_dir, exist_ok=True)
                best_pca_name = self.hmm_model_name + "_best_pca.joblib"
                joblib.dump(best_pca, os.path.join(pca_dir, best_pca_name))
                print(f"[Optuna] Best PCA transformer saved as {best_pca_name}")

        if self.used_autoguide == "AutoNormal":
            from pyro.contrib.autoguide import AutoNormal
            self.auto_guide = AutoNormal(self.model)
        else:
            from pyro.contrib.autoguide import AutoDelta
            self.auto_guide = AutoDelta(self.model)

        if hasattr(self, "svi"):
            del self.svi
        if 'optimizer' in locals():
            del optimizer
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from pyro.optim import Adam as PyroAdam
        optimizer = PyroAdam({"lr": self.learning_rate})
        from pyro.infer import SVI, Trace_ELBO
        self.svi = SVI(model=self.model, guide=self.auto_guide, optim=optimizer, loss=Trace_ELBO())

        # Mark the best model as fitted so that predict_states can run.
        self._fitted = True

        # --- Save the Best Model with Optimal Hyperparameters ---
        best_model_name = self.hmm_model_name + "_best"
        self.save_hmm_model(best_model_name)
        print(f"[Optuna] Done. Best model saved as {best_model_name}.")
 

    ################################################
    # EVALUATION
    ################################################
    def evaluate(self, val_torch):
        """
        Evaluates the model's loss on validation data.
        """
        with torch.no_grad():
            loss = self.svi.evaluate_loss(val_torch)
        return loss


    ################################################
    # PREDICTING STATES
    ################################################
    def predict_states_vectorized(self, features, debug=False):
        """
        Vectorized implementation of the Viterbi algorithm for faster prediction.

        Args:
            features (np.ndarray or torch.Tensor): Input features, shape [T, D].
            debug (bool): If True, enables detailed logging.
        Returns:
            np.ndarray: Predicted state sequence, shape [T_].
        """
        import pyro
        import pyro.distributions as dist
        import torch
        import torch.nn.functional as F
        from pyro.distributions.transforms import StickBreakingTransform

        if not self._fitted:
            print("[PyroHMMAdvanced] Model not fitted yet.")
            return None

        # Preprocess features (e.g. applying time lag)
        features_lagged = self._build_lagged_features(features)
        data_torch = torch.tensor(features_lagged, dtype=torch.float, device=self.device)

        if debug:
            print(f"[PyroHMMAdvanced] Data shape for prediction: {data_torch.shape}")
            print("Parameter store keys:", list(pyro.get_param_store().keys()))

        if self.used_autoguide == "AutoDelta":
            try:
                pi = pyro.param("AutoDelta.pi").detach()
                A = pyro.param("AutoDelta.A").detach()
                means = pyro.param("AutoDelta.means").detach()
                scales = pyro.param("AutoDelta.scales").detach().clamp(min=1e-6)
            except KeyError as e:
                print(f"[PyroHMMAdvanced] Missing parameter in AutoDelta guide: {e}")
                return None
        else:
            # Using AutoNormal for unconstrained representations.
            # For π, apply the stick-breaking transform to the raw unconstrained parameter.
            try:
                raw_pi = pyro.param("AutoNormal.locs.pi").detach()
                sb_transform = StickBreakingTransform()
                pi = sb_transform(raw_pi)
            except KeyError:
                try:
                    # Fallback: use the constrained value directly if available.
                    pi = pyro.param("AutoNormal.pi").detach()
                except KeyError as e:
                    print(f"[PyroHMMAdvanced] Missing parameter for pi in AutoNormal guide: {e}")
                    return None

            # For A, we similarly apply the stick-breaking transform along the last dimension.
            try:
                raw_A = pyro.param("AutoNormal.locs.A").detach()
            except KeyError:
                try:
                    raw_A = pyro.param("AutoNormal.scales.A").detach()
                except KeyError as e:
                    print(f"[PyroHMMAdvanced] Missing parameter for A in AutoNormal guide: {e}")
                    return None
            # raw_A is expected to be of shape [n_states, n_states-1] (e.g. [3, 2])
            sb_transform = StickBreakingTransform()
            A = sb_transform(raw_A)  # now A should have shape [3, 3]

            try:
                means = pyro.param("AutoNormal.locs.means").detach()
            except KeyError:
                try:
                    means = pyro.param("AutoNormal.means").detach()
                except KeyError as e:
                    print(f"[PyroHMMAdvanced] Missing parameter for means in AutoNormal guide: {e}")
                    return None
            try:
                scales = pyro.param("AutoNormal.scales.scales").detach().clamp(min=1e-6)
            except KeyError:
                try:
                    raw_scales = pyro.param("AutoNormal.locs.scales").detach()
                    scales = torch.exp(raw_scales).clamp(min=1e-6)
                except KeyError as e:
                    print(f"[PyroHMMAdvanced] Missing parameter for scales in AutoNormal guide: {e}")
                    return None

        # After transformation, we expect:
        #   pi: a vector of shape [n_states] (e.g. [3])
        #   A: a matrix of shape [n_states, n_states] (e.g. [3, 3])
        #   means: [n_states, D_lagged]
        #   scales: [n_states, D_lagged]
        expected_states = means.shape[0]  # should be 3

        if pi.dim() == 1 and pi.shape[0] != expected_states:
            print(f"[PyroHMMAdvanced] Warning: Expected π to have {expected_states} elements but got {pi.shape[0]}.")
            deficit = expected_states - pi.shape[0]
            if deficit > 0:
                print(f"[PyroHMMAdvanced] Padding π with {deficit} zeros.")
                pi = F.pad(pi, (0, deficit), mode="constant", value=1e-10)
                pi = torch.softmax(pi, dim=0)

        if A.dim() == 2 and A.shape[1] != expected_states:
            print(f"[PyroHMMAdvanced] Warning: Expected A to have {expected_states} columns but got {A.shape[1]}.")
            deficit = expected_states - A.shape[1]
            if deficit > 0:
                print(f"[PyroHMMAdvanced] Padding A with {deficit} zeros on dimension 1.")
                A = F.pad(A, (0, deficit), mode="constant", value=1e-10)
                A = A / A.sum(dim=1, keepdim=True)

        pi_post = torch.softmax(pi, dim=0)
        A_post = torch.softmax(A, dim=1)

        if debug:
            print(f"[PyroHMMAdvanced] π_post shape: {pi_post.shape}")
            print(f"[PyroHMMAdvanced] A_post shape: {A_post.shape}")
            print(f"[PyroHMMAdvanced] means shape: {means.shape}")
            print(f"[PyroHMMAdvanced] scales shape: {scales.shape}")

        n_states = expected_states  # now forced to match the emission parameters
        T_ = data_torch.shape[0]
        D_lagged = data_torch.shape[1]

        # Compute emission log probabilities.
        data_expanded = data_torch.unsqueeze(1)      # Shape: [T_, 1, D_lagged]
        means_expanded = means.unsqueeze(0)          # Shape: [1, n_states, D_lagged]
        scales_expanded = scales.unsqueeze(0)        # Shape: [1, n_states, D_lagged]

        emission_dist = dist.Normal(means_expanded, scales_expanded)
        log_prob_states = emission_dist.log_prob(data_expanded)  # [T_, n_states, D_lagged]
        emission_log_probs = log_prob_states.sum(dim=-1)         # [T_, n_states]

        # Viterbi forward pass: initialize log_delta.
        log_delta = torch.log(pi_post + 1e-15) + emission_log_probs[0]  # [n_states]
        backpointer = torch.zeros((T_, n_states), dtype=torch.long, device=self.device)

        for t in range(1, T_):
            transition_log_probs = log_delta.unsqueeze(1) + torch.log(A_post + 1e-15)  # [n_states, n_states]
            max_log_probs, argmax_states = torch.max(transition_log_probs, dim=0)        # [n_states]
            log_delta = max_log_probs + emission_log_probs[t]                           # [n_states]
            backpointer[t] = argmax_states
            if debug and t % 1000 == 0:
                print(f"[PyroHMMAdvanced] Time {t}: log_delta={log_delta}")

        # Backtracking to recover the most likely state sequence.
        path = torch.zeros(T_, dtype=torch.long, device=self.device)
        _, last_state = torch.max(log_delta, dim=0)
        path[T_ - 1] = last_state

        for t in reversed(range(1, T_)):
            path[t - 1] = backpointer[t, path[t]]

        return path.cpu().numpy()

    def predict_states(self, features, load_model_name=None,debug=False):
        """
        Returns the most likely sequence of hidden states (Viterbi path)
        for the given 'features' array, shape [T, D].
        
        If self.load_existing_model is True, the function always loads the saved model
        and hyperparameters (and PCA transformer if applicable) before prediction.
        Otherwise, it proceeds normally.
        """
        import os
        import joblib
        
        # If load_existing_model is enabled, load the saved model unconditionally.
        if getattr(self, "load_existing_model", False):
            print("[PyroHMMAdvanced] 'load_existing_model' flag is True. Loading saved model and hyperparameters for prediction...")
            if load_model_name is not None:
                self.hmm_model_name = load_model_name
                self.load_hmm_model(load_model_name)
            else:
                self.load_hmm_model(self.hmm_model_name)
            # Always reload the PCA transformer if PCA is used.
            if self.use_pca_for_hmm and getattr(self, "load_pca_transformer", False):
                pca_path = os.path.join(gan_path, "model", "hmm", "pca_hmm.joblib")
                if os.path.exists(pca_path):
                    self.pca_transformer = joblib.load(pca_path)
                    print("[PyroHMMAdvanced] PCA transformer loaded for prediction.")
                else:
                    print("[PyroHMMAdvanced] PCA transformer file not found during prediction.")
        
        # If PCA is enabled, transform the features accordingly.
        if self.use_pca_for_hmm:
            if hasattr(self, "pca_transformer"):
                features = self.pca_transformer.transform(features)
            else:
                # Fit PCA on the fly, then save it.
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.hmm_pca_components)
                features = pca.fit_transform(features)
                pca_dir = os.path.join(gan_path, "model", "hmm")
                os.makedirs(pca_dir, exist_ok=True)
                joblib.dump(pca, os.path.join(pca_dir, "pca_hmm.joblib"))
                print("[PyroHMMAdvanced] PCA transformer fitted and saved during prediction.")
        else:
            if hasattr(features, "values"):
                features = features.values

        # Apply feature selection if specified.
        if hasattr(self, "n_selected_features") and self.n_selected_features < features.shape[1]:
            features = features[:, self.selected_columns]
        
        # Now call the vectorized prediction function.
        return self.predict_states_vectorized(features, debug=debug)
    
    ################################################
    # SAVING & LOADING
    ################################################
    def save_hmm_model(self, filename):
        # Save Pyro's parameter store
        save_dir = os.path.join(gan_path, "model", "hmm")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_file_name_model = filename + "_model.pt"
        save_path_model = os.path.join(save_dir, save_file_name_model)
        pyro.get_param_store().save(save_path_model)
        print(f"[PyroHMMAdvanced] Param store saved to {save_path_model}")

        # Save hyperparameters, now including selected columns if available.
        hyperparameters = {
            "n_states": self.n_states,
            "n_iter": self.n_iter,
            "tol": self.tol,
            "init_params": self.init_params,
            "time_lag": self.time_lag,
            "learning_rate": self.learning_rate,
            "dirichlet_alpha": self.dirichlet_alpha,
            "emission_cov_scale": self.emission_cov_scale,
            "transition_reg_weight": self.transition_reg_weight,
            "emission_reg_weight": self.emission_reg_weight,
            "means_init_scale": self.means_init_scale,
            "l2_reg_weight": self.l2_reg_weight,
            "sparse_transition_weight": self.sparse_transition_weight,
            "hierarchical_prior": self.hierarchical_prior,
            "init_model_randomness": self.init_model_randomness,
            "load_pca_transformer": self.load_pca_transformer,
            "load_existing_model": getattr(self, "load_existing_model", False),
            "hmm_pca_components": self.hmm_pca_components,
            # Save the selected features if available.
            "selected_columns": self.selected_columns if hasattr(self, "selected_columns") else None
        }
        
        if self.use_pca_for_hmm:
            hyperparameters["hmm_pca_components"] = self.hmm_pca_components
        
        save_file_name_model_params = filename + "_hyperparams.json"
        save_path_model_params = os.path.join(save_dir, save_file_name_model_params)
        
        with open(save_path_model_params, "w") as f:
            import json
            json.dump(hyperparameters, f, indent=4)
        print(f"[PyroHMMAdvanced] Hyperparameters saved to {save_path_model_params}")


    

    def load_hmm_model(self, filename):
        import json
        
        save_dir = os.path.join(gan_path, "model", "hmm")
        save_file_name_model = filename + "_model.pt"
        save_path_model = os.path.join(save_dir, save_file_name_model)
        
        # Load Pyro's parameter store
        pyro.clear_param_store()
        pyro.get_param_store().load(save_path_model, map_location=self.device)
        print(f"[PyroHMMAdvanced] Param store loaded from {str(save_path_model)}")
        
        save_file_name_model_params = filename + "_hyperparams.json"
        save_path_model_params = os.path.join(save_dir, save_file_name_model_params)
        # Load hyperparameters
        with open(save_path_model_params, "r") as f:
            hyperparameters = json.load(f)
        
        self.n_states = hyperparameters.get("n_states", self.n_states)
        self.n_iter = hyperparameters.get("n_iter", self.n_iter)
        self.tol = hyperparameters.get("tol", self.tol)
        self.init_params = hyperparameters.get("init_params", self.init_params)
        self.time_lag = hyperparameters.get("time_lag", self.time_lag)
        self.learning_rate = hyperparameters.get("learning_rate", self.learning_rate)
        self.dirichlet_alpha = hyperparameters.get("dirichlet_alpha", self.dirichlet_alpha)
        self.emission_cov_scale = hyperparameters.get("emission_cov_scale", self.emission_cov_scale)
        self.transition_reg_weight = hyperparameters.get("transition_reg_weight", self.transition_reg_weight)
        self.emission_reg_weight = hyperparameters.get("emission_reg_weight", self.emission_reg_weight)
        self.means_init_scale = hyperparameters.get("means_init_scale", self.means_init_scale)
        self.l2_reg_weight = hyperparameters.get("l2_reg_weight", self.l2_reg_weight)
        self.sparse_transition_weight = hyperparameters.get("sparse_transition_weight", self.sparse_transition_weight)
        self.hierarchical_prior = hyperparameters.get("hierarchical_prior", self.hierarchical_prior)
        self.init_model_randomness = hyperparameters.get("init_model_randomness", self.init_model_randomness)
        self.use_pca_for_hmm = hyperparameters.get("use_pca_for_hmm", self.use_pca_for_hmm)
        self.load_pca_transformer = hyperparameters.get("load_pca_transformer", self.load_pca_transformer)
        
        self.selected_columns = hyperparameters.get("selected_columns", None)
        
        if self.use_pca_for_hmm:
            self.hmm_pca_components = hyperparameters.get("hmm_pca_components", self.hmm_pca_components)
        
        # If desired, load the PCA transformer here as well
        if self.use_pca_for_hmm and self.load_pca_transformer:
            pca_path = os.path.join(gan_path, "model", "hmm", "pca_hmm.joblib")
            if os.path.exists(pca_path):
                self.pca_transformer = joblib.load(pca_path)
                print(f"[PyroHMMAdvanced] PCA transformer loaded from {pca_path}.")
            else:
                print("[PyroHMMAdvanced] PCA transformer file not found.")
        
        self._fitted = True
        print(f"[PyroHMMAdvanced] Hyperparameters loaded from {str(save_path_model_params)}")

    ################################################
    # INSPECTING PARAMETERS
    ################################################
    def inspect_learned_parameters(self, debug=False):
        """
        Inspects the learned HMM parameters.
        """
        with torch.no_grad():
            if self.used_autoguide == "AutoNormal":
                try:
                    pi_loc = pyro.param("AutoNormal.pi_loc").detach()
                    pi_post = torch.softmax(pi_loc, dim=0)
                    A_loc = pyro.param("AutoNormal.A_loc").detach()
                    A_post = torch.softmax(A_loc, dim=1)
                    means_post = pyro.param("AutoNormal.means_loc").detach()
                    scales_post = pyro.param("AutoNormal.scales_loc").detach().clamp(min=1e-6)
                except KeyError as e:
                    print(f"[PyroHMMAdvanced] Missing parameter in param store: {e}")
                    return
            else:
                try:
                    pi_loc = pyro.param("AutoDelta.pi").detach()
                    pi_post = torch.softmax(pi_loc, dim=0)
                    A_loc = pyro.param("AutoDelta.A").detach()
                    A_post = torch.softmax(A_loc, dim=1)
                    means_post = pyro.param("AutoDelta.means").detach()
                    scales_post = pyro.param("AutoDelta.scales").detach().clamp(min=1e-6)
                except KeyError as e:
                    print(f"[PyroHMMAdvanced] Missing parameter in param store: {e}")
                    return
        if debug:
            print(f"[PyroHMMAdvanced] Learned pi_post: {pi_post}")
            print(f"[PyroHMMAdvanced] Learned A_post: {A_post}")
            print(f"[PyroHMMAdvanced] Learned means_post: {means_post}")
            print(f"[PyroHMMAdvanced] Learned scales_post: {scales_post}")

        # Additional checks:
        print(f"[PyroHMMAdvanced] pi_post: {pi_post}")
        print(f"[PyroHMMAdvanced] Transition Matrix A_post:\n{A_post}")
        print(f"[PyroHMMAdvanced] Means:\n{means_post}")
        print(f"[PyroHMMAdvanced] Scales:\n{scales_post}")

    def list_parameters(self):
        """
        Lists all parameters in the Pyro parameter store.
        """
        params = pyro.get_param_store().get_state()
        print("[PyroHMMAdvanced] Registered parameters:")
        for name, param in params.items():
            print(f"  {name}: shape={param.shape}, min={param.min().item():.4f}, max={param.max().item():.4f}")

    
    ################################################
    # PLOT MARKET STATES
    ################################################
    def plot_market_states(self, market_states, timestamps, price_data, save_path):
        """
        Plot the market states:
        - Chunk in sets of 2500 data points.
        - Highlight non-background states.
        - Store plots in a date-labeled folder.
        
        States are assigned fixed colors based on their integer value.
        For example, state 2 is always "green", state 5 is always "brown", etc.
        Adjust the mapping as needed.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from datetime import datetime

        today_str = datetime.now().strftime("%Y%m%d")
        out_folder = os.path.join(save_path, today_str)
        os.makedirs(out_folder, exist_ok=True)

        chunk_size = 4000
        unique_states, counts = np.unique(market_states, return_counts=True)
        # Define a fixed color mapping for states.
        # You can adjust these values as desired.
        predef_colors = {
            0: "blue",
            1: "orange",
            2: "green",
            3: "red",
            4: "purple",
            5: "brown",
            6: "pink",
            7: "gray",
            8: "olive",
            9: "cyan",
            10: "magenta",
            11: "lime",
            
        }
        # If a state is not in the dictionary, default to "black".
        state_to_color = {s: predef_colors.get(s, "black") for s in unique_states}
        
        # Optionally, if you wish to treat the most frequent state as background and not color it,
        # you can set its color to None (or "white"). For example:
        most_frequent_state = unique_states[np.argmax(counts)]
        state_to_color[most_frequent_state] = None

        def plot_chunk(ms_chunk, ts_chunk, pd_chunk, fname):
            plt.figure(figsize=(63, 27), dpi=600, facecolor="white")
            # Plot a dummy line for the background state if needed.
            if state_to_color[most_frequent_state] is None:
                plt.plot([], [], color="white", linewidth=5, label=f"State {most_frequent_state}")
            else:
                plt.plot([], [], color=state_to_color[most_frequent_state], linewidth=5, label=f"State {most_frequent_state}")
            plt.plot(ts_chunk, pd_chunk, color="black", linewidth=0.8, label="Price")

            used_labels = {most_frequent_state}
            start_idx = 0
            current_s = ms_chunk[0]
            for i in range(1, len(ms_chunk)):
                if ms_chunk[i] != current_s:
                    run_color = state_to_color[current_s]
                    if run_color is not None:
                        label = None
                        if current_s not in used_labels:
                            label = f"State {current_s}"
                            used_labels.add(current_s)
                        plt.axvspan(ts_chunk[start_idx], ts_chunk[i],
                                    color=run_color, alpha=1, label=label)
                    start_idx = i
                    current_s = ms_chunk[i]

            run_color = state_to_color[current_s]
            if run_color is not None:
                label = None
                if current_s not in used_labels:
                    label = f"State {current_s}"
                    used_labels.add(current_s)
                plt.axvspan(ts_chunk[start_idx], ts_chunk[-1],
                            color=run_color, alpha=1, label=label)

            plt.gca().set_facecolor("white")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.title(f"Market States (Chunked). Background = {most_frequent_state}")
            plt.legend(loc="upper left")
            plt.grid(True)
            plt.savefig(fname)
            plt.close()

        data_len = len(market_states)
        for start_idx in range(0, data_len, chunk_size):
            end_idx = min(start_idx + chunk_size, data_len)
            ms_c = market_states[start_idx:end_idx]
            ts_c = timestamps[start_idx:end_idx]
            pd_c = price_data[start_idx:end_idx]

            if len(ts_c) < 2:
                continue

            chunk_start_str = str(ts_c[0]).replace(":", "-").replace(" ", "_")
            chunk_end_str = str(ts_c[-1]).replace(":", "-").replace(" ", "_")
            fname = os.path.join(out_folder, f"states_{chunk_start_str}_to_{chunk_end_str}.png")
            plot_chunk(ms_c, ts_c, pd_c, fname)

        print(f"[PyroHMMAdvanced] All chunked plots saved in: {out_folder}")