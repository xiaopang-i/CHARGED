# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/10 16:20
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/10 16:20

"""
Utility module providing logging, reproducibility, dataset creation, and evaluation metrics for EV demand prediction.

This module contains essential utilities for the WEVC (Wireless Electric Vehicle Charging) project,
including logging functionality, random seed management, dataset creation, and comprehensive
regression metrics calculation for time series forecasting evaluation.

Functions:
    Logger: Redirects stdout to both terminal and log file with timestamps.
    random_seed: Sets random seeds for Python, NumPy, and Torch for reproducibility.
    get_n_feature: Computes input feature count including auxiliary features.
    CreateDataset: PyTorch Dataset for sequence-to-one prediction, handling auxiliary features.
    create_rnn_data: Generates sliding-window samples and labels from time-series data.
    calculate_regression_metrics: Computes regression metrics (MAE, RMSE, MAPE, RAE, MedAE, R², EVS).
    convert_numpy: Recursively converts NumPy types in structures to native Python types.
    get_data_paths: Constructs per-city data directory paths.
"""

# Standard library imports
import datetime
import os
import random
import sys
from typing import Any, Optional, Tuple, Dict, List

# Third-party imports
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
    mean_absolute_percentage_error,
)


class Logger(object):
    """
    Logs stdout messages to both console and file with timestamps.
    
    This class provides a dual-output logging mechanism that captures all stdout
    messages and writes them to both the terminal and a specified log file with
    automatic timestamping for non-empty messages.

    Args:
        filename (str): Path to the log file where messages will be stored.
    """

    def __init__(self, filename: str = "log.txt") -> None:
        """Initialize the logger with terminal and file output streams."""
        self.terminal = sys.stdout  # Original stdout for console output
        self.log = open(filename, "w", encoding="utf-8")  # File stream for logging

    def write(self, message: str) -> None:
        """
        Write message to console and log file, prepending a timestamp if non-empty.
        
        This method intercepts all stdout writes and duplicates them to both
        the terminal and log file. Non-empty messages are prefixed with timestamps
        for better traceability.

        Args:
            message (str): Message to log and display.
        """
        if message.strip():  # Only add timestamp for non-empty messages
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"{timestamp} - {message}"
        self.terminal.write(message)  # Write to console
        self.log.write(message)       # Write to log file

    def flush(self) -> None:
        """
        Flush both console and file streams to ensure immediate output.
        
        This method ensures that buffered content is immediately written
        to both the terminal and log file.
        """
        self.terminal.flush()  # Flush console output
        self.log.flush()       # Flush file output


def random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across torch, NumPy, and Python.
    
    This function ensures deterministic behavior across all random number
    generators used in the project, including PyTorch, NumPy, and Python's
    built-in random module. It also configures CUDA for deterministic operations.

    Args:
        seed (int): Seed value for all random number generators.
    """
    torch.manual_seed(seed)           # Set PyTorch CPU seed
    torch.cuda.manual_seed(seed)      # Set PyTorch GPU seed
    np.random.seed(seed)              # Set NumPy random seed
    random.seed(seed)                 # Set Python random seed
    torch.backends.cudnn.deterministic = True   # Ensure deterministic CUDA operations
    torch.backends.cudnn.benchmark = False      # Disable CUDA benchmarking for reproducibility
    torch.backends.cudnn.enabled = False        # Disable cuDNN for deterministic behavior


def get_n_feature(extra_feat: Optional[np.ndarray]) -> int:
    """
    Determine the number of input feature channels including auxiliary features.
    
    This function calculates the total number of feature channels that will be
    used as input to the model. It accounts for both the base feature (typically
    the target variable) and any auxiliary features (e.g., weather, POI data).

    Args:
        extra_feat (np.ndarray or None): Auxiliary features array of shape [T, N, C] or None.

    Returns:
        int: Total feature channels (1 base + C auxiliary, or 1 if no auxiliary features).
    """
    return 1 if extra_feat is None else extra_feat.shape[-1] + 1


class CreateDataset(Dataset):
    """
    PyTorch Dataset for sliding-window RNN data with auxiliary features support.
    
    This dataset class converts time series features and optional auxiliary data
    into sequence-label pairs suitable for recurrent neural network training.
    It implements sliding window sampling and handles device placement for tensors.

    Args:
        seq_l (int): Length of input sequence (lookback window).
        pre_len (int): Prediction horizon (steps ahead to predict).
        feat (np.ndarray): Main feature series of shape [T, N].
        extra_feat (np.ndarray or None): Auxiliary feature series of shape [T, N, C].
        device (torch.device): Target device for tensor placement (CPU/GPU).
    """

    def __init__(
            self,
            seq_l: int,
            pre_len: int,
            feat: np.ndarray,
            extra_feat: Optional[np.ndarray],
            device: torch.device,
    ) -> None:
        """Initialize the dataset with features and create sliding window samples."""
        # Create sliding window sequences and labels from main features
        x, y = create_rnn_data(feat, seq_l, pre_len)
        self.feat = torch.Tensor(x)      # Convert to PyTorch tensor
        self.label = torch.Tensor(y)     # Convert labels to tensor
        self.extra_feat = None           # Initialize auxiliary features as None
        
        # Process auxiliary features if provided
        if extra_feat is not None:
            x2, _ = create_rnn_data(extra_feat, seq_l, pre_len)  # Create auxiliary sequences
            self.extra_feat = torch.Tensor(x2)  # Convert to tensor
        self.device = device  # Store target device

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.feat)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample and label, transposed to [N, seq_l] format.
        
        This method returns a single training sample with its corresponding label
        and auxiliary features. The feature tensors are transposed to match the
        expected input format for RNN models.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            feat (torch.Tensor): Input features of shape [N, seq_l].
            label (torch.Tensor): Target label tensor.
            extra_feat (torch.Tensor): Auxiliary features of shape [N, seq_l] or empty tensor.
        """
        # Transpose features to [N, seq_l] format and move to device
        feat = self.feat[idx].transpose(0, 1).to(self.device)
        label = self.label[idx].to(self.device)  # Move label to device
        
        # Handle auxiliary features if available
        if self.extra_feat is not None:
            ef = self.extra_feat[idx].transpose(0, 1).to(self.device)  # Transpose and move to device
            return feat, label, ef
        # Return empty tensor if no auxiliary features
        dummy = torch.empty(0, device=self.device)
        return feat, label, dummy


def create_rnn_data(
        dataset: np.ndarray,
        lookback: int,
        predict_time: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window sequences and corresponding labels from time series data.
    
    This function implements sliding window sampling for time series forecasting.
    It creates input sequences of length 'lookback' and corresponding target
    values that are 'predict_time' steps ahead of the sequence end.

    Args:
        dataset (np.ndarray): Time series data of shape [T, ...].
        lookback (int): Number of past time steps to include in each sequence.
        predict_time (int): Number of steps ahead to predict (prediction horizon).

    Returns:
        x (np.ndarray): Input sequences of shape [num_samples, lookback, ...].
        y (np.ndarray): Target values of shape [num_samples, ...] at position lookback+predict_time-1.
    """
    x, y = [], []  # Initialize lists to store sequences and labels
    L = len(dataset)  # Total length of the dataset
    
    # Create sliding window samples
    for i in range(L - lookback - predict_time):
        x.append(dataset[i: i + lookback])  # Input sequence from i to i+lookback
        y.append(dataset[i + lookback + predict_time - 1])  # Target at i+lookback+predict_time-1
    
    return np.array(x), np.array(y)  # Convert lists to numpy arrays


def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        optimized: bool = True,
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics for time series forecasting evaluation.
    
    This function calculates multiple regression metrics to evaluate the performance
    of forecasting models. It includes both standard metrics and optimized versions
    that handle edge cases like zero values more robustly.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        optimized (bool): If True, applies small-epsilon adjustments to handle zero values.

    Returns:
        dict: Dictionary containing all computed metrics:
            - MAE: Mean Absolute Error
            - RMSE: Root Mean Square Error
            - MAPE: Mean Absolute Percentage Error
            - RAE: Relative Absolute Error
            - MedAE: Median Absolute Error
            - R²: Coefficient of Determination
            - EVS: Explained Variance Score
    """
    eps = 1e-6  # Small epsilon to prevent division by zero
    
    if optimized:
        # Create copies to avoid modifying original arrays
        yt = y_true.copy()
        yp = y_pred.copy()
        # Handle zero values by adding small epsilon
        yt[yt <= eps] = np.abs(yt[yt <= eps]) + eps
        yp[yt <= eps] = np.abs(yp[yt <= eps]) + eps
        # Calculate metrics with optimized handling
        mape = mean_absolute_percentage_error(yt, yp)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + eps)
        evs = 1 - np.var(y_true - y_pred) / (np.var(y_true) + eps)
    else:
        # Use standard metric calculations
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        evs = explained_variance_score(y_true, y_pred)
    
    # Calculate standard metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate Relative Absolute Error with zero handling
    if np.sum(np.abs(y_true - np.mean(y_true))) == 0:
        rae = np.sum(np.abs(y_true - y_pred)) / eps
    else:
        rae = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))
    
    medae = median_absolute_error(y_true, y_pred)
    
    # Return all metrics in a dictionary
    return {
        'MAE'  : mae,    # Mean Absolute Error
        'RMSE' : rmse,   # Root Mean Square Error
        'MAPE' : mape,   # Mean Absolute Percentage Error
        'RAE'  : rae,    # Relative Absolute Error
        'MedAE': medae,  # Median Absolute Error
        'R²'   : r2,     # Coefficient of Determination
        'EVS'  : evs,    # Explained Variance Score
    }


def convert_numpy(obj: Any) -> Any:
    """
    Recursively convert NumPy scalar types in a structure to Python native types.
    
    This function is useful for JSON serialization and other operations that
    require native Python types instead of NumPy types. It handles nested
    structures like dictionaries and lists.

    Args:
        obj: Object to convert (np.generic, dict, list, or other types).

    Returns:
        Object with NumPy types converted to native Python types.
    """
    if isinstance(obj, np.generic):  # Handle NumPy scalar types
        return obj.item()
    if isinstance(obj, dict):  # Handle dictionaries recursively
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):  # Handle lists recursively
        return [convert_numpy(i) for i in obj]
    return obj  # Return unchanged for other types


def get_data_paths(
        ori_path: str,
        cities: str,
        suffix: str = '_remove_zero',
) -> Dict[str, str]:
    """
    Generate data directory paths for multiple cities.
    
    This function constructs the full directory paths for each city's data
    based on the base path, city codes, and optional suffix. It supports
    multi-city configurations separated by '+' characters.

    Args:
        ori_path (str): Base directory path containing city data folders.
        cities (str): City codes separated by '+' (e.g., 'AMS+JHB+LOA').
        suffix (str): Directory suffix for each city (default '_remove_zero').

    Returns:
        dict: Mapping from city code to full data directory path.
    """
    paths = {}  # Initialize dictionary to store city-path mappings
    
    # Process each city code
    for city in cities.split('+'):  # Split by '+' to handle multiple cities
        paths[city] = os.path.join(ori_path, f"{city}{suffix}/")  # Construct full path
    
    return paths
