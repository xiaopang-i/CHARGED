# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/11 2:03
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/11 2:03

"""
Module defining the PredictionModel factory for instantiating various time-series prediction architectures.

This module provides a unified interface for creating and managing different time-series
prediction models. It supports a wide range of architectures including traditional
statistical models, neural networks, and modern transformer-based approaches.

Supported Models:
    - Traditional: LSTM, AR, ARIMA, LO (Last Observation)
    - Neural Networks: FCNN (Fully Connected Neural Network)
    - Modern Architectures: SegRNN, FreTS, ModernTCN, MultiPatchFormer, ConvTimeNet

Each model is initialized with appropriate parameters including sequence length,
feature dimensions, number of nodes, and prediction horizon.
"""

# Standard library imports
from typing import Any

# Third-party imports
import torch

# Local imports
from api.model.modules import Lstm, Lo, Ar, Arima, Fcnn, SegRNN, FreTS, ModernTCN, MultiPatchFormer, ConvTimeNet


class PredictionModel(object):
    """
    Factory and loader for prediction models.

    This class provides a unified interface for selecting and initializing various
    time-series prediction model architectures. It acts as a factory that creates
    the appropriate model based on the specified name and parameters.

    The class supports both traditional statistical models and modern deep learning
    architectures, providing a consistent interface for model creation, loading,
    and configuration management.

    Attributes:
        model_name (str): Identifier of the model architecture.
        model (torch.nn.Module): Instantiated model ready for training or inference.
    """

    def __init__(
            self,
            num_node: int,
            n_fea: int,
            model_name: str,
            seq_l: int,
            pre_len: int,
    ) -> None:
        """
        Initialize a PredictionModel instance by selecting the appropriate architecture.

        This method creates the specified model architecture with the given parameters.
        It supports multiple model types ranging from simple statistical models to
        complex deep learning architectures. Each model is initialized with the
        appropriate parameters for time-series prediction.

        Args:
            num_node (int): Number of nodes or output dimensions (for graph-based models).
            n_fea (int): Number of input feature channels.
            model_name (str): Name of the model to instantiate ('lstm', 'lo', 'ar', etc.).
            seq_l (int): Input sequence length for time-series prediction.
            pre_len (int): Prediction horizon length (number of future time steps).

        Raises:
            ValueError: If the specified model_name is unsupported.
        """
        self.model_name = model_name  # Store model name for reference

        # Initialize model based on specified architecture
        if model_name == 'lstm':
            # Long Short-Term Memory network for sequence modeling
            self.model = Lstm(seq=seq_l, n_fea=n_fea, node=num_node)
        elif model_name == 'lo':
            # Last Observation: simple baseline using last value
            self.model = Lo(pred_len=pre_len)
        elif model_name == 'ar':
            # Autoregressive model with specified lag order
            self.model = Ar(pred_len=pre_len, lags=seq_l)
        elif model_name == 'arima':
            # ARIMA model for time series forecasting
            self.model = Arima(pred_len=pre_len)
        elif model_name == 'fcnn':
            # Fully Connected Neural Network
            self.model = Fcnn(n_fea, node=num_node, seq=seq_l)
        elif model_name == 'segrnn':
            # Segmented Recurrent Neural Network
            self.model = SegRNN(seq_len=seq_l, pred_len=pre_len, n_fea=n_fea)
        elif model_name == 'frets':
            # Frequency-based Time Series model
            self.model = FreTS(seq_len=seq_l, pred_len=pre_len, n_fea=n_fea)
        elif model_name == 'moderntcn':
            # Modern Temporal Convolutional Network
            self.model = ModernTCN(seq_len=seq_l, n_fea=n_fea, pred_len=pre_len)
        elif model_name == 'multipatchformer':
            # Multi-Patch Transformer for time series
            self.model = MultiPatchFormer(seq_len=seq_l, n_fea=n_fea, pred_len=pre_len)
        elif model_name == 'convtimenet':
            # Convolutional Time Network
            self.model = ConvTimeNet(seq_len=seq_l, c_in=n_fea, c_out=pre_len)
        else:
            # Raise error for unsupported model names
            raise ValueError(f"Unsupported model_name: {model_name}")

        # Set default chunk size for processing large sequences in batches
        # This helps manage memory usage for long sequences
        self.model.chunk_size = 512

    def load_model(self, model_path: str) -> None:
        """
        Load pretrained model weights from a checkpoint file.

        This method loads previously saved model parameters from a file and
        applies them to the current model instance. It uses torch's load_state_dict
        for parameter restoration.

        Args:
            model_path (str): Filesystem path to the saved state dictionary file.
        """
        state_dict = torch.load(model_path, weights_only=True)  # Load state dictionary
        self.model.load_state_dict(state_dict)  # Apply loaded parameters to model

    def update_chunksize(self, chunk_size: int) -> None:
        """
        Update the model's internal chunk size parameter.

        This method allows dynamic adjustment of the chunk size used for processing
        long sequences. A larger chunk size may improve throughput but increase
        memory usage, while a smaller chunk size reduces memory usage but may
        decrease processing speed.

        Args:
            chunk_size (int): New chunk size for sequence processing.
        """
        self.model.chunk_size = chunk_size  # Update chunk size parameter
