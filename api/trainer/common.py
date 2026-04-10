# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/11 2:56
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/11 2:56

"""
Module defining PredictionTrainer for training and evaluating prediction models.

This module provides a comprehensive training framework for both statistical
and neural models for electric vehicle charging demand prediction. It handles
training loops, validation, model checkpointing, and evaluation with multiple
regression metrics.

Key Features:
    - Supports both statistical models (LO, AR, ARIMA) and neural models
    - Implements training and validation loops with early stopping
    - Provides model checkpointing and loading
    - Computes comprehensive regression metrics
    - Saves predictions, labels, and evaluation results

Classes:
    PredictionTrainer: Manages training loops, validation, testing, and metric logging.
"""

# Standard library imports
import json
import os
from typing import Optional, Any

# Third-party imports
import numpy as np
import torch
from tqdm import tqdm

# Local imports
from api.utils import calculate_regression_metrics, convert_numpy


class PredictionTrainer(object):
    """
    Trainer class for prediction models on EV charging demand data.

    This class provides a unified interface for training and evaluating both
    statistical and neural models. It handles the complete training pipeline
    including data loading, model training, validation, checkpointing, and
    evaluation with comprehensive metrics.

    The trainer automatically detects model type and configures appropriate
    training strategies. Statistical models use direct prediction methods,
    while neural models use gradient-based optimization with validation.

    Attributes:
        ev_dataset (Any): Dataset object containing DataLoaders or raw arrays.
        ev_model (Any): PredictionModel instance with `.model` and `.model_name`.
        is_train (bool): Flag indicating training mode.
        stat_model (bool): True if using statistical model (no optimizer).
        optim (Optional[torch.optim.Optimizer]): Optimizer for neural models.
        loss_func (Optional[torch.nn.Module]): Loss function for neural models.
        save_path (str): Directory to save checkpoints and outputs.
        test_loader (Any): Raw arrays or DataLoader for test data.
        train_valid_feat (Optional[np.ndarray]): Combined data for statistical models.
    """

    def __init__(
            self,
            dataset: Any,
            model: Any,
            seq_l: int,
            pre_len: int,
            is_train: bool,
            save_path: str,
    ) -> None:
        """
        Initialize PredictionTrainer with dataset, model, and configuration.

        This method sets up the training environment based on the model type.
        For statistical models, it prepares the data arrays and disables training.
        For neural models, it initializes the optimizer and loss function.

        Args:
            dataset (Any): EVDataset or similar with train_loader, valid_loader, test_loader.
            model (Any): PredictionModel containing `.model` and `.model_name`.
            seq_l (int): Input sequence length for statistical models.
            pre_len (int): Prediction horizon length for statistical models.
            is_train (bool): Flag to enable training mode.
            save_path (str): Directory path to save model and results.
        """
        self.ev_dataset = dataset
        self.ev_model = model
        self.save_path = save_path
        self.is_train = is_train

        # Configure training strategy based on model type
        if model.model_name in ('lo', 'ar', 'arima'):
            # Statistical models: no optimizer or loss function needed
            self.optim = None
            self.loss_func = None
            self.is_train = False
            self.stat_model = True
            
            # Prepare combined training and validation data for statistical prediction
            self.train_valid_feat = np.vstack(
                (dataset.train_feat, dataset.valid_feat,
                 dataset.test_feat[: seq_l + pre_len, :])  # Include initial test data for context
            )
            # Test features and labels for statistical prediction
            self.test_loader = [
                self.train_valid_feat,  # Input features (train + valid + initial test)
                dataset.test_feat[pre_len + seq_l:, :]  # Target labels (remaining test data)
            ]
        else:
            # Neural models: set up optimizer and loss function
            self.optim = torch.optim.Adam(
                model.model.parameters(), weight_decay=1e-5  # L2 regularization
            )
            self.stat_model = False
            self.loss_func = torch.nn.MSELoss()  # Mean squared error loss

    def training(self, epoch: int) -> None:
        """
        Train and validate neural model over multiple epochs.

        This method implements the complete training loop for neural models,
        including training epochs, validation, and model checkpointing based
        on validation loss. It saves the best model based on validation performance.

        Args:
            epoch (int): Number of training epochs to run.
        """
        best_val_loss = float('inf')  # Track best validation loss for model saving
        self.ev_model.model.train()  # Set model to training mode

        # Training loop over specified epochs
        for _ in tqdm(range(epoch), desc='Training'):
            # Training phase
            for feat, label, extra in self.ev_dataset.train_loader:
                torch.cuda.empty_cache()
                feat = feat.to(self.ev_model.model.linear.weight.device if hasattr(self.ev_model.model, 'linear') else next(self.ev_model.model.parameters()).device, non_blocking=True)
                label = label.to(next(self.ev_model.model.parameters()).device, non_blocking=True)

                if self.ev_dataset.extra_feat is None:
                    extra = None
                else:
                    extra = extra.to(next(self.ev_model.model.parameters()).device, non_blocking=True)
                self.optim.zero_grad()
                preds = self.ev_model.model(feat, extra)                
                # Align shapes for loss computation
                if preds.shape != label.shape:
                    loss = self.loss_func(preds.unsqueeze(-1), label)  # Add dimension if needed
                else:
                    loss = self.loss_func(preds, label)  # Direct loss computation
                
                # Backward pass and optimization
                loss.backward()  # Compute gradients
                self.optim.step()  # Update model parameters

            # Validation phase
            for feat, label, extra in self.ev_dataset.valid_loader:
                torch.cuda.empty_cache()  # Clear GPU memory
                if self.ev_dataset.extra_feat is None:
                    extra = None  # Handle case with no auxiliary features
                
                # Forward pass for validation
                preds = self.ev_model.model(feat, extra)  # Model prediction
                if preds.shape != label.shape:
                    loss = self.loss_func(preds.unsqueeze(-1), label)  # Add dimension if needed
                else:
                    loss = self.loss_func(preds, label)  # Direct loss computation
                
                val_loss = loss.item()  # Get scalar loss value
                
                # Save best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss  # Update best loss
                    os.makedirs(self.save_path, exist_ok=True)  # Create save directory
                    torch.save(
                        self.ev_model.model.state_dict(),  # Save model state
                        os.path.join(self.save_path, 'train.pth')  # Save path
                    )

    def test(self, model_path: Optional[str] = None) -> None:
        """
        Evaluate model on test data and save predictions, labels, and metrics.

        This method performs comprehensive evaluation of the trained model on
        test data. It handles both neural and statistical models, computes
        multiple regression metrics, and saves all results for analysis.

        For neural models, it loads saved checkpoint if provided.
        For statistical models, it runs direct prediction on raw arrays.

        Args:
            model_path (str, optional): Path to model weights for neural models.
        """
        preds_list, labels_list = [], []  # Lists to collect predictions and labels

        if not self.stat_model:
            # Neural model evaluation
            if model_path:
                self.ev_model.load_model(model_path)  # Load saved model weights
            self.ev_model.model.eval()  # Set model to evaluation mode

            # Iterate through test data
            for feat, label, extra in self.ev_dataset.test_loader:
                torch.cuda.empty_cache()  # Clear GPU memory
                if self.ev_dataset.extra_feat is None:
                    extra = None  # Handle case with no auxiliary features
                
                # Forward pass without gradient computation
                with torch.no_grad():
                    preds = self.ev_model.model(feat, extra)  # Model prediction
                    preds = preds.unsqueeze(-1) if preds.shape != label.shape else preds  # Align shapes
                    preds = preds.cpu().numpy()  # Convert to numpy array
                
                labels = label.cpu().numpy()  # Convert labels to numpy array
                preds_list.append(preds)  # Collect predictions
                labels_list.append(labels)  # Collect labels
        else:
            # Statistical model prediction
            train_valid, test_feats = self.test_loader  # Unpack test data
            preds = self.ev_model.model.predict(train_valid, test_feats)  # Direct prediction
            preds_list.append(preds)  # Collect predictions
            labels_list.append(test_feats)  # Collect labels

        # Concatenate all predictions and labels
        pred_array = np.concatenate(preds_list, axis=0)  # Combine all predictions
        label_array = np.concatenate(labels_list, axis=0)  # Combine all labels
        
        # Inverse transform if scaler is available
        if self.ev_dataset.scaler:
            pred_array = self.ev_dataset.scaler.inverse_transform(pred_array)  # Denormalize predictions
            label_array = self.ev_dataset.scaler.inverse_transform(label_array)  # Denormalize labels

        # Save outputs to files
        os.makedirs(self.save_path, exist_ok=True)  # Create save directory
        np.save(os.path.join(self.save_path, 'predict.npy'), pred_array)  # Save predictions
        np.save(os.path.join(self.save_path, 'label.npy'), label_array)  # Save labels

        # Compute comprehensive regression metrics
        metrics = calculate_regression_metrics(y_true=label_array, y_pred=pred_array)  # Calculate metrics
        metrics = convert_numpy(metrics)  # Convert numpy types to native Python types
        
        # Save metrics to JSON file
        with open(
                os.path.join(self.save_path, 'metrics.json'), 'w', encoding='utf-8'
        ) as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)  # Save metrics with formatting
