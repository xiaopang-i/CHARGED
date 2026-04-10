# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/10 17:16
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/10 17:16

"""
Module for loading and processing electric vehicle dataset features.

This module provides comprehensive functionality for loading and preprocessing
electric vehicle charging data, including volume/duration features, pricing
information, weather data, and site metadata. It supports cross-validation
splits and DataLoader creation for deep learning training.

Key Features:
    - Loads and normalizes multiple data sources (volume/duration, pricing, weather)
    - Supports site selection strategies (top, middle, random)
    - Implements cross-validation splits by month
    - Creates PyTorch DataLoaders for training, validation, and testing
    - Handles auxiliary feature construction and normalization
"""

# Standard library imports
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader

# Local imports
from api.utils import CreateDataset


class EVDataset(object):
    """
    Dataset handler for EV (electric vehicle) features with auxiliary data.

    This class provides a comprehensive interface for loading, preprocessing,
    and managing electric vehicle charging data. It handles multiple data
    sources including main features (volume or duration), pricing information,
    weather data, and site metadata. The class supports various site
    selection strategies and provides methods for cross-validation splitting
    and DataLoader creation.

    Attributes:
        feature (str): Type of feature to load ('volume' or 'duration').
        auxiliary (str): Auxiliary feature modes ('None', 'all', or '+'-separated list).
        data_path (str): Root path to CSV files.
        feat (np.ndarray): Primary feature array [time, sites].
        extra_feat (np.ndarray): Auxiliary feature array [time, sites, features].
        time (pd.DatetimeIndex): Timestamps for each row in feat.
        scaler (StandardScaler): Scaler fitted on training data.
        train_feat, valid_feat, test_feat (np.ndarray): Splits of feat.
        train_extra_feat, valid_extra_feat, test_extra_feat (np.ndarray): Splits of extra_feat.
        train_loader, valid_loader, test_loader (DataLoader): PyTorch DataLoaders.
    """

    def __init__(
            self,
            feature: str,
            auxiliary: str,
            data_path: str,
            max_sites: int = 300,
            weather_columns: list[str] = ['temp', 'precip', 'visibility'],
            selection_mode: str = 'top',
    ) -> None:
        """
        Initialize EVDataset by loading and preprocessing data.

        This method performs the complete data loading and preprocessing pipeline:
        1. Loads main feature data (volume or duration)
        2. Loads and normalizes pricing data
        3. Loads and normalizes weather data
        4. Loads site metadata and applies selection strategy
        5. Constructs auxiliary features based on configuration
        6. Normalizes all data for machine learning

        Args:
            feature (str): Feature file to load; 'volume' or 'duration'.
            auxiliary (str): Auxiliary data mode: 'None', 'all', or combination of 'e_price', 's_price', weather keys.
            data_path (str): Directory path containing CSV files.
            max_sites (int): Maximum number of sites to select (default: 300).
            weather_columns (list[str]): Columns to load from weather data (default: ['temp', 'precip', 'visibility']).
            selection_mode (str): Site selection mode: 'top', 'middle', or 'random' (default: 'top').

        Raises:
            ValueError: If feature name or selection_mode is invalid.
        """
        super(EVDataset, self).__init__()
        self.feature = feature
        self.auxiliary = auxiliary
        self.data_path = data_path

        # Load main feature DataFrame based on feature type
        if self.feature == 'volume':
            self.feat = pd.read_csv(f'{self.data_path}volume.csv', header=0, index_col=0)
        elif self.feature == 'duration':
            self.feat = pd.read_csv(f'{self.data_path}duration.csv', header=0, index_col=0)
        else:
            raise ValueError("Unknown feature - must be 'volume' or 'duration'")

        # Load and normalize price data
        self.e_price = pd.read_csv(f'{self.data_path}e_price.csv', index_col=0, header=0).values  # Electricity price
        self.s_price = pd.read_csv(f'{self.data_path}s_price.csv', index_col=0, header=0).values  # Service price
        self.time = pd.to_datetime(self.feat.index)  # Convert index to datetime

        # Normalize price data to [0, 1] range using MinMaxScaler
        price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.e_price = price_scaler.fit_transform(self.e_price)  # Normalize electricity prices
        self.s_price = price_scaler.fit_transform(self.s_price)  # Normalize service prices

        # Load and normalize weather data
        self.weather = pd.read_csv(f'{self.data_path}weather.csv', header=0, index_col='time')
        self.weather = self.weather[weather_columns]  # Select specified weather columns
        
        # Apply domain-specific normalization for weather features
        if 'temp' in self.weather.columns:
            # Scale temperature from [-5, 40] to [0, 1]
            self.weather['temp'] = (self.weather['temp'] + 5) / 45
        if 'precip' in self.weather.columns:
            # Scale precipitation from [0, 120] to [0, 1]
            self.weather['precip'] = self.weather['precip'] / 120
        if 'visibility' in self.weather.columns:
            # Scale visibility from [0, 50] to [0, 1]
            self.weather['visibility'] = self.weather['visibility'] / 50

        # Load site metadata for selection and feature construction
        sites_info = pd.read_csv(f'{self.data_path}sites.csv', header=0)
        sites_info = sites_info.set_index("site")  # Set site_id as index
        sites_info.index = sites_info.index.astype(str)  # Convert index to string type

        # Select subset of sites if exceeding max_sites limit
        if len(sites_info) > max_sites:
            if selection_mode == 'top':
                # Select top sites by total_duration (most active)
                selected_sites = sites_info.sort_values(
                    by='total_duration', ascending=False
                ).head(max_sites)
            elif selection_mode == 'middle':
                # Select middle-range sites by total_duration
                sorted_sites = sites_info.sort_values(
                    by='total_duration', ascending=True
                )
                start = max((len(sorted_sites) - max_sites) // 2, 0)  # Calculate start index
                selected_sites = sorted_sites.iloc[start:start + max_sites]
            elif selection_mode == 'random':
                # Select random sample of sites
                selected_sites = sites_info.sample(
                    n=max_sites, random_state=42  # Fixed seed for reproducibility
                )
            else:
                raise ValueError(f"Unknown selection_mode: {selection_mode}")

            # Filter features and re-scale prices for selected sites
            selected_ids = selected_sites.index.tolist()  # Get selected site IDs
            self.feat = self.feat[selected_ids]  # Filter main features
            
            # Re-load and re-scale price data for selected sites
            e_price_df = pd.read_csv(f'{self.data_path}e_price.csv', index_col=0, header=0)
            s_price_df = pd.read_csv(f'{self.data_path}s_price.csv', index_col=0, header=0)
            self.e_price = price_scaler.fit_transform(e_price_df[selected_ids])  # Re-scale electricity prices
            self.s_price = price_scaler.fit_transform(s_price_df[selected_ids])  # Re-scale service prices

        # Normalize site latitude and longitude to [0, 1] range
        lat_long = sites_info.loc[self.feat.columns, ['latitude', 'longitude']].values
        lat_norm = (lat_long[:, 0] + 90) / 180  # Normalize latitude: [-90, 90] -> [0, 1]
        lon_norm = (lat_long[:, 1] + 180) / 360  # Normalize longitude: [-180, 180] -> [0, 1]
        self.lat_long_norm = np.stack([lat_norm, lon_norm], axis=1)  # Stack into 2D array
        
        # Create spatial features by repeating coordinates for each time step
        self.extra_feat = np.tile(
            self.lat_long_norm[np.newaxis, :, :], (self.feat.shape[0], 1, 1)
        )

        # Build auxiliary features if requested
        if self.auxiliary != 'None':
            # Initialize with zero channel for base features
            self.extra_feat = np.zeros([
                self.feat.shape[0], self.feat.shape[1], 1
            ])
            
            if self.auxiliary == 'all':
                # Add all auxiliary features: e_price, s_price, weather
                self.extra_feat = np.concatenate([
                    self.extra_feat, self.e_price[:, :, np.newaxis]  # Add electricity price
                ], axis=2)
                self.extra_feat = np.concatenate([
                    self.extra_feat, self.s_price[:, :, np.newaxis]  # Add service price
                ], axis=2)
                self.extra_feat = np.concatenate([
                    self.extra_feat,
                    np.repeat(
                        self.weather.values[:, np.newaxis, :],  # Repeat weather for all sites
                        self.feat.shape[1], axis=1
                    )
                ], axis=2)  # Add weather features
            else:
                # Add specified auxiliary features based on '+' separated list
                for add_feat in self.auxiliary.split('+'):
                    if add_feat == 'e_price':
                        # Add electricity price feature
                        self.extra_feat = np.concatenate([
                            self.extra_feat, self.e_price[:, :, np.newaxis]
                        ], axis=2)
                    elif add_feat == 's_price':
                        # Add service price feature
                        self.extra_feat = np.concatenate([
                            self.extra_feat, self.s_price[:, :, np.newaxis]
                        ], axis=2)
                    else:
                        # Add weather feature channel
                        self.extra_feat = np.concatenate([
                            self.extra_feat,
                            np.repeat(
                                self.weather[add_feat].values[:, np.newaxis, np.newaxis],
                                self.feat.shape[1], axis=1  # Repeat for all sites
                            )
                        ], axis=2)
            
            # Remove initial zero channel to get final auxiliary features
            self.extra_feat = self.extra_feat[:, :, 1:]

        # Convert main feature DataFrame to numpy array for consistency
        self.feat = np.array(self.feat)

    def split_cross_validation(
            self,
            fold: int,
            total_fold: int,
            train_ratio: float,
            valid_ratio: float,
    ) -> None:
        """
        Split data by month for cross-validation.

        This method implements a time-aware cross-validation strategy where
        data is split by months to preserve temporal dependencies. Each fold
        corresponds to a specific month, and within each fold, data is further
        split into training, validation, and test sets.

        Args:
            fold (int): Current fold index (1-based).
            total_fold (int): Total number of folds (should equal number of months).
            train_ratio (float): Fraction of fold data for training (e.g., 0.7).
            valid_ratio (float): Fraction of fold data for validation (e.g., 0.2).

        Raises:
            AssertionError: If time and feature lengths mismatch or train set empty.
        """
        # Validate data consistency
        assert len(self.time) == len(self.feat), "Time and feature lengths must match"
        month_list = sorted(np.unique(self.time.month))  # Get unique months
        assert total_fold == len(month_list), f"Total folds ({total_fold}) must equal number of months ({len(month_list)})"

        # Calculate split points based on month membership
        fold_time = self.time.month.isin(month_list[0:fold]).sum()  # Count timesteps in current fold
        train_end = int(fold_time * train_ratio)  # Training set end index
        valid_start = train_end  # Validation set start index
        valid_end = int(valid_start + fold_time * valid_ratio)  # Validation set end index
        test_start = valid_end  # Test set start index
        test_end = int(fold_time)  # Test set end index

        # Slice numpy arrays for each split
        train_feat = self.feat[:train_end]  # Training features
        valid_feat = self.feat[valid_start:valid_end]  # Validation features
        test_feat = self.feat[test_start:test_end]  # Test features

        # Apply standard scaling to normalize features
        self.scaler = StandardScaler()  # Initialize scaler
        self.train_feat = self.scaler.fit_transform(train_feat)  # Fit and transform training data
        self.valid_feat = self.scaler.transform(valid_feat)  # Transform validation data
        self.test_feat = self.scaler.transform(test_feat)  # Transform test data

        # Initialize auxiliary feature splits
        self.train_extra_feat, self.valid_extra_feat, self.test_extra_feat = None, None, None
        
        # Split auxiliary features if present
        if self.extra_feat is not None:
            self.train_extra_feat = self.extra_feat[:train_end]  # Training auxiliary features
            self.valid_extra_feat = self.extra_feat[valid_start:valid_end]  # Validation auxiliary features
            self.test_extra_feat = self.extra_feat[test_start:test_end]  # Test auxiliary features

        # Validate that training set is not empty
        assert len(train_feat) > 0, "The training set cannot be empty!"

    def create_loaders(
            self,
            seq_l: int,
            pre_len: int,
            batch_size: int,
            device: torch.device,
    ) -> None:
        """
        Create PyTorch DataLoaders for training, validation, and testing.

        This method creates PyTorch DataLoader objects that handle batching,
        shuffling, and device placement for efficient training. Training data
        is shuffled and uses the specified batch size, while validation and
        test data use full-batch loading for consistent evaluation.

        Args:
            seq_l (int): Input sequence length for sliding window sampling.
            pre_len (int): Prediction horizon length.
            batch_size (int): Batch size for training loader.
            device (torch.device): Device for tensor allocation (CPU/GPU).
        """
        # Create training dataset and loader with shuffling
        train_dataset = CreateDataset(
            seq_l, pre_len, self.train_feat, self.train_extra_feat, device
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True  # Shuffle for training
        )

        # Create validation dataset and loader (no shuffling)
        valid_dataset = CreateDataset(
            seq_l, pre_len, self.valid_feat, self.valid_extra_feat, device
        )
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=len(self.valid_feat), shuffle=False  # Full batch, no shuffle
        )

        # Create test dataset and loader (no shuffling)
        test_dataset = CreateDataset(
            seq_l, pre_len, self.test_feat, self.test_extra_feat, device
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=len(self.test_feat), shuffle=False  # Full batch, no shuffle
        )
