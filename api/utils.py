# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/10 16:20
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/10 16:20

"""
Utility module providing logging, reproducibility, dataset creation, and evaluation metrics for EV demand prediction.
"""

import datetime
import os
import random
import sys
from typing import Any, Optional, Tuple, Dict

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
    def __init__(self, filename: str = "log.txt") -> None:
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        if message.strip():
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"{timestamp} - {message}"
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


def random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 为了性能，不再强制关闭 cuDNN
    # 你之前那套 deterministic + benchmark=False + enabled=False，
    # 会牺牲性能，尤其不利于你追求更高 GPU 利用率。
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def get_n_feature(extra_feat: Optional[np.ndarray]) -> int:
    return 1 if extra_feat is None else extra_feat.shape[-1] + 1


class CreateDataset(Dataset):
    """
    PyTorch Dataset for sliding-window RNN data with auxiliary features support.

    关键修改：
    1. 不在 __getitem__ 里做 .to(device)
    2. 不在 sample 粒度搬运到 GPU
    3. 只返回 CPU Tensor，训练循环里按 batch 搬运
    """

    def __init__(
        self,
        seq_l: int,
        pre_len: int,
        feat: np.ndarray,
        extra_feat: Optional[np.ndarray],
        device: torch.device,  # 保留参数，为兼容旧调用；当前内部不使用
    ) -> None:
        x, y = create_rnn_data(feat, seq_l, pre_len)
        self.feat = torch.tensor(x, dtype=torch.float32)
        self.label = torch.tensor(y, dtype=torch.float32)
        self.extra_feat = None

        if extra_feat is not None:
            x2, _ = create_rnn_data(extra_feat, seq_l, pre_len)
            self.extra_feat = torch.tensor(x2, dtype=torch.float32)

        self.device = device

    def __len__(self) -> int:
        return len(self.feat)

    def __getitem__(self, idx: int):
        feat = self.feat[idx].transpose(0, 1)   # [node, seq]
        label = self.label[idx]

        if self.extra_feat is not None:
            ef = self.extra_feat[idx].transpose(0, 1)  # [node, seq, channel] 或相关结构
            return feat, label, ef

        dummy = torch.empty(0, dtype=torch.float32)
        return feat, label, dummy


def create_rnn_data(
    dataset: np.ndarray,
    lookback: int,
    predict_time: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    L = len(dataset)

    for i in range(L - lookback - predict_time):
        x.append(dataset[i: i + lookback])
        y.append(dataset[i + lookback + predict_time - 1])

    return np.array(x), np.array(y)


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    optimized: bool = True,
) -> Dict[str, float]:
    eps = 1e-6

    if optimized:
        yt = y_true.copy()
        yp = y_pred.copy()
        mask = yt <= eps
        yt[mask] = np.abs(yt[mask]) + eps
        yp[mask] = np.abs(yp[mask]) + eps
        mape = mean_absolute_percentage_error(yt, yp)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + eps)
        evs = 1 - np.var(y_true - y_pred) / (np.var(y_true) + eps)
    else:
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        evs = explained_variance_score(y_true, y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    denom = np.sum(np.abs(y_true - np.mean(y_true)))
    if denom == 0:
        rae = np.sum(np.abs(y_true - y_pred)) / eps
    else:
        rae = np.sum(np.abs(y_true - y_pred)) / denom

    medae = median_absolute_error(y_true, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'RAE': rae,
        'MedAE': medae,
        'R²': r2,
        'EVS': evs,
    }


def convert_numpy(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj


def get_data_paths(
    ori_path: str,
    cities: str,
    suffix: str = '_remove_zero',
) -> Dict[str, str]:
    paths = {}
    for city in cities.split('+'):
        paths[city] = os.path.join(ori_path, f"{city}{suffix}/")
    return paths