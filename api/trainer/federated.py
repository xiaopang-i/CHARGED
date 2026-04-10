# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/15 1:33
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/15 1:33

"""
Module defining ClientTrainer for per-client training and evaluation in federated learning.
"""

import json
import os
from typing import Optional, Any

import numpy as np
import torch
from tqdm import tqdm

from api.utils import calculate_regression_metrics, convert_numpy


class ClientTrainer(object):
    def __init__(
        self,
        train_loader,
        test_loader,
        extra_feat_tag: bool,
        model: Any,
        save_path: str,
        scaler,
        device: torch.device,
    ) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.extra_feat_tag = extra_feat_tag
        self.ev_model = model
        self.save_path = save_path
        self.device = device
        self.scaler = scaler
        self.now_epoch = 0
        self.deploy_epoch = 0

        if model.model_name in ('lo', 'ar', 'arima'):
            raise ValueError("ClientTrainer is not applicable to statistical models.")

        self.optim = torch.optim.Adam(
            model.model.parameters(),
            weight_decay=1e-5
        )
        self.loss_func = torch.nn.MSELoss()

    def training(
        self,
        epoch: int,
        save_model: bool = False
    ) -> None:
        self.ev_model.model.to(self.device)
        self.ev_model.model.train()

        for _ in tqdm(range(epoch), desc='Training'):
            for feat, label, extra_feat in self.train_loader:
                feat = feat.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)

                if self.extra_feat_tag:
                    extra_feat = extra_feat.to(self.device, non_blocking=True)
                else:
                    extra_feat = None

                self.optim.zero_grad(set_to_none=True)
                preds = self.ev_model.model(feat, extra_feat)

                if preds.shape != label.shape:
                    loss = self.loss_func(preds.unsqueeze(-1), label)
                else:
                    loss = self.loss_func(preds, label)

                loss.backward()
                self.optim.step()

        if save_model:
            os.makedirs(self.save_path, exist_ok=True)
            path = os.path.join(
                self.save_path,
                f'train_{self.now_epoch}_{self.deploy_epoch}.pth'
            )
            torch.save(self.ev_model.model.state_dict(), path)

    def test(
        self,
        model_path: Optional[str] = None
    ) -> None:
        preds_list, labels_list = [], []

        self.ev_model.model.to(self.device)

        if model_path:
            self.ev_model.load_model(model_path=model_path)
        self.ev_model.model.eval()

        with torch.no_grad():
            for feat, label, extra_feat in self.test_loader:
                feat = feat.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)

                if self.extra_feat_tag:
                    extra_feat = extra_feat.to(self.device, non_blocking=True)
                else:
                    extra_feat = None

                preds = self.ev_model.model(feat, extra_feat)
                if preds.shape != label.shape:
                    preds = preds.unsqueeze(-1)

                preds = preds.detach().cpu().numpy()
                labels = label.detach().cpu().numpy()

                preds_list.append(preds)
                labels_list.append(labels)

        pred_array = np.concatenate(preds_list, axis=0)
        label_array = np.concatenate(labels_list, axis=0)

        if self.scaler is not None:
            pred_array = pred_array * self.scaler
            label_array = label_array * self.scaler

        os.makedirs(self.save_path, exist_ok=True)
        np.save(
            os.path.join(
                self.save_path,
                f'predict_{self.now_epoch}_{self.deploy_epoch}.npy'
            ),
            pred_array
        )
        np.save(
            os.path.join(
                self.save_path,
                f'label_{self.now_epoch}_{self.deploy_epoch}.npy'
            ),
            label_array
        )

        metrics = calculate_regression_metrics(y_true=label_array, y_pred=pred_array)
        metrics = convert_numpy(metrics)
        with open(
            os.path.join(
                self.save_path,
                f'metrics_{self.now_epoch}_{self.deploy_epoch}.json'
            ),
            'w',
            encoding='utf-8'
        ) as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)