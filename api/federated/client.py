# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/13 18:55
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/13 18:55

"""
Module defining the CommonClient class for federated learning client-side operations.
"""

from typing import Dict, Any, Callable, Optional, Generator
import os
import torch
from torch.utils.data import DataLoader, Subset

from api.utils import CreateDataset


class CommonClient(object):
    def __init__(
        self,
        client_id: str,
        data_dict: Dict[str, Any],
        scaler: Any,
        model_module: Callable[..., Any],
        trainer_module: Callable[..., Any],
        seq_l: int,
        pre_len: int,
        model_name: str,
        n_fea: int,
        batch_size: int,
        device: torch.device,
        save_path: str,
        support_rate: float = 1.0,
    ) -> None:
        super(CommonClient, self).__init__()

        self.client_id = client_id
        self.scaler = scaler
        self.feat = data_dict['feat']
        self.extra_feat = data_dict['extra_feat']

        model = model_module(
            num_node=1,
            n_fea=n_fea,
            model_name=model_name,
            seq_l=seq_l,
            pre_len=pre_len,
        )

        dataset = CreateDataset(seq_l, pre_len, self.feat, self.extra_feat, device)
        total_samples = len(dataset)
        support_count = int(total_samples * support_rate)

        train_indices = list(range(support_count))
        test_indices = list(range(support_count, total_samples))

        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)

        use_cuda = (device.type == "cuda")
        num_workers = min(4, os.cpu_count() or 1)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=(num_workers > 0),
            drop_last=True if len(train_dataset) >= batch_size else False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )

        extra_feat_tag = self.extra_feat is not None

        self.trainer = trainer_module(
            train_loader=train_loader,
            test_loader=test_loader,
            extra_feat_tag=extra_feat_tag,
            model=model,
            save_path=save_path,
            scaler=scaler,
            device=device,
        )

    def train(
        self,
        now_epoch: int,
        local_epochs: int,
        save_model: bool = False,
    ) -> None:
        self.trainer.now_epoch = now_epoch
        self.trainer.deploy_epoch = 1
        self.trainer.training(epoch=local_epochs, save_model=save_model)

    def test(
        self,
        now_epoch: int,
        model_path: Optional[str] = None,
    ) -> None:
        self.trainer.now_epoch = now_epoch
        self.trainer.test(model_path=model_path)

    def localize(
        self,
        now_epoch: int,
        deploy_epochs: int,
        save_model: bool = False,
        model_path: Optional[str] = None,
    ) -> None:
        self.trainer.now_epoch = now_epoch

        for deploy_epoch in range(1, deploy_epochs + 1):
            self.trainer.deploy_epoch = deploy_epoch
            self.trainer.training(epoch=1, save_model=save_model)
            self.trainer.test(model_path=model_path)

    def refresh(self, model: Any) -> None:
        for w_local, w_global in zip(
            self.trainer.ev_model.model.parameters(),
            model.model.parameters()
        ):
            w_local.data.copy_(w_global.data)

    def get_model(self) -> Generator[torch.Tensor, None, None]:
        return self.trainer.ev_model.model.parameters()