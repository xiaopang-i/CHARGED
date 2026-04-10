# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/13 20:14
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/13 20:14

"""
Module for loading and processing electric vehicle dataset features in distributed settings.
"""

from collections import defaultdict
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from api.utils import get_n_feature


class DistributedEVDataset(object):
    """
    Distributed dataset handler for EV features across multiple clients (cities/sites).
    """

    def __init__(
        self,
        feature: str,
        auxiliary: str,
        data_paths: Dict[str, str],
        pred_type: str,
        eval_percentage: float,
        eval_city: str,
        max_sites: int = 300,
        weather_columns: List[str] = ['temp', 'precip', 'visibility'],
        selection_mode: str = 'middle',
    ) -> None:
        super(DistributedEVDataset, self).__init__()
        self.feature = feature
        self.auxiliary = auxiliary
        self.clients_data = {}
        self.city_scalers = {}

        for city_abbr, data_path in data_paths.items():
            # 1) 读取主特征
            if self.feature == 'volume':
                feat_df = pd.read_csv(f'{data_path}volume.csv', header=0, index_col=0)
            elif self.feature == 'duration':
                feat_df = pd.read_csv(f'{data_path}duration.csv', header=0, index_col=0)
            else:
                raise ValueError("Unknown feature - must be 'volume' or 'duration'")

            feat_df.columns = feat_df.columns.astype(str)

            # 2) 按城市最大值归一化
            max_val = feat_df.max().max()
            if max_val == 0:
                max_val = 1.0
            feat_df = feat_df / max_val
            self.city_scalers[city_abbr] = max_val

            time_series = pd.to_datetime(feat_df.index)

            # 3) 电价 / 服务费
            e_price_df = pd.read_csv(f'{data_path}e_price.csv', index_col=0, header=0)
            s_price_df = pd.read_csv(f'{data_path}s_price.csv', index_col=0, header=0)
            e_price_df.columns = e_price_df.columns.astype(str)
            s_price_df.columns = s_price_df.columns.astype(str)

            # 先按 feat_df 对齐
            common_cols = [c for c in feat_df.columns if c in e_price_df.columns and c in s_price_df.columns]
            feat_df = feat_df[common_cols]
            e_price_df = e_price_df[common_cols]
            s_price_df = s_price_df[common_cols]

            price_scaler = MinMaxScaler(feature_range=(0, 1))
            e_price_all = price_scaler.fit_transform(e_price_df.values)
            s_price_all = price_scaler.fit_transform(s_price_df.values)

            # 4) 天气
            weather = pd.read_csv(f'{data_path}weather.csv', header=0, index_col='time')
            weather = weather[weather_columns].copy()

            if 'temp' in weather.columns:
                weather['temp'] = (weather['temp'] + 5) / 45
            if 'precip' in weather.columns:
                weather['precip'] = weather['precip'] / 120
            if 'visibility' in weather.columns:
                weather['visibility'] = weather['visibility'] / 50

            # 5) 站点信息：这里按你的真实文件 sites.csv 处理
            site_info = pd.read_csv(f'{data_path}sites.csv')
            if 'site' not in site_info.columns:
                raise ValueError("{}sites.csv must contain column: site".format(data_path))
            site_info = site_info.set_index('site')
            site_info.index = site_info.index.astype(str)

            # 只保留同时存在于 feat_df 和 site_info 的站点
            common_site_ids = [c for c in feat_df.columns if c in site_info.index]
            feat_df = feat_df[common_site_ids]
            e_price_df = e_price_df[common_site_ids]
            s_price_df = s_price_df[common_site_ids]
            site_info = site_info.loc[common_site_ids].copy()

            # 更新价格归一化结果
            e_price_all = price_scaler.fit_transform(e_price_df.values)
            s_price_all = price_scaler.fit_transform(s_price_df.values)

            sites = list(feat_df.columns)

            # 6) 可选站点筛选
            if pred_type == 'site' and len(sites) > max_sites:
                if 'total_duration' not in site_info.columns:
                    raise ValueError(
                        "{}sites.csv must contain column 'total_duration' when site selection is needed".format(
                            data_path
                        )
                    )

                if selection_mode == 'top':
                    selected_site_info = site_info.sort_values(
                        by='total_duration', ascending=False
                    ).head(max_sites)
                elif selection_mode == 'middle':
                    sorted_site_info = site_info.sort_values(
                        by='total_duration', ascending=True
                    )
                    start = max((len(sorted_site_info) - max_sites) // 2, 0)
                    selected_site_info = sorted_site_info.iloc[start:start + max_sites]
                elif selection_mode == 'random':
                    selected_site_info = site_info.sample(n=max_sites, random_state=42)
                else:
                    raise ValueError("Unknown selection_mode: {}".format(selection_mode))

                selected_ids = selected_site_info.index.astype(str).tolist()

                feat_df = feat_df[selected_ids]
                e_price_df = e_price_df[selected_ids]
                s_price_df = s_price_df[selected_ids]
                site_info = selected_site_info
                sites = selected_ids

                e_price_all = price_scaler.fit_transform(e_price_df.values)
                s_price_all = price_scaler.fit_transform(s_price_df.values)

            # 7) 额外特征
            extra_feat = None

            if self.auxiliary != 'None':
                parts = []

                if self.auxiliary == 'all':
                    parts.append(e_price_all[:, :, np.newaxis])
                    parts.append(s_price_all[:, :, np.newaxis])
                    parts.append(
                        np.repeat(weather.values[:, np.newaxis, :], len(sites), axis=1)
                    )
                else:
                    add_feat_list = self.auxiliary.split('+')
                    for add_feat in add_feat_list:
                        if add_feat == 'e_price':
                            parts.append(e_price_all[:, :, np.newaxis])
                        elif add_feat == 's_price':
                            parts.append(s_price_all[:, :, np.newaxis])
                        else:
                            if add_feat not in weather.columns:
                                raise ValueError(
                                    "Unknown auxiliary feature: {}. Available weather columns: {}".format(
                                        add_feat, list(weather.columns)
                                    )
                                )
                            parts.append(
                                np.repeat(
                                    weather[add_feat].values[:, np.newaxis, np.newaxis],
                                    len(sites),
                                    axis=1
                                )
                            )

                if len(parts) > 0:
                    extra_feat = np.concatenate(parts, axis=2)

            feat_array = feat_df.values.astype(np.float32)
            if extra_feat is not None:
                extra_feat = extra_feat.astype(np.float32)

            self.n_fea = get_n_feature(extra_feat)

            # 8) 按预测粒度切客户端
            if pred_type == 'site':
                for idx, site in enumerate(sites):
                    client_feat = feat_array[:, idx:idx + 1]
                    client_extra = extra_feat[:, idx:idx + 1, :] if extra_feat is not None else None
                    client_id = "{}_{}".format(city_abbr, site)
                    self.clients_data[client_id] = {
                        'feat': client_feat,
                        'extra_feat': client_extra,
                        'time': time_series,
                    }

            elif pred_type == 'city':
                aggregated_feat = np.sum(feat_array, axis=1, keepdims=True)
                aggregated_extra = (
                    np.mean(extra_feat, axis=1, keepdims=True) if extra_feat is not None else None
                )
                self.clients_data[city_abbr] = {
                    'feat': aggregated_feat,
                    'extra_feat': aggregated_extra,
                    'time': time_series,
                }
            else:
                raise ValueError("Unknown pred_type - must be 'site' or 'city'")

        self.partition_clients(eval_percentage, eval_city, pred_type)

    def partition_clients(
        self,
        eval_percentage: float,
        eval_city: str,
        pred_type: str,
    ) -> None:
        if pred_type == 'site':
            training_clients_data = {}
            eval_clients_data = {}
            city_clients = defaultdict(list)

            for client_id in self.clients_data.keys():
                city = client_id.split('_')[0]
                city_clients[city].append(client_id)

            for city, client_ids in city_clients.items():
                client_ids_sorted = sorted(client_ids)
                num_clients = len(client_ids_sorted)
                num_eval = int(num_clients * eval_percentage / 100)

                if num_clients > 0 and num_eval == 0 and eval_percentage > 0:
                    num_eval = 1

                eval_ids = client_ids_sorted[:num_eval]
                train_ids = client_ids_sorted[num_eval:]

                for cid in train_ids:
                    training_clients_data[cid] = self.clients_data[cid]
                for cid in eval_ids:
                    eval_clients_data[cid] = self.clients_data[cid]

            self.training_clients_data = training_clients_data
            self.eval_clients_data = eval_clients_data

        elif pred_type == 'city':
            training_clients_data = {}
            eval_clients_data = {}

            for city, data in self.clients_data.items():
                if city == eval_city:
                    eval_clients_data[city] = data
                else:
                    training_clients_data[city] = data

            self.training_clients_data = training_clients_data
            self.eval_clients_data = eval_clients_data

        else:
            raise ValueError("Unknown pred_type - must be 'site' or 'city'")

    def get_client_ids(self) -> List[str]:
        return list(self.clients_data.keys())

    def get_client_data(self, client_id: str) -> Optional[Dict[str, Any]]:
        return self.clients_data.get(client_id, None)