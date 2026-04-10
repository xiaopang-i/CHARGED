# -*- coding: utf-8 -*-
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from api.utils import get_n_feature


class DistributedEVDataset(object):
    def __init__(
            self,
            feature: str,
            auxiliary: str,
            data_paths: dict,
            pred_type: str,
            eval_percentage: float,
            eval_city: str,
            max_sites: int = 300,
            weather_columns: list = ['temp', 'precip', 'visibility'],
            selection_mode: str = 'middle',
    ) -> None:
        super(DistributedEVDataset, self).__init__()
        self.feature = feature
        self.auxiliary = auxiliary
        self.clients_data = {}
        self.city_scalers = {}

        for city_abbr, data_path in data_paths.items():
            # 1. 主特征
            if self.feature == 'volume':
                feat_df = pd.read_csv(f'{data_path}volume.csv', header=0, index_col=0)
            elif self.feature == 'duration':
                feat_df = pd.read_csv(f'{data_path}duration.csv', header=0, index_col=0)
            else:
                raise ValueError("Unknown feature - must be 'volume' or 'duration'")

            max_val = feat_df.max().max()
            feat_df = feat_df / max_val
            self.city_scalers[city_abbr] = max_val

            # 统一列名为字符串，方便和 sites.csv 的 index 对齐
            feat_df.columns = feat_df.columns.astype(str)
            sites = list(feat_df.columns)

            # 2. 价格
            e_price_df = pd.read_csv(f'{data_path}e_price.csv', index_col=0, header=0)
            s_price_df = pd.read_csv(f'{data_path}s_price.csv', index_col=0, header=0)
            e_price_df.columns = e_price_df.columns.astype(str)
            s_price_df.columns = s_price_df.columns.astype(str)

            price_scaler = MinMaxScaler(feature_range=(0, 1))
            e_price_all = price_scaler.fit_transform(e_price_df[sites])
            s_price_all = price_scaler.fit_transform(s_price_df[sites])

            # 3. 时间
            time_series = pd.to_datetime(feat_df.index)

            # 4. 天气
            weather = pd.read_csv(f'{data_path}weather.csv', header=0, index_col='time')
            weather = weather[weather_columns].copy()

            if 'temp' in weather.columns:
                weather['temp'] = (weather['temp'] + 5) / 45
            if 'precip' in weather.columns:
                weather['precip'] = weather['precip'] / 120
            if 'visibility' in weather.columns:
                weather['visibility'] = weather['visibility'] / 50

            # 5. 站点信息
            siteinfo = pd.read_csv(f'{data_path}sites.csv')
            siteinfo['site'] = siteinfo['site'].astype(str)
            siteinfo = siteinfo.set_index('site')

            # 6. site 模式时做站点筛选
            if pred_type == 'site' and len(siteinfo) > max_sites:
                if selection_mode == 'top':
                    selected_siteinfo = siteinfo.sort_values(
                        by='total_duration', ascending=False
                    ).head(max_sites)
                elif selection_mode == 'middle':
                    sorted_siteinfo = siteinfo.sort_values(
                        by='total_duration', ascending=True
                    )
                    start = max((len(sorted_siteinfo) - max_sites) // 2, 0)
                    selected_siteinfo = sorted_siteinfo.iloc[start:start + max_sites]
                elif selection_mode == 'random':
                    selected_siteinfo = siteinfo.sample(n=max_sites, random_state=42)
                else:
                    raise ValueError(f"Unknown selection_mode: {selection_mode}")

                selected_ids = selected_siteinfo.index.tolist()

                # 只保留 feat 中真实存在的列
                selected_ids = [sid for sid in selected_ids if sid in feat_df.columns]

                feat_df = feat_df[selected_ids]
                e_price_all = price_scaler.fit_transform(e_price_df[selected_ids])
                s_price_all = price_scaler.fit_transform(s_price_df[selected_ids])
                siteinfo = selected_siteinfo.loc[selected_ids]
                sites = selected_ids

            # 7. 空间特征（经纬度）
            lat_long = siteinfo.loc[feat_df.columns, ['latitude', 'longitude']].values
            lat_norm = (lat_long[:, 0] + 90) / 180
            lon_norm = (lat_long[:, 1] + 180) / 360
            lat_long_norm = np.stack([lat_norm, lon_norm], axis=1)

            # 默认先做空间特征
            extra_feat = np.tile(lat_long_norm[np.newaxis, :, :], (feat_df.shape[0], 1, 1))

            # 8. 辅助特征
            if self.auxiliary != 'None':
                extra_feat = np.zeros([feat_df.shape[0], feat_df.shape[1], 1])

                if self.auxiliary == 'all':
                    extra_feat = np.concatenate([extra_feat, e_price_all[:, :, np.newaxis]], axis=2)
                    extra_feat = np.concatenate([extra_feat, s_price_all[:, :, np.newaxis]], axis=2)
                    extra_feat = np.concatenate(
                        [
                            extra_feat,
                            np.repeat(weather.values[:, np.newaxis, :], feat_df.shape[1], axis=1)
                        ],
                        axis=2
                    )
                else:
                    add_feat_list = self.auxiliary.split('+')
                    for add_feat in add_feat_list:
                        if add_feat == 'e_price':
                            extra_feat = np.concatenate([extra_feat, e_price_all[:, :, np.newaxis]], axis=2)
                        elif add_feat == 's_price':
                            extra_feat = np.concatenate([extra_feat, s_price_all[:, :, np.newaxis]], axis=2)
                        elif add_feat in weather.columns:
                            extra_feat = np.concatenate(
                                [
                                    extra_feat,
                                    np.repeat(
                                        weather[add_feat].values[:, np.newaxis, np.newaxis],
                                        feat_df.shape[1],
                                        axis=1
                                    )
                                ],
                                axis=2
                            )
                        elif add_feat in ('longitude', 'latitude', 'location'):
                            # 保底支持地理位置
                            extra_feat = np.concatenate([extra_feat, np.tile(lat_long_norm[np.newaxis, :, :], (feat_df.shape[0], 1, 1))], axis=2)
                        else:
                            raise ValueError(f"Unknown auxiliary feature: {add_feat}")

                extra_feat = extra_feat[:, :, 1:]

            feat_array = np.array(feat_df)
            self.n_fea = get_n_feature(extra_feat)

            # 9. client 划分
            if pred_type == 'site':
                for idx, site in enumerate(sites):
                    client_feat = feat_array[:, idx:idx + 1]
                    client_extra = extra_feat[:, idx:idx + 1, :] if extra_feat is not None else None
                    client_id = f"{city_abbr}_{site}"
                    self.clients_data[client_id] = {
                        'feat': client_feat,
                        'extra_feat': client_extra,
                        'time': time_series
                    }
            elif pred_type == 'city':
                aggregated_feat = np.sum(feat_array, axis=1, keepdims=True)
                aggregated_extra = np.mean(extra_feat, axis=1, keepdims=True) if extra_feat is not None else None
                self.clients_data[city_abbr] = {
                    'feat': aggregated_feat,
                    'extra_feat': aggregated_extra,
                    'time': time_series
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

    def get_client_ids(self):
        return list(self.clients_data.keys())

    def get_client_data(self, client_id: str):
        return self.clients_data.get(client_id, None)
