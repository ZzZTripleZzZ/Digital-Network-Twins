
import argparse
import h5py
import pandas as pd
import numpy as np
import copy
import torch
import torch.nn.functional as F
from sklearn import cluster
import random
import os
from scipy import linalg
from sklearn import metrics
import networkx as nx
from networkx.algorithms.community import girvan_newman
import itertools
from utils.fed_update import LocalUpdate, test_inference
import matplotlib.pyplot as plt


class MinMaxNormalization(object):

    def __init__(self):
        pass

    def fit(self, X):
        self.min = X.min()
        self.max = X.max()
        print("min:", self.min, "max:", self.max)

    def transform(self, X):
        X = 1. * (X - self.min) / (self.max - self.min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self.max - self.min) + self.min
        return X


class MinMaxNormalization_01(object):

    def __init__(self):
        pass

    def fit(self, X):
        self.min = X.min()
        self.max = X.max()
        print("min:", self.min, "max:", self.max)

    def transform(self, X):
        X = 1. * (X - self.min) / (self.max - self.min)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = 1. * X * (self.max - self.min) + self.min
        return X


def jfi(d):
    return np.sum(d) ** 2 / (len(d) * np.sum(d ** 2))


def cv(d):
    return np.std(d) / np.mean(d)

def get_data(args):
    path = os.getcwd()
    f = h5py.File(path + '/dataset/' + args.file, 'r')

    idx = f['idx'][()]
    cell = f['cell'][()]
    lng = f['lng'][()]
    lat = f['lat'][()]
    data = f[args.type][()][:, cell - 1]

    df = pd.DataFrame(data, index=pd.to_datetime(idx.ravel(), unit='s'), columns=cell)
    df.fillna(0, inplace=True)

    random.seed(args.seed)
    cell_pool = cell
    selected_cells = sorted(random.sample(list(cell_pool), args.bs))
    selected_cells_idx = np.where(np.isin(list(cell), selected_cells))
    cell_lng = lng[selected_cells_idx]
    cell_lat = lat[selected_cells_idx]

    df_cells = df[selected_cells]

    train_data = df_cells.iloc[:-args.test_days * 24]

    mean = train_data.mean()
    std = train_data.std()

    normalized_df = (df_cells - mean) / std

    return normalized_df, df_cells, selected_cells, mean, std, cell_lng, cell_lat


def get_cluster_label(args, df_traffic, lng, lat):
    df_ori = copy.deepcopy(df_traffic)

    df_ori['lng'] = lng
    df_ori['lat'] = lat
    df_ori['label'] = -1

    loc_init = np.zeros((args.cluster, 2))
    tp_init = np.zeros((args.cluster, df_ori.drop(['lng', 'lat', 'label'], axis=1).shape[1]))
    geo_old_label = tp_old_label = [0] * args.bs
    for i in range(20):

        km_geo = cluster.KMeans(n_clusters=args.cluster, init=loc_init, n_init=1).fit(df_ori[['lng', 'lat']].values)
        km_tp = cluster.KMeans(n_clusters=args.cluster, init=tp_init, n_init=1).fit(
            df_ori.drop(['lng', 'lat', 'label'], axis=1).values
        )
        if args.pattern == 'geo':
            vm_geo = metrics.v_measure_score(geo_old_label, km_geo.labels_)
            if vm_geo == 1:
                break
            else:
                df_ori['label'] = km_tp.labels_
                loc_init = df_ori.groupby(['label']).mean()[['lng', 'lat']].values
                geo_old_label = km_geo.labels_
        elif args.pattern == 'tp':
            vm_tp = metrics.v_measure_score(tp_old_label, km_tp.labels_)
            if vm_tp == 1:
                break
            else:
                df_ori['label'] = km_geo.labels_
                tp_init = df_ori.groupby(['label']).mean().drop(['lng', 'lat'], axis=1).values
                tp_old_label = km_tp.labels_
        else:
            print('wrong choice')
    if args.pattern == 'geo':
        return km_geo.labels_
    elif args.pattern == 'tp':
        return km_tp.labels_
    else:
        return km_tp.labels_


def process_centralized(args, dataset):
    train_x_close, val_x_close, test_x_close = [], [], []
    train_x_period, val_x_period, test_x_period = [], [], []
    train_label, val_label, test_label = [], [], []

    column_names = dataset.columns

    for col in column_names:
        close_arr = []
        period_arr = []
        label_arr = []

        cell_traffic = dataset[col]
        start_idx = max(args.close_size, args.period_size * 24)
        for idx in range(start_idx, len(dataset) - args.out_dim + 1):
            y_ = [cell_traffic.iloc[idx + i] for i in range(args.out_dim)]
            label_arr.append(y_)
            if args.close_size > 0:
                x_close = [cell_traffic.iloc[idx - c] for c in range(1, args.close_size + 1)]
                close_arr.append(x_close)
            if args.period_size > 0:
                x_period = [cell_traffic.iloc[idx - p * 24] for p in range(1, args.period_size + 1)]
                period_arr.append(x_period)
        cell_arr_close = np.array(close_arr)

        cell_label = np.array(label_arr)

        test_len = args.test_days * 24
        val_len = args.val_days * 24
        train_len = len(cell_arr_close) - test_len - val_len
        train_x_close.append(cell_arr_close[:train_len])
        val_x_close.append(cell_arr_close[train_len:train_len + val_len])
        test_x_close.append(cell_arr_close[-test_len:])

        train_label.append(cell_label[:train_len])
        val_label.append(cell_label[train_len:train_len + val_len])
        test_label.append(cell_label[-test_len:])

        if args.period_size > 0:
            cell_arr_period = np.array(period_arr)
            train_x_period.append(cell_arr_period[:train_len])
            val_x_period.append(cell_arr_period[train_len:train_len + val_len])
            test_x_period.append(cell_arr_period[-test_len:])
        else:
            train_x_period = train_x_close
            val_x_period = val_x_close
            test_x_period = test_x_close

    train_xc = np.concatenate(train_x_close)[:, :, np.newaxis]
    if len(val_x_close) > 0:
        val_xc = np.concatenate(val_x_close)[:, :, np.newaxis]
    test_xc = np.concatenate(test_x_close)[:, :, np.newaxis]

    train_xp = np.concatenate(train_x_period)[:, :, np.newaxis]
    if len(val_x_period) > 0:
        val_xp = np.concatenate(val_x_period)[:, :, np.newaxis]
    test_xp = np.concatenate(test_x_period)[:, :, np.newaxis]

    train_y = np.concatenate(train_label)
    val_y = np.concatenate(val_label)
    test_y = np.concatenate(test_label)

    return (train_xc, train_xp, train_y), (val_xc, val_xp, val_y), (test_xc, test_xp, test_y)


def process_isolated(args, dataset):
    train, val, test = dict(), dict(), dict()
    column_names = dataset.columns

    for col in column_names:
        close_arr, period_arr, label_arr = [], [], []

        cell_traffic = dataset[col]
        start_idx = max(args.close_size, args.period_size * 24)
        for idx in range(start_idx, len(dataset) - args.out_dim + 1):
            y_ = [cell_traffic.iloc[idx + i] for i in range(args.out_dim)]
            # y_ = [cell_traffic.iloc[idx + args.out_dim - 1]]
            label_arr.append(y_)

            if args.close_size > 0:
                x_close = [cell_traffic.iloc[idx - c] for c in range(1, args.close_size + 1)]
                close_arr.append(x_close)
            if args.period_size > 0:
                x_period = [cell_traffic.iloc[idx - p * 24] for p in range(1, args.period_size + 1)]
                period_arr.append(x_period)

        cell_arr_close = np.array(close_arr)
        cell_arr_close = cell_arr_close[:, :, np.newaxis]
        cell_label = np.array(label_arr)

        test_len = args.test_days * 24
        val_len = args.val_days * 24
        train_len = len(cell_arr_close) - test_len - val_len

        train_x_close = cell_arr_close[:train_len]
        val_x_close = cell_arr_close[train_len:train_len + val_len]
        test_x_close = cell_arr_close[-test_len:]

        train_label = cell_label[:train_len]
        val_label = cell_label[train_len:train_len + val_len]
        test_label = cell_label[-test_len:]

        if args.period_size > 0:
            cell_arr_period = np.array(period_arr)
            cell_arr_period = cell_arr_period[:, :, np.newaxis]

            train_x_period = cell_arr_period[:train_len]
            val_x_period = cell_arr_period[train_len:train_len + val_len]
            test_x_period = cell_arr_period[-test_len:]

        else:
            train_x_period = train_x_close
            val_x_period = val_x_close
            test_x_period = test_x_close

        train[col] = (train_x_close, train_x_period, train_label)
        val[col] = (val_x_close, val_x_period, val_label)
        test[col] = (test_x_close, test_x_period, test_label)

    return train, val, test


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_weights_cluster(args, w, community):
    w_avg = copy.deepcopy(w[0])

    total_count = {}

    for key in w_avg.keys():
        w_avg[key].zero_()
        total_count[key] = 0

        for i in range(len(w)):
            w_avg[key] += w[i][key] * len(community[i])
            total_count[key] += len(community[i])

        w_avg[key] = torch.div(w_avg[key], total_count[key])

    return w_avg


def average_weights_att(w_clients, w_server, epsilon=1.0):
    w_next = copy.deepcopy(w_server)
    att = {}
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k])
        att[k] = torch.zeros(len(w_clients))

    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            att[k][i] = torch.from_numpy(np.array(linalg.norm(w_server[k].cpu() - w_clients[i][k].cpu())))

    for k in w_next.keys():
        att[k] = F.softmax(att[k], dim=0)

    for k in w_next.keys():
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            att_weight += torch.mul(w_server[k] - w_clients[i][k], att[k][i])

        w_next[k] = w_server[k] - torch.mul(att_weight, epsilon)

    return w_next



def avg_dual_att(w_clients, w_server, warm_server, epsilon=1.0, rho=0.1):
    w_next = copy.deepcopy(w_server)
    att = {}
    att_warm = {}
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k])
        att[k] = torch.zeros(len(w_clients))
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            att[k][i] = torch.from_numpy(np.array(linalg.norm(w_server[k].cpu() - w_clients[i][k].cpu())))
        sw_diff = w_server[k].cpu() - warm_server[k].cpu()
        att_warm[k] = torch.FloatTensor(np.array(linalg.norm(sw_diff)))
    warm_tensor = torch.FloatTensor([v for k, v in att_warm.items()])
    layer_w = F.softmax(warm_tensor, dim=0)
    for i, k in enumerate(w_next.keys()):
        att[k] = F.softmax(att[k], dim=0)
        att_warm[k] = layer_w[i]
    for k in w_next.keys():
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            att_weight += torch.mul(w_server[k] - w_clients[i][k], att[k][i])
        att_weight += torch.mul(w_server[k] - warm_server[k], rho*att_warm[k])
        w_next[k] = w_server[k] - torch.mul(att_weight, epsilon)
    return w_next


def get_warm_up_data(args, data):
    close_arr, period_arr, label_arr = [], [], []
    start_idx = max(args.close_size, args.period_size * 24)
    for idx in range(start_idx, len(data) - args.out_dim + 1):
        y_ = [data.iloc[idx + i] for i in range(args.out_dim)]
        label_arr.append(y_)
        if args.close_size > 0:
            x_close = [data.iloc[idx - c] for c in range(1, args.close_size + 1)]
            close_arr.append(x_close)
        if args.period_size > 0:
            x_period = [data.iloc[idx - p * 24] for p in range(1, args.period_size + 1)]
            period_arr.append(x_period)
    cell_arr_close = np.array(close_arr)
    cell_arr_period = np.array(period_arr)
    cell_label = np.array(label_arr)
    return cell_arr_close, cell_arr_period, cell_label


def clustering(args, cluster_num, adj, lng, lat, weights):
    comm = []
    param = copy.deepcopy(adj)
    for i in range(args.bs):
        for j in range(args.bs):
            if param[i][j] != 0:
                dist = 1 / distance(lng[i], lat[i], lng[j], lat[j])
                param[i][j] = weights[0] * dist + weights[1] * param[i][j] * np.random.rand()
    G = nx.from_numpy_array(np.matrix(param))
    k = cluster_num
    comp = girvan_newman(G)
    limited = itertools.takewhile(lambda c: len(c) <= k, comp)
    for communities in limited:
        comm = list(sorted(c) for c in communities)
    return comm


def distance(lng1, lat1, lng2, lat2):
    return np.sqrt(np.power((lng1-lng2), 2) + np.power((lat1-lat2), 2))


def merge_arrays(id_array, group_array):
    merged_array = []
    for subarray in group_array:
        merged_subarray = []
        for index in subarray:
            merged_subarray.append(id_array[index])
        merged_array.append(merged_subarray)
    return merged_array


def compare_state_dicts(state_dict1, state_dict2):
    mse_list = []
    for key in state_dict1.keys():
        if key in state_dict2:
            tensor1 = state_dict1[key]
            tensor2 = state_dict2[key]
            if tensor1.shape != tensor2.shape:
                print(f"Skipping {key} due to shape mismatch.")
                continue
            mse = F.mse_loss(tensor1, tensor2).item()
            mse_list.append(mse)
    if len(mse_list) == 0:
        return float('inf')
    average_mse = sum(mse_list) / len(mse_list)
    return sum(mse_list)

def update_model(args, global_model, cell, past, future, local_weights, local_losses, cell_loss, epoch):
    cell_past, cell_future = past[cell], future[cell]
    local_model = LocalUpdate(args, cell_future, cell_future)
    w, loss, epoch_loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
    local_weights.append(copy.deepcopy(w))
    local_losses.append(copy.deepcopy(loss))
    cell_loss.append(loss)