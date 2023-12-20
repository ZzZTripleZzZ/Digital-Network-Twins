import numpy as np
import h5py
import tqdm
import copy
import torch
import pandas as pd
import random
from collections import defaultdict
import os
import sys
from sklearn import metrics
import argparse
import time
import threading

from utils.misc import average_weights_att, average_weights, average_weights_cluster,compare_state_dicts, update_model
from utils.misc import get_data, process_isolated, clustering, merge_arrays
from utils.models import LSTM
from utils.fed_update import LocalUpdate, test_inference

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# input parameters
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='multi-layer digital twins')
    parser.add_argument('--file', type=str, default='milano.h5',
                        help='file path and name')
    parser.add_argument('--type', type=str, default='net', help='which kind of wireless traffic')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='Use CUDA for training')
    parser.add_argument('--close_size', type=int, default=5,
                        help='how many time slots before target are used to model closeness')
    parser.add_argument('--period_size', type=int, default=5,
                        help='how many trend slots before target are used to model periodicity')
    parser.add_argument('--test_days', type=int, default=10,
                        help='how many days data are used to test model performance')
    parser.add_argument('--val_days', type=int, default=10,
                        help='how many days data are used to valid model performance')
    parser.add_argument('--bs', type=int, default=50, help='number of base stations')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='fraction of clients: C')
    parser.add_argument('--cluster', type=int, default=10, help='number of clusters to be divided')
    parser.add_argument('--cluster_weight', type=float, default=30,
                        help='weight of distance and connectivity')
    parser.add_argument('--topo', type=str, default='regular_graph_50agents_10degree', help='network topology')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='threshold of dt similarities')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size of centralized training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epochs of vertical twinning')
    parser.add_argument('--epochs_horizontal', type=int, default=20,
                        help='epochs of horizontal twinning')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='saving the model')
    parser.add_argument('--seed', type=int, default=1, help='random seeds')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate of NN')
    parser.add_argument('--opt', type=str, default='adam', help='optimization techniques')
    parser.add_argument('--momentum', type=float, default=0.95, help='momentum')
    parser.add_argument('--epsilon', type=float, default=1, help='stepsize')
    parser.add_argument('--fedsgd', type=int, default=0, help='FedSGD')
    parser.add_argument('--local_epoch', type=int, default=3,
                        help='the number of local epochs: E')
    parser.add_argument('--local_batch', type=int, default=32,
                        help='local batch size for training')
    parser.add_argument('--gen_batch', type=int, default=48,
                        help='local batch size for prediction')
    parser.add_argument('--phi', type=float, default=1.0, help='how many samples are shared')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='input feature dimension of LSTM')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='hidden neurons of LSTM layer')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers of LSTM')
    parser.add_argument('--out_dim', type=int, default=1,
                        help='how many steps we would like to predict for the future')
    parser.add_argument('--local_batch_2', type=int, default=20,
                        help='local batch size for prediction')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, df_ori, selected_cells, mean, std, lng, lat = get_data(args)
    device = 'cuda' if args.gpu else 'cpu'
    past, future, val = process_isolated(args, data)
    global_model = LSTM(args).to(device)
    global_weight = global_model.state_dict()
    if args.topo == 'regular_graph_50agents_10degree':
        adjacency_file = 'topo/regular_graph_50agents_10degree.txt'

    elif args.topo == 'regular_graph_50agents_20degree':
        adjacency_file = 'topo/regular_graph_50agents_20degree.txt'

    elif args.topo == 'regular_graph_50agents_30degree':
        adjacency_file = 'topo/regular_graph_50agents_30degree.txt'

    elif args.topo == 'regular_graph_50agents_40degree':
        adjacency_file = 'topo/regular_graph_50agents_40degree.txt'

    elif args.topo == 'regular_graph_25agents_10degree':
        adjacency_file = 'topo/regular_graph_25agents_10degree.txt'

    elif args.topo == 'regular_graph_100agents_40degree':
        adjacency_file = 'topo/regular_graph_100agents_40degree.txt'

    elif args.topo == 'regular_graph_200agents_80degree':
        adjacency_file = 'topo/regular_graph_200agents_80degree.txt'

    else:
        adjacency_file = 'topo/regular_graph_2.txt'

    with open(adjacency_file, 'r') as f:
        adjacency = [[int(num) for num in line.split(' ')] for line in f]

    weights = [args.cluster_weight, 100-args.cluster_weight]
    community = clustering(args, args.cluster, adjacency, lng, lat, weights)
    bs_group = merge_arrays(selected_cells, community)
    best_val_loss = None
    val_loss = []
    val_acc = []
    cell_loss = []
    loss_hist = []
    global_twins = []
    global_twin = global_model.state_dict()
    execution_times = []
    start_time = time.time()
    for epoch in tqdm.tqdm(range(args.epochs)):
        global_twins = []
        for cluster in range(args.cluster):
            local_weights, local_losses = [], []
            cell_idx = bs_group[cluster]
            for cell in cell_idx:
                cell_past, cell_future = past[cell], future[cell]
                local_model = LocalUpdate(args, cell_past, cell_future)
                w, loss, epoch_loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                                                 global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
                cell_loss.append(loss)
            loss_hist.append(sum(cell_loss) / len(cell_loss))
            global_weight = average_weights_att(local_weights, global_weight, args.epsilon)
            global_model.load_state_dict(global_weight)
            global_twins.append(copy.deepcopy(global_weight))
        global_twin = average_weights_cluster(args, global_twins, community)
        global_model.load_state_dict(global_twin)
        if epoch % 20 == 0:
            community = clustering(args, args.cluster, adjacency, lng, lat, weights)
    end_time = time.time()
    pred, truth = defaultdict(), defaultdict()
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0
    global_model.load_state_dict(global_twin)
    with torch.no_grad():
        for cell in selected_cells:
            pred_list = []
            truth_list = []
            cell_future = future[cell]
            test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, global_model, cell_future)
            pred_list.append(pred[cell])
            truth_list.append(truth[cell])
            nrmse += test_nrmse
    df_pred = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in pred.items()]))
    df_truth = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in truth.items()]))
    df_pred = df_pred.dropna()
    df_truth = df_truth.dropna()
    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    comm = 0
    comm_hist = []
    start_time_2 = time.time()
    val_loss = []
    val_acc = []
    cell_loss = []
    loss_hist = []
    global_twin = global_model.state_dict()
    for epoch in tqdm.tqdm(range(args.epochs_horizontal)):
        global_twins = []
        for cluster in range(args.cluster):
            local_weights, local_losses = [], []
            cell_idx = bs_group[cluster]
            for cell in cell_idx:
                cell_past, cell_future = past[cell], future[cell]
                local_model = LocalUpdate(args, cell_future, cell_future)
                w, loss, epoch_loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                                                 global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
                cell_loss.append(loss)
            loss_hist.append(sum(cell_loss) / len(cell_loss))
            cluster_weight = global_model.state_dict()
            cluster_weight = average_weights_att(local_weights, cluster_weight, args.epsilon)
            diff = compare_state_dicts(cluster_weight, global_twin)
            if diff > args.threshold:
                comm = comm + len(community[cluster])
                global_twins.append(copy.deepcopy(cluster_weight))
            else:
                global_twins.append(copy.deepcopy(global_model.state_dict()))
            comm_hist.append(comm)
        global_twin = average_weights_cluster(args, global_twins, community)
        global_model.load_state_dict(global_twin)
        if epoch % 5 == 0:
            community = clustering(args, args.cluster, adjacency, lng, lat, weights)
    end_time_2 = time.time()
    pred, truth = defaultdict(), defaultdict()
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0
    global_model.load_state_dict(global_twin)
    with torch.no_grad():
        for cell in selected_cells:
            pred_list = []
            truth_list = []
            cell_val = val[cell]
            test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, global_model,
                                                                                      cell_val)
            pred_list.append(pred[cell])
            truth_list.append(truth[cell])
            nrmse += test_nrmse
    df_pred = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in pred.items()]))
    df_truth = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in truth.items()]))
    df_pred = df_pred.dropna()
    df_truth = df_truth.dropna()
    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)