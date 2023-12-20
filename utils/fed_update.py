import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
import random

torch.manual_seed(2020)
np.random.seed(2020)


class LocalUpdate(object):
    def __init__(self, args, train, test):
        self.args = args
        self.train_loader = self.process_data(train)
        self.test_loader = self.process_data(test)
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.MSELoss().to(self.device)

    def process_data(self, dataset):
        data = list(zip(*dataset))
        if self.args.fedsgd == 1:
            loader = DataLoader(data, shuffle=False, batch_size=len(data))
        else:
            loader = DataLoader(data, shuffle=False, batch_size=self.args.local_batch)
        return loader

    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []
        lr = self.args.lr

        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        for iter in range(self.args.local_epoch):
            batch_loss = []

            for batch_idx, (xc, xp, y) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)

                model.zero_grad()
                pred = model(xc, xp)

                loss = self.criterion(y, pred)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss)/len(epoch_loss), epoch_loss


def test_inference(args, model, dataset):
    model.eval()
    loss, mse = 0.0, 0.0
    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.MSELoss().to(device)
    data_loader = DataLoader(list(zip(*dataset)), batch_size=args.gen_batch, shuffle=False)

    # data_loader = DataLoader(list(zip(*dataset)), batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    # print(data_loader)
    pred_list, truth_list = [], []

    with torch.no_grad():
        for batch_idx, (xc, xp, y) in enumerate(data_loader):
            # print(batch_idx)
            xc, xp = xc.float().to(device), xp.float().to(device)
            # print(xc)
            y = y.float().to(device)
            pred = model(xc, xp)

            batch_loss = criterion(y, pred)
            loss += batch_loss.item()

            batch_mse = torch.mean((pred - y) ** 2)
            mse += batch_mse.item()

            pred_list.append(pred.detach().cpu())
            truth_list.append(y.detach().cpu())

    final_prediction = np.concatenate(pred_list).ravel()
    final_truth = np.concatenate(truth_list).ravel()
    nrmse= (metrics.mean_squared_error(final_truth, final_prediction) ** 0.5) / (max(final_truth) - min(final_truth)+ 0.000001)
    avg_loss = loss / len(data_loader)
    avg_mse = mse / len(data_loader)

    return avg_loss, avg_mse, nrmse, final_prediction, final_truth


def custom_collate_fn(batch):
    min_batch_size = 16
    max_batch_size = 32

    random_batch_size = random.randint(min_batch_size, max_batch_size)

    random_batch = random.sample(batch, random_batch_size)

    xc, xp, y = zip(*random_batch)

    xc = torch.stack([torch.tensor(x, dtype=torch.float32) for x in xc])
    xp = torch.stack([torch.tensor(x, dtype=torch.float32) for x in xp])
    y = torch.stack([torch.tensor(x, dtype=torch.float32) for x in y])

    return xc, xp, y


