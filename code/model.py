from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, average_precision_score, roc_auc_score
import torch
import argparse
import time
from tqdm import tqdm
import copy as cp
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DataParallel, global_mean_pool
from torch_geometric.nn import global_max_pool as gmp
from data_loader import *


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cpu',
                    help='specify cuda devices')
parser.add_argument('--dataset', type=str,
                    default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float,
                    default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float,
                    default=0.0, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=30,
                    help='maximum number of epochs')
parser.add_argument('--concat', type=bool, default=True,
                    help='whether concat news embedding and graph embedding')
parser.add_argument('--multi_gpu', type=bool,
                    default=False, help='multi-gpu mode')
parser.add_argument('--feature', type=str, default='bert',
                    help='feature type, [profile, spacy, bert, content]')
parser.add_argument('--model', type=str, default='sage',
                    help='model type, [gcn, gat, sage]')
args = parser.parse_args()
torch.manual_seed(args.seed)


def eval_func(log, loader):
    data_size = len(loader.dataset.indices)
    batch_size = loader.batch_size
    if data_size % batch_size == 0:
        size_list = [batch_size] * (data_size//batch_size)
    else:
        size_list = [batch_size] * \
            (data_size // batch_size) + [data_size % batch_size]

    assert len(log) == len(size_list)

    accuracy, f1_macro, f1_micro, precision, recall = 0, 0, 0, 0, 0

    prob_log, label_log = [], []

    for batch, size in zip(log, size_list):
        pred_y, y = batch[0].data.cpu().numpy().argmax(
            axis=1), batch[1].data.cpu().numpy().tolist()
        prob_log.extend(batch[0].data.cpu().numpy()[:, 1].tolist())
        label_log.extend(y)

        accuracy += accuracy_score(y, pred_y) * size
        f1_macro += f1_score(y, pred_y, average='macro') * size
        f1_micro += f1_score(y, pred_y, average='micro') * size
        precision += precision_score(y, pred_y, zero_division=0) * size
        recall += recall_score(y, pred_y, zero_division=0) * size

    auc = roc_auc_score(label_log, prob_log)
    ap = average_precision_score(label_log, prob_log)

    return accuracy/data_size, f1_macro/data_size, f1_micro/data_size, precision/data_size, recall/data_size, auc, ap


class Model(torch.nn.Module):
    def __init__(self, args, concat=False):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.model = args.model
        self.concat = concat

        if self.model == 'gcn':
            self.conv1 = GCNConv(self.num_features, self.nhid * 2)
            self.conv2 = GCNConv(self.nhid * 2, self.nhid * 2)
        elif self.model == 'gat':
            self.conv1 = GATConv(self.num_features, self.nhid * 2)
            self.conv2 = GATConv(self.nhid * 2, self.nhid * 2)
        elif self.model == 'sage':
            self.conv1 = SAGEConv(self.num_features, self.nhid * 2)
            self.conv2 = SAGEConv(self.nhid * 2, self.nhid * 2)

        self.fc0 = torch.nn.Linear(self.num_features, self.nhid * 2)
        self.fc1 = torch.nn.Linear(self.nhid * 4, self.nhid)
        self.fc2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        edge_attr = None

        x = F.selu(self.conv1(x, edge_index))
        x = F.selu(self.conv2(x, edge_index))
        x = F.selu(global_mean_pool(x, batch))

        news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[
                           0]] for idx in range(data.num_graphs)])
        news = F.relu(self.fc0(news))
        x = torch.cat([x, news], dim=1)
        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)

@torch.no_grad()
def compute_test(loader, verbose=False):
    model.eval()
    loss_test = 0.0
    out_log = []
    for data in loader:
        if not args.multi_gpu:
            data = data.to(args.device)
        out = model(data)
        if args.multi_gpu:
            y = torch.cat([d.y.unsqueeze(0)
                          for d in data]).squeeze().to(out.device)
        else:
            y = data.y
        if verbose:
            print(F.softmax(out, dim=1).cpu().numpy())
        out_log.append([F.softmax(out, dim=1), y])
        loss_test += F.nll_loss(out, y).item()
    return eval_func(out_log, loader), loss_test


dataset = FNNDataset(root='../data', feature=args.feature,
                     empty=False, name=args.dataset, transform=Undirected())

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features
print(args)

num_training = int(len(dataset) * 0.2)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(
    dataset, [num_training, num_val, num_test])
loader = DataLoader
train_loader = loader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = loader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = loader(test_set, batch_size=args.batch_size, shuffle=False)

model = Model(args, concat=args.concat)
if args.multi_gpu:
    model = DataParallel(model)
model = model.to(args.device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


if __name__ == '__main__':
    min_loss = 1e10
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    for epoch in tqdm(range(args.epochs)):
        loss_train = 0.0
        out_log = []
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            if not args.multi_gpu:
                data = data.to(args.device)
            out = model(data)
            if args.multi_gpu:
                y = torch.cat([d.y.unsqueeze(0)
                              for d in data]).squeeze().to(out.device)
            else:
                y = data.y
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            out_log.append([F.softmax(out, dim=1), y])
        acc_train, _, _, _, recall_train, auc_train, _ = eval_func(
            out_log, train_loader)
        [acc_val, _, _, _, recall_val, auc_val,
            _], loss_val = compute_test(val_loader)
        print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
              f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
              f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
              f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')

    [acc, f1_macro, f1_micro, precision, recall, auc,
        ap], test_loss = compute_test(test_loader, verbose=False)
    print(f'{args.dataset} {args.model} Testing Results:\n'
          f'acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, '
          f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')
