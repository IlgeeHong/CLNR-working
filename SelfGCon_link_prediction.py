import os
import os.path as osp
import argparse
import pandas as pd
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.nn import GAE, VGAE, APPNP
from torch_geometric.transforms import RandomLinkSplit
import torch_geometric.transforms as T
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import add_self_loops

from sklearn.metrics import roc_auc_score, average_precision_score

from model import *
from aug import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='VGNAE')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--n_experiments', type=int, default=10)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--channels', type=int, default=512)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--non_linear', type=bool, default=True)
parser.add_argument('--scaling_factor', type=float, default=1.8)
parser.add_argument('--lr1', type=float, default=0.001)
parser.add_argument('--lr2', type=float, default=0.01)
parser.add_argument('--drop_rate_edge', type=float, default=0.2)
parser.add_argument('--drop_rate_feat', type=float, default=0.2)
parser.add_argument('--result_file', type=str, default="/results/SelfGCon_link_prediction")
args = parser.parse_args()

file_path = os.getcwd() + args.result_file

if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root='Planetoid', name=args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

if args.dataset in ['cs', 'physics']:
    dataset = Coauthor(args.dataset, 'public', transform=T.NormalizeFeatures())
    data = dataset[0]

if args.dataset in ['computers', 'photo']:
    dataset = Amazon(args.dataset, 'public', transform=T.NormalizeFeatures())
    data = dataset[0]

out_dim = args.channels
hid_dim = args.channels
n_layers = args.n_layers
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_dim = data.num_features
N = data.num_nodes

def compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm

def get_scores(edges_pos, edges_neg, adj_rec):
    adj_rec = adj_rec.cpu()
    # Predict on test set of edges
    preds = []
    for i in range(edges_pos.shape[1]):
        preds.append(adj_rec[edges_pos[0,i], edges_pos[1,i]].item())
    preds_neg = []
    for i in range(edges_neg.shape[1]):
        preds_neg.append(adj_rec[edges_neg[0,i], edges_neg[1,i]].item())
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

def sim(z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

def semi_loss(z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 0.5)
        refl_sim = f(sim(z1, z1))
        between_sim = f(sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

def cl_loss_fn(z1: torch.Tensor, z2: torch.Tensor, mean: bool = True):
    l1 = semi_loss(z1, z2)
    l2 = semi_loss(z2, z1)
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret
##### Train the CCA model
results =[]
# for training_rate in [0.1, 0.2, 0.4, 0.6, 0.8, 0.85]:
for training_rate in [0.85]:
    val_ratio = (1.0 - training_rate) / 3
    test_ratio = (1.0 - training_rate) / 3 * 2
    for exp in range(args.n_experiments):
        model = CCA_SSG(in_dim, hid_dim, out_dim, n_layers, use_mlp=False)
        lr1 = args.lr1
        wd1 = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr1, weight_decay=wd1)
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()

            dfr = args.drop_rate_feat
            der = args.drop_rate_edge

            new_data1 = random_aug(data, dfr, der)
            new_data2 = random_aug(data, dfr, der)

            z1, z2 = model(new_data1, new_data2)

            #c = torch.mm(z1.T, z2)
            #c1 =torch.mm(z1.T, z1)
            #c2 = torch.mm(z2.T, z2)

            #c = c / N
            #c1 = c1 / N
            #c2 = c2 / N
            loss = cl_loss_fn(z1, z2)
            #loss_inv = -torch.diagonal(c).sum()
            #iden = torch.tensor(np.eye(c.shape[0]))
            #loss_dec1 = (iden - c1).pow(2).sum()
            #loss_dec2 = (iden - c2).pow(2).sum()

            #lambd = 1e-3

            #loss = loss_inv + lambd * (loss_dec1 + loss_dec2)

            loss.backward()
            optimizer.step()

            print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))

        print("=== Evaluation ===")
        transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                    is_undirected=True, split_labels=True)
        train_data, val_data, test_data = transform(data)
        embeds = model.get_embedding(train_data)
        adj_train = to_dense_adj(train_data.edge_index, max_num_nodes=N)
        adj_train = adj_train[0]
        weight_tensor, norm = compute_loss_para(adj_train)
        logreg = LogReg(embeds.shape[1], adj_train.shape[1])
        lr2 = args.lr2
        wd2 = 1e-4
        opt = torch.optim.Adam(logreg.parameters(), lr=lr2, weight_decay=wd2)

        loss_fn = F.binary_cross_entropy
        output_activation = nn.Sigmoid()

        best_val_roc = 0
        eval_roc = 0
        best_val_ap = 0
        eval_ap = 0

        for epoch in range(args.epochs):
            logreg.train()
            opt.zero_grad()
            logits_temp = logreg(embeds)
            logits = output_activation(torch.mm(logits_temp, logits_temp.t()))
            loss = norm * loss_fn(logits.view(-1), adj_train.view(-1), weight = weight_tensor)
            loss.backward()
            opt.step()
            logreg.eval()
            with torch.no_grad():
                val_roc, val_ap = get_scores(val_data.pos_edge_label_index, val_data.neg_edge_label_index, logits)
                test_roc, test_ap = get_scores(test_data.pos_edge_label_index, test_data.neg_edge_label_index, logits)

                if val_roc >= best_val_roc:
                    best_val_roc = val_roc
                    if test_roc > eval_roc:
                        eval_roc = test_roc

                if val_ap >= best_val_ap:
                    best_val_ap = val_ap
                    if test_ap > eval_ap:
                        eval_ap = test_ap

            print('Epoch:{}, val_ap:{:.4f}, val_roc:{:4f}, test_ap:{:4f}, test_roc:{:4f}'.format(epoch, val_ap, val_roc, test_ap, test_roc))
            print('Linear evaluation AP:{:.4f}'.format(eval_ap))
            print('Linear evaluation ROC:{:.4f}'.format(eval_roc))
        results += [[exp, 'CCA', args.dataset, True, True, args.lr1, args.channels,
                                training_rate, val_ratio, test_ratio, 0, eval_roc, eval_ap, args.epochs, args.drop_rate_edge, args.drop_rate_feat]]
        res1 = pd.DataFrame(results, columns=['exp', 'model', 'dataset', 'non-linearity', 'normalize',  'lr', 'channels',
                                                'train_rate', 'val_ratio',
                                                'test_ratio', 'alpha', 'auc', 'ap', 'epoch', 'drop_edge_rate', 'drop_feat_rate'])
        res1.to_csv(file_path  +  args.model + str(args.non_linear) + "_norm" +  str(args.normalize) +  "_lr"+ str(args.lr1) +
                        '_channels' + str(args.channels) + "_dropedgerate" + str( args.drop_rate_edge) + "_dropfeatrate" + str( args.drop_rate_feat)+
                        ".csv", index=False)
