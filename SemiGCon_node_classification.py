import os
import os.path as osp
import argparse
import pandas as pd
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, Coauthor, Amazon
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import add_self_loops

from matplotlib import pyplot as plt
import umap.umap_ as umap

from model import *
from aug import *
# from aug_gae import gae_aug # delete + add

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GRACE')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--split', type=str, default='PublicSplit')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--n_experiments', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=2) #CiteSeer: 1, Rest: 2
parser.add_argument('--channels', type=int, default=512) #512
parser.add_argument('--tau', type=float, default=0.5) #
parser.add_argument('--lr1', type=float, default=1e-3) #
parser.add_argument('--lr2', type=float, default=1e-2)
parser.add_argument('--wd2', type=float, default=1e-4)
parser.add_argument('--drop_rate_edge', type=float, default=0.4)
parser.add_argument('--drop_rate_feat', type=float, default=0.4)
parser.add_argument('--result_file', type=str, default="/results/SemiGCon_node_classification_random")
args = parser.parse_args()

file_path = os.getcwd() + args.result_file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sim(x1: torch.Tensor, x2: torch.Tensor):
    z1 = F.normalize(x1)
    z2 = F.normalize(x2)
    return torch.mm(z1, z2.t())

def semi_loss(z1: torch.Tensor, z2: torch.Tensor, pos_idx):
    f = lambda x: torch.exp(x / args.tau) 
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(
        (1/(2*num_per_class-1))*(between_sim[pos_idx].reshape(N,num_per_class).sum(1) + refl_sim[pos_idx].reshape(N,num_per_class).sum(1) - between_sim.diag())
        / (between_sim.sum(1) + refl_sim.sum(1) - refl_sim.diag()))

def cl_loss_fn(z1: torch.Tensor, z2: torch.Tensor, pos_idx):
    l1 = semi_loss(z1, z2, pos_idx)
    l2 = semi_loss(z2, z1, pos_idx)
    ret = (l1 + l2) * 0.5
    ret = ret.mean()

    return ret

results =[]
for exp in range(args.n_experiments): 
    if args.split == "PublicSplit":
        transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device)])#, T.RandomNodeSplit(split="random", 
                                                                         #                       num_train_per_class = 20,
                                                                         #                       num_val = 500,
                                                                         #                       num_test = 1000)])
        num_per_class  = 20
    if args.split == "SupervisedSplit":
        transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device), T.RandomNodeSplit()])

    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='Planetoid', name=args.dataset, transform=transform)
        data = dataset[0]
    if args.dataset in ['cs', 'physics']:
        dataset = Coauthor(args.dataset, 'public', transform=transform)
        data = dataset[0]
    if args.dataset in ['computers', 'photo']:
        dataset = Amazon(args.dataset, 'public', transform=transform)
        data = dataset[0]

    train_idx = data.train_mask 
    val_idx = data.val_mask 
    test_idx = data.test_mask  

    in_dim = data.num_features
    hid_dim = args.channels
    out_dim = args.channels
    n_layers = args.n_layers

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_class = int(data.y.max().item()) + 1 
    N = data.num_nodes

    class_idx = []
    for c in range(num_class):
        index = (data.y == c) * train_idx
        class_idx.append(index)
    class_idx = torch.stack(class_idx).bool()
    pos_idx = class_idx[data.y]
    # class_idx = torch.BoolTensor([class_idx])

    ##### Train the GRACE model #####
    print("=== train GRACE model ===")

    model = CCA_SSG(in_dim, hid_dim, out_dim, n_layers, use_mlp=False)
    lr1 = args.lr1
    wd1 = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr1, weight_decay=wd1)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        dfr = args.drop_rate_feat
        der = args.drop_rate_edge

        new_data1 = random_aug(data, dfr, der)
        new_data2 = random_aug(data, dfr, der)

        z1, z2 = model(new_data1, new_data2)
        # pdb.set_trace()
        loss = cl_loss_fn(z1, z2, pos_idx)
    
        loss.backward()
        optimizer.step()

        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))
        # add self loop # data.edge_index

    embeds = model.get_embedding(data)

    train_embs = embeds[train_idx]
    val_embs = embeds[val_idx]
    test_embs = embeds[test_idx]

    label = data.y
    feat = data.x

    train_labels = label[train_idx]
    val_labels = label[val_idx]
    test_labels = label[test_idx]

    train_feat = feat[train_idx]
    val_feat = feat[val_idx]
    test_feat = feat[test_idx] 

    ''' Linear Evaluation '''
    logreg = LogReg(train_embs.shape[1], num_class)
    lr2 = args.lr2
    wd2 = 1e-4
    opt = torch.optim.Adam(logreg.parameters(), lr=lr2, weight_decay=wd2)

    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    eval_acc = 0

    for epoch in range(2000):
        logreg.train()
        opt.zero_grad()
        logits = logreg(train_embs)
        preds = torch.argmax(logits, dim=1)
        train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

        logreg.eval()
        with torch.no_grad():
            val_logits = logreg(val_embs)
            test_logits = logreg(test_embs)

            val_preds = torch.argmax(val_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)

            val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                if test_acc > eval_acc:
                    eval_acc = test_acc

        print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc, test_acc))
        print('Linear evaluation accuracy:{:.4f}'.format(eval_acc))
    results += [['GRACE', args.dataset, args.epochs, args.n_layers, args.tau, args.lr1, args.lr2, args.wd2, args.channels, args.drop_rate_edge, args.drop_rate_feat, eval_acc]]
    res1 = pd.DataFrame(results, columns=['model', 'dataset', 'epochs', 'layers', 'tau', 'lr1', 'lr2', 'wd2', 'channels', 'drop_edge_rate', 'drop_feat_rate', 'accuracy'])
    res1.to_csv(file_path + "_" + args.dataset +  ".csv", index=False)

def visualize_umap(out, color, size=30, epoch=None, loss = None):
    umap_2d = umap.UMAP(n_components=2, init="random", random_state=0)
    z = umap_2d.fit_transform(out.detach().cpu().numpy())
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    scatter = plt.scatter(z[:, 0], z[:, 1], s=size, c=color, cmap="Set2")
    # produce a legend with the unique colors from the scatter
    legend1 = plt.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss:.4f}', fontsize=16)
    plt.show()

visualize_umap(test_logits, test_labels.numpy())    