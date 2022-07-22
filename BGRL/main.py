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

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='BGRL')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--split', type=str, default='PublicSplit')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--n_experiments', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=2) #CiteSeer: 1, Rest: 2
parser.add_argument('--hid_dim', type=int, default=512)
parser.add_argument('--out_dim', type=int, default=256)
parser.add_argument('--pred_hid', type=int, default=512)
parser.add_argument('--lr1', type=float, default=1e-3)
parser.add_argument('--lr2', type=float, default=1e-2)
parser.add_argument('--wd2', type=float, default=1e-2)
parser.add_argument('--dre1', type=float, default=0.2)
parser.add_argument('--dre2', type=float, default=0.2)
parser.add_argument('--drf1', type=float, default=0.4)
parser.add_argument('--drf2', type=float, default=0.4)
parser.add_argument('--result_file', type=str, default="/results/BGRL_node_classification")
args = parser.parse_args()

file_path = os.getcwd() + args.result_file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, data):
    model.train()
    optimizer.zero_grad()

    new_data1 = random_aug(data, args.drf1, args.dre1)
    new_data2 = random_aug(data, args.drf2, args.dre2)

    z1, z2, loss = model(new_data1, new_data2)
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    model.update_moving_average()

    return loss.item()

results =[]
for exp in range(args.n_experiments): 
    if args.split == "PublicSplit":
        transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device)])#, T.RandomNodeSplit(split="random", 
                                                                         #                   num_train_per_class = 20,
                                                                         #                   num_val = 500,
                                                                         #                   num_test = 1000)])
    if args.split == "RandomSplit":
        transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device), T.RandomNodeSplit(split="random", 
                                                                                            num_train_per_class = 50,
                                                                                            num_val = 400,
                                                                                            num_test = 3200 )])
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='Planetoid', name=args.dataset, transform=transform)
        data = dataset[0]
    if args.dataset in ['cs', 'physics']:
        dataset = Coauthor(args.dataset, 'public', transform=transform)
        data = dataset[0]
    if args.dataset in ['Computers', 'Photo']:
        dataset = Amazon("/Users/ilgeehong/Desktop/GRACE_CCA/", args.dataset, transform=transform)
        data = dataset[0]

    train_idx = data.train_mask 
    val_idx = data.val_mask 
    test_idx = data.test_mask  

    in_dim = data.num_features
    hid_dim = args.hid_dim
    out_dim = args.out_dim
    n_layers = args.n_layers

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_class = int(data.y.max().item()) + 1 #data.num_classes
    N = data.num_nodes

    ##### Train the BGRL model #####
    print("=== train BGRL model ===")
    model = BGRL(in_dim, hid_dim, out_dim, n_layers, args.pred_hid)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr1, weight_decay= 1e-5)
    s = lambda epoch: epoch / 1000 if epoch < 1000 \
                    else ( 1 + np.cos((epoch-1000) * np.pi / (args.epochs - 1000))) * 0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = s)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = train(model, data)
        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss))
                
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
    wd2 = args.wd2
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
    results += [['BGRL', args.dataset, True, True, args.lr1, args.hid_dim, args.epochs, args.dre1, args.drf1, eval_acc]]
    res1 = pd.DataFrame(results, columns=['model', 'dataset', 'non-linearity', 'normalize',  'lr', 'hid_dim', 'epoch', 'drf1', 'dre1', 'accuracy'])
    res1.to_csv(file_path + "_" + args.dataset + ".csv", index=False)


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

# visualize_umap(test_embs, test_labels.numpy())    