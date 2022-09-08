import os
import os.path as osp
import argparse
import sys
sys.path.append('/scratch/midway3/ilgee/SelfGCon')
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch_geometric.datasets import Planetoid, Coauthor, Amazon
import torch_geometric.transforms as T

from model import * 
from aug import *
# from cluster import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GRACE')
parser.add_argument('--dataset', type=str, default='CS')
parser.add_argument('--split', type=str, default='RandomSplit')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--n_experiments', type=int, default=20)
parser.add_argument('--n_layers', type=int, default=2) 
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--proj_hid_dim', type=int, default=256)
parser.add_argument('--tau', type=float, default=0.5) 
parser.add_argument('--lr1', type=float, default=5e-4)
parser.add_argument('--lr2', type=float, default=1e-2)
parser.add_argument('--wd1', type=float, default=1e-5)
parser.add_argument('--wd2', type=float, default=1e-4)
parser.add_argument('--result_file', type=str, default="/GRACE/results/Final_accuracy")
args = parser.parse_args()

file_path = os.getcwd() + args.result_file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results =[]
for proj in ["nonlinear-hid","nonlinear","linear"]:
    if proj == "nonlinear-hid":
        fmr = 0.2
        edr = 0.2
    elif proj == "nonlinear":
        fmr = 0.3
        edr = 0.1
    elif proj == "linear":
        fmr = 0.3
        edr = 0.1
        
    def train(model, data, fmr, edr, proj):
        model.train()
        optimizer.zero_grad()
        new_data1 = random_aug(data, fmr, edr)
        new_data2 = random_aug(data, fmr, edr)
        new_data1 = new_data1.to(device)
        new_data2 = new_data2.to(device)
        z1, z2 = model(new_data1, new_data2)   
        loss = model.loss(z1, z2, layer=proj)
        loss.backward()
        optimizer.step()
        return loss.item()

    for exp in range(args.n_experiments):      
        if args.split == "PublicSplit":
            transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device)])                                                                                                          
        if args.split == "RandomSplit":
            transform = T.Compose([T.ToDevice(device), T.RandomNodeSplit(split="train_rest", num_val = 0.1, num_test = 0.8)])
        if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(root='Planetoid', name=args.dataset, transform=transform)
            data = dataset[0]
        if args.dataset in ['CS', 'Physics']:
            dataset = Coauthor("/scratch/midway3/ilgee/SelfGCon", args.dataset, transform=transform)
            data = dataset[0]
        if args.dataset in ['Computers', 'Photo']:
            dataset = Amazon("/scratch/midway3/ilgee/SelfGCon", args.dataset, transform=transform)
            data = dataset[0]

        train_idx = data.train_mask 
        val_idx = data.val_mask 
        test_idx = data.test_mask  

        in_dim = data.num_features
        hid_dim = args.channels
        proj_hid_dim = args.proj_hid_dim
        n_layers = args.n_layers
        tau = args.tau

        num_class = int(data.y.max().item()) + 1 
        N = data.num_nodes

        ##### Train GRACE model #####
        print("=== train GRACE model ===")
        model = GRACE(in_dim, hid_dim, proj_hid_dim, n_layers, tau, use_mlp = True)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
        for epoch in range(args.epochs):
            loss = train(model, data, fmr, edr, proj)
            print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss))

        embeds = model.get_embedding(data)
        train_embs = embeds[train_idx]
        val_embs = embeds[val_idx]
        test_embs = embeds[test_idx]
        
        label = data.y
        label = label.to(device)
        feat = data.x
        
        train_labels = label[train_idx]
        val_labels = label[val_idx]
        test_labels = label[test_idx]

        train_feat = feat[train_idx]
        val_feat = feat[val_idx]
        test_feat = feat[test_idx] 

        ''' Linear Evaluation '''
        logreg = LogReg(train_embs.shape[1], num_class)
        logreg = logreg.to(device)
        opt = torch.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)
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
        results += [[args.model, args.dataset, proj, args.epochs, args.n_layers, args.lr1, args.lr2, args.wd2, args.channels, args.proj_hid_dim, args.tau, edr, fmr, eval_acc.item()]]
        res1 = pd.DataFrame(results, columns=['model', 'dataset', 'proj', 'epochs', 'layers', 'lr1', 'lr2', 'wd2', 'channels', 'proj_dim', 'tau', 'edr', 'fmr', 'accuracy'])
        res1.to_csv(file_path + "_" + args.dataset + "_" + ".csv", index=False)


# visualize_umap(test_embs, test_labels.numpy())    
# visualize_tsne(test_embs, test_labels.numpy())
# visualize_pca(test_embs, test_labels.numpy(), 1, 2)
# visualize_pca(test_embs, test_labels.numpy(), 1, 3)
# visualize_pca(test_embs, test_labels.numpy(), 2, 3)

# from sklearn.metrics import silhouette_score
# from sklearn.metrics import davies_bouldin_score
# from sklearn.metrics import calinski_harabasz_score

# results2 = []

# sil = silhouette_score(test_embs,test_labels.numpy())
# dav = davies_bouldin_score(test_embs,test_labels.numpy())
# cal =calinski_harabasz_score(test_embs,test_labels.numpy())
# print(sil, dav, cal)
# # print(silhouette_score(test_logits,test_labels.numpy()))
# # print(davies_bouldin_score(test_logits,test_labels.numpy()))
# # print(calinski_harabasz_score(test_logits,test_labels.numpy()))
# file_path2 = os.getcwd() + args.embeddings
# results2 += [[args.model, args.dataset, sil, dav, cal]]
# res2 = pd.DataFrame(results2, columns=['model', 'dataset', 'silhouette', 'davies', 'c-h'])
# res2.to_csv(file_path2 + "_" + args.dataset +  ".csv", index=False)