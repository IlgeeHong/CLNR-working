import os
import os.path as osp
import argparse
import sys
sys.path.append('/scratch/midway3/ilgee/SelfGCon')
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from statistics import mean, stdev

from dataset import * ### dataset dataset_cpu
from model import *
from aug import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='CLGR') 
# parser.add_argument('--model', type=str, default='GRACE') 
# parser.add_argument('--model', type=str, default='CCA_SSG') 
parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--tau', type=float, default=0.5) 
parser.add_argument('--lr1', type=float, default=1e-2)
parser.add_argument('--wd1', type=float, default=0.0) # CLGR
# parser.add_argument('--wd1', type=float, default=1e-5) # GRACE
parser.add_argument('--lr2', type=float, default=1e-2)
parser.add_argument('--wd2', type=float, default=1e-4)
parser.add_argument('--channels', type=int, default=256) # GRACE: 128, CLGR: 512
parser.add_argument('--proj_hid_dim', type=int, default=256) # GRACE
parser.add_argument('--mlp_use', type=bool, default=False)
parser.add_argument('--epochs', type=int, default= 100) # CCA_SSG:50, CLGR:200, GRACE:1500
parser.add_argument('--lambd', type=float, default=1e-3) # CCA-SSG
parser.add_argument('--n_experiments', type=int, default=3)
parser.add_argument('--result_file', type=str, default="/Ours/ablation/hyperparameter/results/hyperparam")
args = parser.parse_args()

file_path = os.getcwd() + args.result_file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, fmr, edr, data, k=None):
    model.train()
    optimizer.zero_grad()
    new_data1 = random_aug(data, fmr, edr)
    new_data2 = random_aug(data, fmr, edr)
    new_data1 = new_data1.to(device)
    new_data2 = new_data2.to(device)
    z1, z2 = model(new_data1, new_data2)   
    loss = model.loss(z1, z2)
    loss.backward()
    optimizer.step()
    return loss.item()

results =[]
for edr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    for fmr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]: 
        best_val_acc_list = []
        for exp in range(args.n_experiments):      
            data, train_idx, val_idx, test_idx = load(args.dataset, device)
            in_dim = data.num_features
            num_class = int(data.y.max().item()) + 1 
            N = data.num_nodes
            ##### Train the model #####
            print("=== train CLGR model ===")
            # model = GRACE(in_dim, args.channels, args.proj_hid_dim, args.n_layers, args.tau)
            # model = CCA_SSG(in_dim, args.channels, args.channels, args.n_layers, args.lambd, N, use_mlp=args.mlp_use)
            model = CLGR(in_dim, args.channels, args.channels, args.n_layers, args.tau, use_mlp = args.mlp_use)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
            for epoch in range(args.epochs):
                # loss = train(model, fmr, edr, data)
                loss = train(model, fmr, edr, data, k=2048) 
                # print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss))
            embeds = model.get_embedding(data)
            train_embs = embeds[train_idx]
            val_embs = embeds[val_idx]
            test_embs = embeds[test_idx]
            label = data.y.squeeze()
            label = label.to(device)
            train_labels = label[train_idx]
            val_labels = label[val_idx]
            test_labels = label[test_idx]
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
            best_val_acc_list.append(best_val_acc.item())
        best_val_acc_mean = mean(best_val_acc_list)
        results += [[args.model, args.dataset, edr, fmr, best_val_acc_mean]]
        res1 = pd.DataFrame(results, columns=['model', 'dataset', 'edge_drop_rate', 'feat_mask_rate', 'accuracy'])
        res1.to_csv(file_path + "_" +  args.model + "_"  + args.dataset + '_' + ".csv", index=False)

