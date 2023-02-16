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

from model2 import * 
from dataset_perturbed import *
from statistics import mean, stdev

# cora : 400 / 5e-4 / 1e-5 /
# citeseer : 100 / 5e-4 / 1e-5 /
# pubmed : 1500 / 1e-3 / 0.0 /
# computers : 1000 / 1e-3 / 0.0 /
# cs : 1000 / 1e-3 / 0.0 /
# photo : 1000 / 1e-3 / 1e-5 /
# physics : 1000 / 1e-3 / 0.0 /

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora') 
parser.add_argument('--n_experiments', type=int, default=10)
parser.add_argument('--n_layers', type=int, default=2) 
parser.add_argument('--tau', type=float, default=0.5) 
parser.add_argument('--lr2', type=float, default=5e-3)
parser.add_argument('--wd2', type=float, default=1e-4)
parser.add_argument('--hid_dim', type=int, default=512)
parser.add_argument('--out_dim', type=int, default=512) 
parser.add_argument('--fmr', type=float, default=0.2) #0.0 #0.2
parser.add_argument('--edr', type=float, default=0.5) #0.6 #0.5
parser.add_argument('--lambd', type=float, default=1e-3) # citeseer, computer, ogbn-arxiv 5e-4
parser.add_argument('--batch', type=int, default=1024) #None
parser.add_argument('--sigma', type=float, default=None) #None
parser.add_argument('--alpha', type=float, default=None) #None
parser.add_argument('--mlp_use', type=bool, default=False)
parser.add_argument('--result_file', type=str, default="/Ours/ccc/results/Test_edge_robust")

args = parser.parse_args()
file_path = os.getcwd() + args.result_file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# newly added
results =[]
# for args.sigma in [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]:
# for args.sigma in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
for args.alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    for args.model in ['nCLNR','CLNR','GRACE','GCLNR','dCLNR']:
        if args.model in ['nCLNR','bCLNR','dCLNR','CLNR','GCLNR']:
            args.epochs = 50
            args.lr1 = 1e-3 # 1e-2
            args.wd1 = 0.0
            args.loss_type = 'ntxent'
        elif args.model in ['GRACE','gCCA-SSG']:
            args.epochs = 400
            args.lr1 = 5e-4
            args.wd1 = 0.0
            args.loss_type = 'ntxent'
        elif args.model in ['CCA-SSG','dCCA-SSG']:
            args.epochs = 50
            args.lr1 = 1e-3
            args.wd1 = 0.0
            args.loss_type = 'cca'

        eval_acc_list = []
        for exp in range(args.n_experiments):
            data, clean, train_idx, val_idx, test_idx = load(args.dataset, args.sigma, args.alpha)
            model = ContrastiveLearning(args, data, clean, device)
            model.train()
            eval_acc = model.LinearEvaluation(train_idx, val_idx, test_idx) #
            eval_acc_list.append(eval_acc.item())
            
        eval_acc_mean = round(mean(eval_acc_list),4)
        eval_acc_std = round(stdev(eval_acc_list),4)

        print('model: ' + args.model + ' done')
        #results += [[args.model, args.dataset, args.epochs, args.n_layers, args.tau, args.lr1, args.lr2, args.wd1, args.wd2, args.out_dim, args.edr, args.fmr, eval_acc_mean, eval_acc_std,args.loss_type]]#
        results += [[args.model, args.dataset, args.epochs, args.alpha, args.outlier, eval_acc_mean, eval_acc_std]]#
res = pd.DataFrame(results, columns=['model', 'dataset', 'epochs', 'alpha', 'outlier', 'acc_mean', 'acc_std'])#, 
res.to_csv(file_path + "_" + str(args.batch) + "_" + str(args.out_dim) + "_" + str(args.hid_dim) + "_" + args.dataset + ".csv", index=False) #str(args.epochs)args.model + "_" + + "_" + str(args.sigma) + + args.model + "_" 
