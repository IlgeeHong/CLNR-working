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

from model3 import * 
from dataset import *
from statistics import mean, stdev

# cora : 400 / 5e-4 / 1e-5 /
# citeseer : 100 / 5e-4 / 1e-5 /
# pubmed : 1500 / 1e-3 / 0.0 /
# computers : 1000 / 1e-3 / 0.0 /
# cs : 1000 / 1e-3 / 0.0 /
# photo : 1000 / 1e-3 / 1e-5 /
# physics : 1000 / 1e-3 / 0.0 /

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="PubMed")
parser.add_argument('--n_experiments', type=int, default=10)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--tau', type=float, default=0.5) 
parser.add_argument('--lr2', type=float, default=1e-2) # ogbn 5e-3 ???
parser.add_argument('--wd2', type=float, default=1e-4) 
parser.add_argument('--fmr', type=float, default=0.3) #0.0 #0.2 # 0.0 #
parser.add_argument('--edr', type=float, default=0.5) #0.6 #0.5 # ogbn
parser.add_argument('--lambd', type=float, default=1e-3) # citeseer, computer, ogbn-arxiv 5e-4 
parser.add_argument('--batch', type=int, default=1024) #None
parser.add_argument('--sigma', type=float, default=None) #None
parser.add_argument('--alpha', type=float, default=None) #None
parser.add_argument('--mlp_use', type=bool, default=False)
parser.add_argument('--result_file', type=str, default="/Ours/ccc/results/FINAL2") #Final_test1

args = parser.parse_args()

file_path = os.getcwd() + args.result_file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results =[]
for args.model in ['GRACE','CCA-SSG']: #,'nCLNR','CLNR'
    for args.out_dim in [64,126,256,512]: 
        if args.model in ['nCLNR','bCLNR','dCLNR','CLNR','GCLNR']:
            args.epochs = 600
            args.hid_dim = args.out_dim        
            args.lr1 = 1e-3
            args.wd1 = 0.0
            args.loss_type = 'ntxent'
        elif args.model in ['GRACE']:
            args.epochs = 1000
            args.hid_dim = args.out_dim
            args.lr1 = 1e-3
            args.wd1 = 0.0
            args.loss_type = 'ntxent'
        elif args.model in ['CCA-SSG']:
            args.epochs = 100
            args.hid_dim = args.out_dim
            args.lr1 = 1e-3
            args.wd1 = 0.0
            args.loss_type = 'cca'                         

        eval_acc_list = []
        for exp in range(args.n_experiments):
            data, train_idx, val_idx, test_idx = load(args.dataset)
            model = ContrastiveLearning(args, data, device)
            model.train()
            eval_acc = model.LinearEvaluation(train_idx, val_idx, test_idx) #
            eval_acc_list.append(eval_acc.item())
                        
        eval_acc_mean = round(mean(eval_acc_list),4)
        eval_acc_std = round(stdev(eval_acc_list),4)
        print('model: ' + args.model + ' done')
        results += [[args.model, args.dataset, args.epochs, args.out_dim, eval_acc_mean, eval_acc_std]]
res = pd.DataFrame(results, columns=['model', 'dataset', 'epochs', 'out_dim', 'acc_mean', 'acc_std'])
res.to_csv(file_path + "_" + str(args.batch) + "_" + args.dataset +  ".csv", index=False) #"_" + args.model + 
