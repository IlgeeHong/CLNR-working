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

from model4 import * 
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
parser.add_argument('--dataset', type=str, default='PubMed') 
parser.add_argument('--n_experiments', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=2) 
parser.add_argument('--tau', type=float, default=0.5) 
parser.add_argument('--lr2', type=float, default=1e-2)
parser.add_argument('--wd2', type=float, default=1e-4)
parser.add_argument('--hid_dim', type=int, default=512)
parser.add_argument('--out_dim', type=int, default=2) 
parser.add_argument('--fmr', type=float, default=0.3) #0.0 #0.2
parser.add_argument('--edr', type=float, default=0.5) #0.6 #0.5
parser.add_argument('--lambd', type=float, default=1e-3) # citeseer, computer, ogbn-arxiv 5e-4
parser.add_argument('--batch', type=int, default=1024)
parser.add_argument('--mlp_use', type=bool, default=False)
parser.add_argument('--result_file', type=str, default="/Ours/ccc/results/plotting")

args = parser.parse_args()
file_path = os.getcwd() + args.result_file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results =[]
args.model = 'CLNR'
args.epochs = 600
args.lr1 = 1e-3
args.wd1 = 0.0
args.loss_type = 'ntxent'
    
for exp in range(args.n_experiments):
    data, train_idx, val_idx, test_idx = load(args.dataset)
    model = ContrastiveLearning(args, data, device)
    model.train()
    eval_acc = model.LinearEvaluation(train_idx, val_idx, test_idx)
    
results += [[args.model, args.dataset, args.epochs, eval_acc.item()]]
res = pd.DataFrame(results, columns=['model', 'dataset', 'epochs', 'acc'])
res.to_csv(file_path + "_" + args.dataset + ".csv", index=False)
