import sys, os
import os.path as osp
import argparse
sys.path.append('/scratch/midway3/ilgee/SelfGCon')
# import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
# import yaml

from torch_geometric.datasets import Planetoid, Coauthor, Amazon
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import add_self_loops

from model import *
from aug import *
# from cluster import *

# os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
# with open(r'config/config.yaml') as file:
#     sweep = yaml.load(file, Loader=yaml.FullLoader)

sweep_config = {'method': 'grid'}
metric = {'goal' : 'minimize' , 
          'name' : 'best_val_loss'}
parameters = {
    'n_experiments': {
        'values' : [1]
    },
    'split': {
        'values' : ['PublicSplit']
    },
    'dataset': {
        'values' : ['PubMed']
    },
    'epochs': {
        'values' : [100] ###, 20, 50
    },
    'n_layers': {
        'values' : [2] # 1
    },
    'channels': {
        'values' : [512] ###, 128, 256, 1024
    },
    'tau': {
        'values' : [0.5]
    },
    'lr1': {
        'values' : [1e-3] # (Cora) 5e-4 ###, 5e-3, 5e-4, 1e-2
    },
    'lr2': {
        'values' : [1e-2] # (Cora) 1e-3, 5e-3, 
    },
    'wd1': {
        'values' : [0] ### 1e-4, 1e-3, 1e-2
    },
    'wd2': {
        'values' : [1e-4] #, 1e-3, 1e-2
    },
    'edr': {
        'values' : [0.2] # (Cora) 0,0.1, ,0.3,0.4,0.5
    },
    'fmr': {
        'values' : [0.2] # (Cora) 0,0.1,,0.3,0.4,0.5
    },
    'use_mlp': {
        'values' : [False]
    }
}         
sweep_config['metric'] = metric
sweep_config['parameters'] = parameters

sweep_id = wandb.sweep(sweep_config, project="SelfGCon_Cora_hyperparameter_tunning")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args=None):    
    with wandb.init(config=sweep_config):
        args = wandb.config
        def train(model, data):
            model.train()
            optimizer.zero_grad()
            new_data1 = random_aug(data, args.edr, args.fmr)
            new_data2 = random_aug(data, args.edr, args.fmr)
            new_data1 = new_data1.to(device)
            new_data2 = new_data2.to(device)
            z1, z2 = model(new_data1, new_data2)   
            loss = model.loss(z1, z2)
            loss.backward()
            optimizer.step()
            return loss.item()

        # results =[]
        for exp in range(args.n_experiments): 
            if args.split == "PublicSplit":
                transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device)]) #, T.RandomNodeSplit(split="random", 
                                                                                #                   num_train_per_class = 20,
                                                                                #                   num_val = 500,
                                                                                #                   num_test = 1000)])
            if args.split == "RandomSplit":
                transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device), T.RandomNodeSplit(split="random", 
                                                                                                    num_train_per_class = 20,
                                                                                                    num_val = 160,
                                                                                                    num_test = 1280)])
            if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
                dataset = Planetoid(root='Planetoid', name=args.dataset, transform=transform)
                data = dataset[0]
            if args.dataset in ['cs', 'physics']:
                dataset = Coauthor(args.dataset, 'public', transform=transform)
                data = dataset[0]
            if args.dataset in ['Computers', 'Photo']:
                dataset = Amazon("/Users/ilgeehong/Desktop/SemGCon/", args.dataset, transform=transform)
                data = dataset[0]

            train_idx = data.train_mask 
            val_idx = data.val_mask 
            test_idx = data.test_mask  
            in_dim = data.num_features
            hid_dim = args.channels
            out_dim = args.channels
            n_layers = args.n_layers
            tau = args.tau
            num_class = int(data.y.max().item()) + 1 
            N = data.num_nodes

            ##### Train model #####
            print("=== train model ===")
            model = SelfGCon(in_dim, hid_dim, out_dim, n_layers, tau, use_mlp=args.use_mlp)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
            for epoch in range(args.epochs):
                loss = train(model, data)
                # print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss))
            
            embeds = model.get_embedding(data)

            train_embs = embeds[train_idx]
            val_embs = embeds[val_idx]
            test_embs = embeds[test_idx]

            label = data.y
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
                
            wandb.log({"val acc": best_val_acc})  
    
wandb.agent(sweep_id, main)
    