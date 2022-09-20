import os
import os.path as osp
import argparse
import sys
# sys.path.append('/Users/ilgeehong/Desktop/SemGCon/') ###
sys.path.append('/scratch/midway3/ilgee/SelfGCon')
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from statistics import mean, stdev

from dataset import * ### dataset_cpu
from model import *
from aug import *
from cluster import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='BGRL')
parser.add_argument('--dataset', type=str, default='Photo')
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--n_experiments', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--out_dim', type=int, default=64)
parser.add_argument('--hid_dim', type=int, default=128)
parser.add_argument('--pred_hid', type=int, default=128)
parser.add_argument('--lr1', type=float, default=1e-4)
parser.add_argument('--wd1', type=float, default=1e-5)
parser.add_argument('--lr2', type=float, default=1e-2)
parser.add_argument('--wd2', type=float, default=1e-4)
parser.add_argument('--fmr1', type=float, default=0.1)
parser.add_argument('--fmr2', type=float, default=0.2)
parser.add_argument('--edr1', type=float, default=0.4)
parser.add_argument('--edr2', type=float, default=0.1)
parser.add_argument('--result_file', type=str, default="/BGRL/results/Final_accuracy") ###/BGRL
parser.add_argument('--result_file1', type=str, default="/BGRL/results/Clustering_score") ###/BGRL
args = parser.parse_args()

file_path = os.getcwd() + args.result_file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, data):
    model.train()
    optimizer.zero_grad()
    new_data1 = random_aug(data, args.fmr1, args.edr1)
    new_data2 = random_aug(data, args.fmr2, args.edr2)
    new_data1 = new_data1.to(device)
    new_data2 = new_data2.to(device)
    _, _, loss = model(new_data1, new_data2)
    loss.backward()
    optimizer.step()
    scheduler.step()
    model.update_moving_average()
    return loss.item()

results =[]
for exp in range(args.n_experiments): 
    data, train_idx, val_idx, test_idx = load(args.dataset, device)
    train_idx = data.train_mask 
    val_idx = data.val_mask 
    test_idx = data.test_mask  
    in_dim = data.num_features
    hid_dim = args.hid_dim
    out_dim = args.out_dim
    layer_config = [in_dim, hid_dim, out_dim]  
    n_layers = args.n_layers
    num_class = int(data.y.max().item()) + 1
    N = data.num_nodes
    ##### Train the BGRL model #####
    print("=== train BGRL model ===")
    model = BGRL(layer_config, args.pred_hid, args.epochs)
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr1, weight_decay= args.wd1) #W
    s = lambda epoch: epoch / 1000 if epoch < 1000 else ( 1 + np.cos((epoch-1000) * np.pi / (args.epochs - 1000))) * 0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=s)
    for epoch in range(args.epochs):
        loss = train(model, data)
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
    results += [['BGRL', args.dataset, args.lr1, args.hid_dim, args.epochs, args.edr1, args.fmr1, args.edr2, args.fmr2, eval_acc.item()]]
    res1 = pd.DataFrame(results, columns=['model', 'dataset', 'lr', 'hid_dim', 'epoch', 'edr1', 'fmr1', 'edr2', 'fmr2', 'accuracy'])
    res1.to_csv(file_path + "_" +  args.model + "_"  + args.dataset + '_' + str(args.out_dim) + ".csv", index=False)

Y = torch.Tensor.cpu(test_labels).numpy()
visualize_pca(test_embs, Y, 1, 2, file_path, args.dataset)
visualize_pca(test_embs, Y, 1, 3, file_path, args.dataset)
visualize_pca(test_embs, Y, 2, 3, file_path, args.dataset)

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

file_path = os.getcwd() + args.result_file1
results2 = []
embs = torch.Tensor.cpu(test_embs).numpy()
sil = silhouette_score(embs,Y)
dav = davies_bouldin_score(embs,Y)
cal =calinski_harabasz_score(embs,Y)
print(sil, dav, cal)
results2 += [[args.model, args.dataset, sil, dav, cal]]
res2 = pd.DataFrame(results2, columns=['model', 'dataset', 'silhouette', 'davies', 'c-h'])
res2.to_csv(file_path + "_" + args.dataset + '_' + str(args.out_dim) + ".csv", index=False) 
