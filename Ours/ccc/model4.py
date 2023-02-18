import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
from torch_geometric.nn import GCNConv
from dbn import *
from aug import *
import pdb
from torch_geometric.loader import NeighborLoader
from matplotlib import pyplot as plt
# from ogb.nodeproppred import Evaluator

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)
        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, x, _):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act_fn(x)
        x = self.layer2(x)
        return x

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hid_dim))
        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GCNConv(hid_dim, hid_dim))
            self.convs.append(GCNConv(hid_dim, out_dim))

    def forward(self, x, edge_index):
        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](x, edge_index)) #
        x = self.convs[-1](x, edge_index)
        return x

class Model(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, tau, lambd, device, model="CLNR", use_mlp=False):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
        else:
            self.backbone = MLP(in_dim, hid_dim, out_dim)
        self.tau = tau
        self.lambd = lambd
        self.model = model
        self.device = device
        self.fc1 = nn.Linear(out_dim, out_dim * 2)
        self.fc2 = nn.Linear(out_dim * 2, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def get_embedding(self, data):
        out = self.backbone(data.x, data.edge_index) 
        # No projection head here
        return out.detach()

    def forward(self, data1, data2):
        # Encode the graph
        u = self.backbone(data1.x, data1.edge_index)
        v = self.backbone(data2.x, data2.edge_index)
        return u, v
        
    def projection(self, u):
        if self.model in ["GRACE"]: #,"gCCA-SSG"
            u = F.elu(self.fc1(u))
            z = self.fc2(u)
        elif self.model in ["CCA-SSG","CLNR"]:
            z = (u - u.mean(0)) / u.std(0)
        elif self.model in ["GCLNR"]:
            u = F.elu(self.fc1(u))
            u = self.fc2(u)
            z = (u - u.mean(0)) / u.std(0)
        elif self.model in ["dCCA-SSG","dCLNR"]:
            dbn = DBN(device=u.device, num_features=u.shape[1], num_groups=1, dim=2, affine=False, momentum=1.)
            z = dbn(u)    
        elif self.model == "bCLNR":
            z = self.bn(u)
        else:
            z = u              
        return z
    
    def sim(self, z1, z2, indices):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1[indices,:].squeeze(), z2[indices,:].squeeze().T)

    def semi_loss(self, z1, z2, indices, loss_type='ntxent'):
        f = lambda x: torch.exp(x / self.tau)
        if loss_type == "ntxent":
            refl_sim = f(self.sim(z1, z1, indices))
            between_sim = f(self.sim(z1, z2, indices))   
            loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        return loss

    def loss(self, u, v, k, loss_type='ntxent', mean = True):
        if k is not None:
            N = u.shape[0]
            indices = torch.LongTensor(random.sample(range(N), k))
        else:
            indices = None
        z1, z2 = self.projection(u), self.projection(v)        
        if loss_type == "ntxent":
            l1 = self.semi_loss(z1, z2, indices, loss_type)
            l2 = self.semi_loss(z2, z1, indices, loss_type)
            ret = (l1 + l2) * 0.5
            ret = ret.mean() if mean else ret.sum()
        elif loss_type == 'cca':
            N = z1.shape[0]
            c = torch.mm(z1.T, z2)
            c1 = torch.mm(z1.T, z1)
            c2 = torch.mm(z2.T, z2)
            c = c / N
            c1 = c1 / N
            c2 = c2 / N
            loss_inv = - torch.diagonal(c).sum()
            iden = torch.tensor(np.eye(c.shape[0])).to(self.device)
            loss_dec1 = (iden - c1).pow(2).sum()
            loss_dec2 = (iden - c2).pow(2).sum()
            ret = loss_inv + self.lambd * (loss_dec1 + loss_dec2)
        return ret

class ContrastiveLearning(nn.Module):
    def __init__(self, args, data, device):
        super().__init__()
        self.dataset = args.dataset
        self.model_name = args.model
        self.epochs = args.epochs
        self.fmr = args.fmr
        self.edr = args.edr
        self.lambd = args.lambd
        self.batch = args.batch
        self.loss_type = args.loss_type
        self.data = data
        self.device = device
        self.num_class = int(self.data.y.max().item()) + 1 
        # encoder (GCN)
        self.model = Model(self.data.num_features, args.hid_dim, args.out_dim, args.n_layers, args.tau, args.lambd, self.device, self.model_name, args.mlp_use)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr1, weight_decay=args.wd1)
        # logistic regression
        self.logreg = LogReg(args.out_dim, self.num_class)
        self.logreg = self.logreg.to(self.device)
        self.opt = torch.optim.Adam(self.logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)

    def train(self):   
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            new_data1 = random_aug(self.data, self.fmr, self.edr)
            new_data2 = random_aug(self.data, self.fmr, self.edr)
            new_data1 = new_data1.to(self.device)
            new_data2 = new_data2.to(self.device)
            u, v = self.model(new_data1, new_data2)
            loss = self.model.loss(u, v, self.batch, self.loss_type)
            loss.backward()
            self.optimizer.step()

    def LinearEvaluation(self, train_idx, val_idx, test_idx):
        if self.dataset == "ogbn-arxiv":
            self.model = self.model.cpu()
            embeds = self.model.get_embedding(self.data)
            embeds = embeds.to(self.device)
        else:
            embeds = self.model.get_embedding(self.data.to(self.device))
        
        train_embs = embeds[train_idx]
        val_embs = embeds[val_idx]
        test_embs = embeds[test_idx]

        if self.dataset == "ogbn-arxiv":
            label = self.data.y.squeeze()
        else:
            label = self.data.y
        label = label.to(self.device)

        embedding = self.model.projection(self.model.get_embedding(self.data.to(self.device)))
        val_embedding = F.normalize(embedding[val_idx]).cpu()
        val_label = label[val_idx].cpu()

        plt.figure(figsize=(7,7))
        plt.xticks([])
        plt.yticks([])
        plt.scatter(val_embedding[:,0], val_embedding[:,1],s=200) #c=val_label,
        plt.title("Uniformity", fontsize = 25)
        plt.savefig('/scratch/midway3/ilgee/SelfGCon/Ours/ccc/figure/Uniformity' + '_' + str(self.dataset) + '.png')    

        plt.figure(figsize=(7,7))
        plt.xticks([])
        plt.yticks([])
        y1 = (val_label==0)
        y2 = (val_label==1)
        y3 = (val_label==2)
        y4 = (val_label==3)
        y5 = (val_label==4)
        y6 = (val_label==5)
        y7 = (val_label==6)

        X1 = val_embedding[y1]
        X2 = val_embedding[y2]
        X3 = val_embedding[y3]
        X4 = val_embedding[y4]
        X5 = val_embedding[y5]
        X6 = val_embedding[y6]
        X7 = val_embedding[y7]

        plt.scatter(X1[:,0],X1[:,1],s=200) #c=val_label,
        plt.title("Class 0", fontsize = 20)
        plt.savefig('/scratch/midway3/ilgee/SelfGCon/Ours/ccc/figure/class_0' + '_' + str(self.dataset) + '.png')    

        plt.figure(figsize=(7,7))
        plt.xticks([])
        plt.yticks([])
        plt.scatter(X2[:,0],X2[:,1],s=200) #c=val_label,
        plt.title("Class 1", fontsize = 20)
        plt.savefig('/scratch/midway3/ilgee/SelfGCon/Ours/ccc/figure/class_1' + '_' + str(self.dataset) + '.png')

        plt.figure(figsize=(7,7))
        plt.xticks([])
        plt.yticks([])
        plt.scatter(X3[:,0],X3[:,1],s=200) #c=val_label,
        plt.title("Class 2", fontsize = 20)
        plt.savefig('/scratch/midway3/ilgee/SelfGCon/Ours/ccc/figure/class_2' + '_' + str(self.dataset) + '.png')

        plt.figure(figsize=(7,7))
        plt.xticks([])
        plt.yticks([])
        plt.scatter(X4[:,0],X4[:,1],s=200) #c=val_label,
        plt.title("Class 3", fontsize = 20)
        plt.savefig('/scratch/midway3/ilgee/SelfGCon/Ours/ccc/figure/class_3' + '_' + str(self.dataset) + '.png')

        plt.figure(figsize=(7,7))
        plt.xticks([])
        plt.yticks([])
        plt.scatter(X5[:,0],X5[:,1],s=200) #c=val_label,
        plt.title("Class 4", fontsize = 20)
        plt.savefig('/scratch/midway3/ilgee/SelfGCon/Ours/ccc/figure/class_4' + '_' + str(self.dataset) + '.png')

        plt.figure(figsize=(7,7))
        plt.xticks([])
        plt.yticks([])
        plt.scatter(X6[:,0],X6[:,1],s=200) #c=val_label,
        plt.title("Class 5", fontsize = 20)
        plt.savefig('/scratch/midway3/ilgee/SelfGCon/Ours/ccc/figure/class_5' + '_' + str(self.dataset) + '.png')

        plt.figure(figsize=(7,7))
        plt.xticks([])
        plt.yticks([])
        plt.scatter(X7[:,0],X7[:,1],s=200) #c=val_label,
        plt.title("Class 6", fontsize = 20)
        plt.savefig('/scratch/midway3/ilgee/SelfGCon/Ours/ccc/figure/class_6' + '_' + str(self.dataset) + '.png')

        new_data1 = random_aug(self.data,self.fmr,self.edr)
        new_data2 = random_aug(self.data,self.fmr,self.edr)
        new_data1 = new_data1.to(self.device)
        new_data2 = new_data2.to(self.device)
        u, v = self.model(new_data1, new_data2)
        u = self.model.projection(u).detach()
        v = self.model.projection(v).detach()
        z1 = F.normalize(u)[val_idx]
        z2 = F.normalize(v)[val_idx]
        align = (z1-z2).norm(p=2, dim=1).pow(2).cpu()
        align_mean = (z1-z2).norm(p=2, dim=1).pow(2).mean().cpu()
        
        plt.figure(figsize=(7,7))
        plt.hist(align, edgecolor='black', zorder=0)
        plt.axvline(align_mean, linestyle='--', linewidth = 5, c='black', zorder=1)
        plt.ylabel('Count',size=25)
        plt.xlabel(r'$\ell_{2}$'+' Distances',size=25)
        plt.title('Alignment',size=25)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.savefig('/scratch/midway3/ilgee/SelfGCon/Ours/ccc/figure/alignment' + '_' + str(self.dataset) + '.png')

        train_labels = label[train_idx]
        val_labels = label[val_idx]
        test_labels = label[test_idx]

        loss_fn = nn.CrossEntropyLoss()       
        best_val_acc = 0
        eval_acc = 0

        for epoch in range(2000):
            self.logreg.train()
            self.opt.zero_grad()
            logits = self.logreg(train_embs)        
            loss = loss_fn(logits, train_labels)
            loss.backward()
            self.opt.step()

            self.logreg.eval()
            with torch.no_grad():
                val_logits = self.logreg(val_embs)
                test_logits = self.logreg(test_embs)

                val_preds = torch.argmax(val_logits, dim=1)
                test_preds = torch.argmax(test_logits, dim=1)

                val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
                test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    if test_acc > eval_acc:
                        eval_acc = test_acc

        return eval_acc
    

 # elif self.type == "bGRACE":
        #     z = F.relu(self.bnh(self.fc4(z)))
        #     h = self.bn(self.fc5(z))
        # elif self.type == "nonlinear":
        #     h = F.elu(self.fc3(z))
        # elif self.type == "linear":
        #     h = self.fc3(z)
        # elif self.method == "CLNR2":
        #     z = torch.vstack((z1,z2))
        #     h = (z - z.mean(0)) / z.std(0)
        #     h1, h2 = torch.split(h, [z1.shape[0],z1.shape[0]])
        # elif loss_type == "align":
        #     ret = self.semi_loss(h1, h2, indices, loss_type)
        # elif loss_type == "uniform":
        #     l1 = self.semi_loss(h1, h2, indices, loss_type)    
        #     l2 = self.semi_loss(h2, h1, indices, loss_type)    
        #     ret = ((l1 + l2) * 0.5)

# elif self.model == "CLNR":
#             z1 = (u - u.mean(0)) / u.std(0)
#             z2 = (v - v.mean(0)) / v.std(0)
#         elif self.model == "bCLNR":
#             z1 = self.bn(u)
#             z2 = self.bn(v)
#         elif self.model == 'dCLNR':
#             dbn1 = DBN(device=u.device, num_features=u.shape[1], num_groups=1, dim=2, affine=False, momentum=1.)
#             z1 = dbn1(u)
#             dbn2 = DBN(device=v.device, num_features=v.shape[1], num_groups=1, dim=2, affine=False, momentum=1.)
#             z2 = dbn2(v)
        # self.fc3 = nn.Linear(out_dim, out_dim)
        # # bgrace
        # self.fc4 = nn.Linear(out_dim, out_dim * 2)
        # self.fc5 = nn.Linear(out_dim * 2, out_dim)
        #self.bnh = nn.BatchNorm1d(out_dim * 2)
        # self.optimizer.zero_grad()
        #         new_data1 = random_aug(self.data, self.fmr, self.edr)
        #         new_data2 = random_aug(self.data, self.fmr, self.edr)
        #         new_data1 = new_data1.to(self.device)
        #         new_data2 = new_data2.to(self.device)
        #         u, v = self.model(new_data1, new_data2)   
        #         loss = self.model.loss(u, v, self.batch, self.loss_type)
        #         loss.backward()
        #         self.optimizer.step()
        #         print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss))
        # if k is not None:
        #     N = u.shape[0]
        #     indices = torch.LongTensor(random.sample(range(N), k))
        # else:
        #     indices = None