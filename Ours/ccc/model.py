import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
from torch_geometric.nn import GCNConv
from dbn import *
from aug import *

# CUDA support
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

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
            x = F.relu(self.convs[i](x, edge_index))
        x = self.convs[-1](x, edge_index)
        # x = (x - x.mean(0)) / x.std(0)
        return x

class Model(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, tau, lambd, device, method="CLNR", use_mlp=False):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
        else:
            self.backbone = MLP(in_dim, hid_dim, out_dim)
        self.tau = tau
        self.lambd = lambd
        self.method = method
        self.device = device
        self.fc1 = nn.Linear(out_dim, out_dim * 2)
        self.fc2 = nn.Linear(out_dim * 2, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)
        # bgrace
        self.fc4 = nn.Linear(out_dim, out_dim * 2)
        self.fc5 = nn.Linear(out_dim * 2, out_dim)
        self.bnh = nn.BatchNorm1d(out_dim * 2)
        self.bn = nn.BatchNorm1d(out_dim)

    def get_embedding(self, data):
        out = self.backbone(data.x, data.edge_index)
        # No projection head here
        return out.detach()

    def forward(self, data1, data2):
        # Encode the graph
        if self.method == "CCA-SSG":
            z1 = self.backbone(data1.x, data1.edge_index)
            z2 = self.backbone(data2.x, data2.edge_index)
            h1 = (z1 - z1.mean(0)) / z1.std(0)
            h2 = (z2 - z2.mean(0)) / z2.std(0)
        if self.method == "dCLNR2":
            z1 = self.backbone(data1.x, data1.edge_index)
            z2 = self.backbone(data2.x, data2.edge_index)
            dbn1 = DBN(device=z1.device, num_features=z1.shape[1], num_groups=1, dim=2, affine=False, momentum=1.)
            dbn2 = DBN(device=z2.device, num_features=z2.shape[1], num_groups=1, dim=2, affine=False, momentum=1.)
            h1 = dbn1(z1)
            h2 = dbn2(z2)
        if self.method == "bCLNR2":
            z1 = self.backbone(data1.x, data1.edge_index)
            z2 = self.backbone(data2.x, data2.edge_index)
            h1 = self.bn(z1)
            h2 = self.bn(z2)
        else:
            h1 = self.backbone(data1.x, data1.edge_index)
            h2 = self.backbone(data2.x, data2.edge_index)
        return h1, h2
        
    def projection(self, z1, z2):
        if self.method == "GRACE":
            z1 = F.elu(self.fc1(z1))
            h1 = self.fc2(z1)
            z2 = F.elu(self.fc1(z2))
            h2 = self.fc2(z2)
        # elif self.type == "bGRACE":
        #     z = F.relu(self.bnh(self.fc4(z)))
        #     h = self.bn(self.fc5(z))
        # elif self.type == "nonlinear":
        #     h = F.elu(self.fc3(z))
        # elif self.type == "linear":
        #     h = self.fc3(z)
        elif self.method == "CLNR":
            h1 = (z1 - z1.mean(0)) / z1.std(0)
            h2 = (z2 - z2.mean(0)) / z2.std(0)
        # elif self.method == "CLNR2":
        #     z = torch.vstack((z1,z2))
        #     h = (z - z.mean(0)) / z.std(0)
        #     h1, h2 = torch.split(h, [z1.shape[0],z1.shape[0]])
        elif self.method == "bCLNR":
            h1 = self.bn(z1)
            h2 = self.bn(z2)
        elif self.method == 'dCLNR':
            dbn1 = DBN(device=z1.device, num_features=z1.shape[1], num_groups=1, dim=2, affine=False, momentum=1.)
            h1 = dbn1(z1)
            dbn2 = DBN(device=z2.device, num_features=z2.shape[1], num_groups=1, dim=2, affine=False, momentum=1.)
            h2 = dbn2(z2)
        elif self.method in ["CCA-SSG","dCLNR2","bCLNR2"]:
            h1 = z1              
            h2 = z2
        return h1, h2
    
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2, indices, loss_type='ntxent'):
        if loss_type == "ntxent":
            f = lambda x: torch.exp(x / self.tau)
            if indices is not None:
                z1 = z1[indices,:]
                z2 = z2[indices,:]
            refl_sim = f(self.sim(z1, z1))
            between_sim = f(self.sim(z1, z2))
            loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        elif loss_type == "align":
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            loss = (z1-z2).norm(dim=1).pow(2).mean()
        elif loss_type == "uniform":
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            sq_pdist = torch.pdist(z1, p=2).pow(2)
            loss = sq_pdist.mul(-2).exp().mean()
        return loss

    def loss(self, z1, z2, k=None, loss_type='ntxent', mean = True):
        if k is not None:
            N = z1.shape[0]
            indices = torch.LongTensor(random.sample(range(N), k))
        else:
            indices = None
        h1, h2 = self.projection(z1, z2)
        if loss_type == "ntxent":
            l1 = self.semi_loss(h1, h2, indices, loss_type)
            l2 = self.semi_loss(h2, h1, indices, loss_type)
            ret = (l1 + l2) * 0.5
            ret = ret.mean() if mean else ret.sum()
        elif loss_type == "align":
            ret = self.semi_loss(h1, h2, indices, loss_type)
        elif loss_type == "uniform":
            l1 = self.semi_loss(h1, h2, indices, loss_type)    
            l2 = self.semi_loss(h2, h1, indices, loss_type)    
            ret = ((l1 + l2) * 0.5).log()
        elif loss_type == "cca":
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
        self.model = args.model
        self.epochs = args.epochs
        self.fmr = args.fmr
        self.edr = args.edr
        self.lambd = args.lambd
        self.batch = args.batch
        self.loss_type = args.loss_type
        self.data = data
        self.device = device
        self.num_class = int(self.data.y.max().item()) + 1 
        self.model = Model(self.data.num_features, args.hid_dim, args.out_dim, args.n_layers, args.tau, args.lambd, self.device, self.model, args.mlp_use)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr1, weight_decay=args.wd1)
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
            z1, z2 = self.model(new_data1, new_data2)   
            loss = self.model.loss(z1, z2, self.batch, self.loss_type)
            loss.backward()
            self.optimizer.step()
            print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss))

    def uniformity(self, val_embed):
        z = F.normalize(val_embed)
        sq_pdist = torch.pdist(z, p=2).pow(2)
        return sq_pdist.mul(-2).exp().mean().log()

    def alignment(self, val_idx):
        new_data1 = random_aug(self.data,self.fmr,self.edr)
        new_data2 = random_aug(self.data,self.fmr,self.edr)
        z1 = F.normalize(self.model.get_embedding(new_data1)[val_idx])
        z2 = F.normalize(self.model.get_embedding(new_data2)[val_idx])
        return (z1-z2).norm(dim=1).pow(2).mean()

    def LinearEvaluation(self, train_idx, val_idx, test_idx):
        embeds = self.model.get_embedding(self.data)
        train_embs = embeds[train_idx]
        val_embs = embeds[val_idx]
        test_embs = embeds[test_idx]

        label = self.data.y
        label = label.to(self.device)

        train_labels = label[train_idx]
        val_labels = label[val_idx]
        test_labels = label[test_idx]

        # calculate metric
        Lu = self.uniformity(val_embs)
        La = self.alignment(val_idx)
 
        loss_fn = nn.CrossEntropyLoss()

        best_val_acc = 0
        eval_acc = 0

        for epoch in range(2000):
            self.logreg.train()
            self.opt.zero_grad()
            logits = self.logreg(train_embs)
            preds = torch.argmax(logits, dim=1)
            train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]
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
            print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc, test_acc))
        print('Linear evaluation accuracy:{:.4f}'.format(eval_acc))
        return eval_acc, Lu, La
    