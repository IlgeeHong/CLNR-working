import torch
import torch.nn as nn
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

    def forward(self, x, edge_index, p):
        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](x, edge_index))
            x = F.dropout(x, p, training=self.training)
            
        x = self.convs[-1](x, edge_index)
        return x

class Model(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, tau, use_mlp=False):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
        else:
            self.backbone = MLP(in_dim, hid_dim, out_dim)
        self.tau = tau

    def get_embedding(self, data, p):
        out = self.backbone(data.x, data.edge_index, p)
        # No projection head here
        return out.detach()

    def forward(self, data1, data2, p1, p2):
        # Encode the graph
        h1 = self.backbone(data1.x, data1.edge_index, p1)
        h2 = self.backbone(data2.x, data2.edge_index, p2)
        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)
        return z1, z2
    
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2, indices):
        f = lambda x: torch.exp(x / self.tau)
        if indices is not None:
            z1 = z1[indices,:]
            z2 = z2[indices,:]
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, z1, z2, k=None, mean = True):
        if k is not None:
            N = z1.shape[0]
            indices = torch.LongTensor(random.sample(range(N), k))
        else:
            indices = None
        l1 = self.semi_loss(z1, z2, indices)
        l2 = self.semi_loss(z2, z1, indices)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret

class ContrastiveLearning(nn.Module):
    def __init__(self, args, data, device):
        super().__init__()
        self.model = args.model
        self.epochs = args.epochs
        self.p1 = args.p1
        self.p2 = args.p2
        self.batch = args.batch
        self.data = data
        self.device = device
        self.num_class = int(self.data.y.max().item()) + 1 
        self.model = Model(self.data.num_features, args.hid_dim, args.out_dim, args.n_layers, args.tau, use_mlp = args.mlp_use)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr1, weight_decay=args.wd1)
        self.logreg = LogReg(args.out_dim, self.num_class)
        self.logreg = self.logreg.to(self.device)
        self.opt = torch.optim.Adam(self.logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            new_data1 = self.data.to(self.device)
            new_data2 = self.data.to(self.device)
            z1, z2 = self.model(new_data1, new_data2, self.p1, self.p2)   
            loss = self.model.loss(z1, z2, self.batch)
            loss.backward()
            self.optimizer.step()
            print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss))

    def LinearEvaluation(self, train_idx, val_idx, test_idx):
        embeds = self.model.get_embedding(self.data.to(self.device),p=0.0)
        train_embs = embeds[train_idx]
        val_embs = embeds[val_idx]
        test_embs = embeds[test_idx]

        label = self.data.y
        label = label.to(self.device)

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
        return test_embs, test_labels, eval_acc
    