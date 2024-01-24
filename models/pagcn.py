import torch
import torch.nn as nn
import torch.nn.functional as F
from .gclayer import GConv, FC

# Partial graph convolutional network (Normalized)
class PaGCN(nn.Module):
    def __init__(self, data, M, Z, nhid=16, dropout=0.5):
        super(PaGCN, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        self.gc0 = PaGConv(nfeat, nfeat, M, Z)
        self.gc1 = PaGConv(nfeat, nhid, M, Z)
        self.gc2 = GConv(nhid, nclass)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc0.reset_parameters()
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, adj, adjZ = data.features, data.adj, data.adjZ
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if data.dataset_name == "cora" or "citeseer":
            x = F.relu(self.gc0(x, adjZ))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.gc1(x, adjZ))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class PaGCN_ogb(torch.nn.Module):
    def __init__(self, data, M, Z, nhid=16, dropout=0.5):
        super(PaGCN_ogb, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes
        self.convs = torch.nn.ModuleList()
        self.convs.append(PaGConv(nfeat, nhid, M, Z))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.convs.append(GConv(nhid, nhid))
        self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.convs.append(GConv(nhid, nclass))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, adj, adjZ = data.features, data.adj, data.adjZ
        x = self.convs[0](x,adjZ)
        x = self.bns[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[1](x,adj)
        x = self.bns[1](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, adj)
        return x.log_softmax(dim=-1)


# Partial graph convolutional layer
class PaGConv(nn.Module):
    def __init__(self, in_features, out_features, M, A):
        super(PaGConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features)
        self.M = M
        self.AM = A
        self.layernorm = nn.LayerNorm(out_features, eps=1e-6)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        self.fc.bias.data.fill_(0)

    def forward(self, x, adj):
        H = torch.spmm(adj, self.M*x) * self.AM
        output = self.fc(H)
        return output
