import torch
import torch.nn as nn
import torch.nn.functional as F

class GConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        self.fc.bias.data.fill_(0)

    def forward(self, x, adj):
        x = torch.spmm(adj, x)
        x = self.fc(x)
        return x

class FC(nn.Module):
    def __init__(self, in_features, out_features):
        super(FC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.dropout = 0.5
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.414)
        self.fc1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.414)
        self.fc2.bias.data.fill_(0)

    def forward(self, x, *args):
        x = self.fc1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(x)
        x = self.fc2(x)
        return x