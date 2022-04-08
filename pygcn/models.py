import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        
        self.gc2 = GraphConvolution(nhid1, nclass)
        # self.gc3 = GraphConvolution(nhid2, nhid3)
        # adding la
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # print("x1:", x)
        x = F.dropout(x, self.dropout, training=self.training)
        # print("x2:", x)
        x = self.gc2(x, adj)
        # print("x3:", x)
        # torch.set_printoptions(threshold=1000000000000000000000000000000000000000000000000000)
        # print(F.log_softmax(x, dim=1))
        # exit(0)

        # x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc3(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)

        return F.log_softmax(x, dim=1)
