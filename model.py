import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class my_model(nn.Module):
    def __init__(self, dims):
        super(my_model, self).__init__()
        self.layers1 = nn.Linear(dims[0], dims[1])    #参数非共享
        self.layers2 = nn.Linear(dims[0], dims[1])
        self.layers3 = nn.Linear(dims[1],dims[0])

    def forward(self, x, is_train=False, sigma=0.01):
        #out1 = F.relu(self.layers1(x))
        #out2 = F.relu(self.layers2(x))
        out1 = self.layers1(x)
        out2 = self.layers2(x)

        out1 = F.normalize(out1, dim=1, p=2)

        if is_train:
            out2 = F.normalize(out2, dim=1, p=2) + torch.normal(0, torch.ones_like(out2) * sigma).cuda()    #添加高斯噪声
            #out2 = F.normalize(out2, dim=1, p=2) + torch.normal(0, torch.ones_like(out2) * sigma).cpu()  # 添加高斯噪声
        else:
            out2 = F.normalize(out2, dim=1, p=2)

        #rec_x = F.relu(F.normalize(self.layers3(0.5*out1+0.5*out2)))
        rec_x = F.normalize(self.layers3(0.5 * out1 + 0.5 * out2))
        
        return out1, out2,rec_x


class GraphConvolution(nn.Module):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.dropout = dropout
        self.act = act

    def forward(self, inputs,adj):
        self.adj = adj
        x = inputs
        x = F.dropout(x, training=self.training)  # dropout层，防止过拟合
        x = torch.matmul(x, self.W)
        x = torch.sparse.mm(self.adj, x)
        outputs = self.act(x)

        return outputs