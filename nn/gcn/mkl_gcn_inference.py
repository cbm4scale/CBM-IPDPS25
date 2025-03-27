from torch import empty
from torch.nn import Module, Linear


class MKLGCNInference(Module):
    def __init__(self, in_features, out_features, bias, a):
        super(MKLGCNInference, self).__init__()
        #assert isinstance(a, mkl4gcn), "the adjacency matrix should be an instance of mkl4gcn"
        self.a = a
        self.y = empty((a.num_nodes, out_features))
        self.dloss_dx = empty((a.num_nodes, out_features))  # here x is in fact XW therefore dLoss_dX -> dLoss_dXW -> shape: (num_nodes, out_features)
        self.lin = Linear(in_features, out_features, False)

    def forward(self, x, edge_index):
        # X @ W
        x = self.lin(x)
        # DAD @ XW
        self.a.matmul(x, self.y)
        return self.y
