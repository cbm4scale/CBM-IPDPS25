from torch import sparse_coo_tensor, Tensor, zeros, ones, float32
from cbm.mkl4mm import mkl4mm


class mkl4gcn(mkl4mm):

    def __init__(self, edge_index):
        self.num_nodes = edge_index.max().item() + 1
        mkl_edge_weight = self.normalize_edge_index(edge_index)
        self.a = sparse_coo_tensor(edge_index, mkl_edge_weight, size=(self.num_nodes, self.num_nodes)).to_sparse_csr()

    def degree(self, index: Tensor, num_nodes: int = None, dtype=None) -> Tensor:
        if self.num_nodes is None:
            self.num_nodes = index.max().item() + 1
        if dtype is None:
            dtype = index.dtype
        out = zeros((self.num_nodes,), dtype=dtype, device=index.device)
        one = ones((index.size(0),), dtype=out.dtype, device=out.device)
        return out.scatter_add_(0, index, one)

    def normalize_edge_index(self, edge_index: Tensor, num_nodes: int = None) -> Tensor:
        if self.num_nodes is None:
            self.num_nodes = edge_index.max().item() + 1
        row, col = edge_index[0], edge_index[1]
        deg = self.degree(col, self.num_nodes, float32)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return edge_weight