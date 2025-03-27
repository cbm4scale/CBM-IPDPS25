from torch import int32, float32, ones, zeros, sparse_coo_tensor, sparse_csr_tensor, Tensor
from cbm.mkl4mm import mkl4mm


class mkl4dad(mkl4mm):
    '''
    Converts matrix to CSR and support multiplication with dense matrices:
    - Represents D^(-1/2) @ A @ D^(-1/2) in CSR format.
    - Supports matrix products D^(-1/2) @ A @ D^(-1/2) @ X. 

    Where:
    - A is the adjacency matrix of the dataset.
    - D is the diagonal degree matrix of A.
    - X is a dense real-valued matrix.

    Attributes:
         '.num_nodes' (int): 
            Number of rows (and columns) of the squared adjacency matrix A. 

        '.a' (torch.Tensor): 
            Sparse CSR tensor that corresponds to matrix A:
                - '.crow_indices()' (torch.tensor (dtype=torch.int32)).
                - '.col_indices()' (torch.tensor (dtype=torch.int32)).
                - '.values()'(torch.tensor (dtype=torch.float)).
    Shape:
        - .a: (num_rows, num_rows)
    '''

    def __init__(self, edge_index):
        '''
        Converts matrix D^(-1/2) @ A @ D^(-1/2) to CSR format.

        Args:
            edge_index (torch.Tensor (dtype=torch.int32)): 
                Coordinates of nonzero elements in A.

        Shape:
            - edge_index: (2, nnz(A))
            - edge_values: (nnz(A),)

        Note:
            - nnz(A) represents the number of nonzero elements in matrix A.
        '''
        self.num_nodes = edge_index.max().item() + 1
        mkl_edge_weight = self.normalize_edge_index(edge_index)
        self.a = sparse_coo_tensor(
            edge_index, 
            mkl_edge_weight, 
            (self.num_nodes, self.num_nodes)
        ).coalesce().to_sparse_csr()

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