from torch import int32, float32, ones, tensor, sparse_coo_tensor, sparse_csr_tensor
from cbm.mkl4mm import mkl4mm


class mkl4ad(mkl4mm):
    '''
    Converts matrix to CSR and support multiplication with dense matrices:
    - Represents A @ D^(-1/2) in CSR format.
    - Supports matrix products A @ D^(-1/2) @ X. 

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
        Converts matrix A @ D^(-1/2) to CSR format.

        Args:
            edge_index (torch.Tensor (dtype=torch.int32)): 
                Coordinates of nonzero elements in A.

        Shape:
            - edge_index: (2, nnz(A))
            - edge_values: (nnz(A),)

        Note:
            - nnz(A) represents the number of nonzero elements in matrix A.
        '''

        # get number of nodes
        self.num_nodes = edge_index.max().item() + 1    
    
        # convert edge index to csr format
        mkl_csr = sparse_coo_tensor(
            edge_index.to(int32), 
            ones(edge_index.size(1)).to(float32), 
            (self.num_nodes, self.num_nodes)
        ).to_sparse_csr()

        d = [(mkl_csr.crow_indices()[row_idx + 1] - mkl_csr.crow_indices()[row_idx]) ** (-1/2) for row_idx in range(0, len(mkl_csr.crow_indices()) - 1)]
        d = tensor(d)

        # get csr column indices of csr matrix
        column_indices = mkl_csr.col_indices()
        
        # scale columns of matrix of deltas
        new_values = mkl_csr.values() * d[column_indices]

        self.a = sparse_csr_tensor(
            mkl_csr.crow_indices().to(int32),
            mkl_csr.col_indices().to(int32),
            new_values.to(float32), 
            (self.num_nodes, self.num_nodes))


        
