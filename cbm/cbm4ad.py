import torch
from cbm.cbm4mm import cbm4mm
from cbm import cbm_mkl_cpp as cbm_cpp

class cbm4ad(cbm4mm):
    '''
    CBM matrix representation and support multiplication with dense matrices:
    - Represents matrix A @ D^(-1/2)in CBM, where D is a diagonal degree matrix of A.
    - Supports matrix products (A @ D^(-1/2)) @ X, where X is a real-valued dense matrix. 

    Attributes:
         '.num_nodes' (int): 
            Number of rows (and columns) of the squared adjacency matrix A. 

        '.deltas' (torch.Tensor): 
            Sparse CSR tensor represents the matrix of deltas of the CBM format:
                - '.crow_indices()' (torch.tensor (dtype=torch.int32)).
                - '.col_indices()' (torch.tensor (dtype=torch.int32)).
                - '.values()'(torch.tensor (dtype=torch.float)).

        '.mca_branches' (torch.Tensor):
            1D tensor containing the number of elements in each branch of the 
            virtual node of the compression tree of the CBM format.

        '.mca_src_idx' (torch.Tensor):
            1D tensor containing the src rows of the compression tree's edges.

        '.mca_dst_idx' (torch.Tensor):
            1D tensor containing the dst rows of the compression tree's edges.

    Shape:
        - deltas: (num_rows, num_rows)
        
    Note: 
        - the compression algorithm determines the remaining tensor shapes.
        - .matmul(...) method is inherited from parent class. (no need for mods).
    '''

    def __init__(self, edge_index, edge_values, alpha=0):
        '''
        Stores matrix A @ D^(-1/2) in CBM format.

        Args:
            edge_index (torch.Tensor (dtype=torch.int32)): 
                Coordinates of nonzero elements in A.

            edge_values (torch.Tensor (dtype=torch.float)): 
                Values of the nonzero elements in A.

            alpha (int, optional):
                alpha value used by the compression algorithm.
                If not provided alpha is assumed to be 0.

        Shape:
            - edge_index: (2, nnz(A))
            - edge_values: (nnz(A),)

        Note:
            - nnz(A) represents the number of nonzero elements in matrix A.
        '''
        super().__init__(edge_index, edge_values, alpha)

        # Represents Â.D^{⁻1/2} in cbm format 
        num_rows = self.deltas.size()[0]
        d = torch.zeros(num_rows,1)
        x = torch.ones(num_rows,1)

        # Resort to cbm4mm to compute the outdegree
        super().matmul(x, d)

        # Compute D^{⁻1/2} and flattens 
        D = (d ** (-1/2)).view(-1)

        # Get csr column indices of matrix of deltas
        column_indices = self.deltas.col_indices()
        
        # Scale columns of matrix of deltas
        new_values = self.deltas.values() * D[column_indices]

        self.deltas = torch.sparse_csr_tensor(
            self.deltas.crow_indices(),
            self.deltas.col_indices(),
            new_values, 
            (num_rows, num_rows)
        ).to(torch.float32)


