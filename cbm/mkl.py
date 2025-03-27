import torch
from cbm import cbm_mkl_cpp as mkl_cpp

class mkl():
    '''
    CSR matrix representation and support multiplication with dense matrices.

    Attributes:
        num_nodes (int): 
            Number of rows (and columns) of the squared adjacency matrix A. 

        a (torch.Tensor): 
            Sparse CSR tensor represents the adjacency matrix in CSR format:
                - '.crow_indices()' (torch.tensor (dtype=torch.int32)).
                - '.col_indices()' (torch.tensor (dtype=torch.int32)).
                - '.values()'(torch.tensor (dtype=torch.float)).

    Shape:
        - a: (num_rows, num_rows)
          
    '''

    def __init__(self, edge_index, edge_values, d_left=None, d_right=None):
        '''
        Stores the product of matrices D1 * A * D2 in CSR format.

        Args:
            edge_index (torch.Tensor (dtype=torch.int32)): 
                Coordinates of nonzero elements in A.

            edge_values (torch.Tensor (dtype=torch.float)): 
                Values of the nonzero elements in A.
            
            d_left (torch.Tensor (dtype=torch.float), optional): 
                1D tensor containing the diagonal entries of matrix D1. 
                If not provided, D1 acts as the identity matrix.
            
            d_right (torch.Tensor (dtype=torch.float), optional): 
                1D tensor containing the diagonal entries of matrix D2. 
                If not provided, D2 as the identity matrix.

        Shape:
            - edge_index: (2, nnz(A))
            - edge_values: (nnz(A),)
            - d_left: ('num_rows',) or 'None'
            - d_right: ('num_rows',) or 'None'

        Note:
            - nnz(A) represents the number of nonzero elements in matrix A.
        '''
        # get number of rows in input dataset
        self.num_nodes = max(edge_index[0].max(), edge_index[1].max()) + 1
        
        # convert edge index to csr format
        self.a = torch.sparse_coo_tensor(
            edge_index.to(torch.int32), 
            edge_values.to(torch.float32), 
            (self.num_nodes, self.num_nodes)
        ).coalesce().to_sparse_csr()

        # scale values
        row_indices = torch.arange(self.num_nodes)
        nnz_per_row = torch.diff(self.a.crow_indices())
        nnz_row_indices = row_indices.repeat_interleave(nnz_per_row)
        scaled_values = self.a.values() 
        
        # scale rows of '.a'
        if d_left is not None:
            scaled_values *= d_left[nnz_row_indices]
        
        # scale columns of '.a'
        if d_right is not None:
            scaled_values *= d_right[self.a.col_indices()]
        
        # store row- and/or column-scaled '.a'
        if d_left is not None or d_right is not None:
            self.a = torch.sparse_csr_tensor(
                self.a.crow_indices().to(torch.int32),
                self.a.col_indices().to(torch.int32),
                scaled_values.to(torch.float), 
                (self.num_nodes, self.num_nodes)
            ).to(torch.float32)

    def matmul(self, x, y):
        """
        Matrix multiplication with CSR format:
        
        Computes the product between a CSR matrix '.a' and a dense real-valued 
        matrix 'x'. The result of this operations is stored in another dense 
        real-valued matrix 'y'.
        
        Args:
            x (pytorch.Tensor): right-hand side operand matrix.
            y (pytorch.Tensor): result matrix.
        
        Notes: 
            -This method wraps C++ code and resorts to Intel MKL sparse BLAS
            routines to speedup the matrix product between '.a' and 'x'.
               
            -Use OpenMP environment variables to control parallelism:
            (e.g. OMP_NUM_THREADS=16 OMP_PLACES=cores ...)
        """
        
        mkl_cpp.s_spmm_csr_int32(
            self.a.crow_indices()[:-1].to(torch.int32), 
            self.a.crow_indices()[1:].to(torch.int32), 
            self.a.col_indices().to(torch.int32), 
            self.a.values().to(torch.float32), 
            x, y)
