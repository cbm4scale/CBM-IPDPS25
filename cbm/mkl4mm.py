from torch import int32, float32, ones, sparse_coo_tensor
from cbm import cbm_mkl_cpp as mkl

class mkl4mm:
    '''
    Converts matrix to CSR and support multiplication with dense matrices:
    - Represents matrix A in CSR format.
    - Supports matrix products A @ X.

     Where:
    - A is the adjacency matrix of the dataset.
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
        Converts matrix A to CSR format.

        Args:
            edge_index (torch.Tensor (dtype=torch.int32)): 
                Coordinates of nonzero elements in A.

        Shape:
            - edge_index: (2, nnz(A))
            - edge_values: (nnz(A),)

        Note:
            - nnz(A) represents the number of nonzero elements in matrix A.
        '''
        # get number of rows in input dataset
        self.num_nodes = edge_index.max().item() + 1
        
        # convert edge index to csr format
        self.a = sparse_coo_tensor(edge_index.to(int32), 
                                   ones(edge_index.size(1)), 
                                   (self.num_nodes, self.num_nodes)).to_sparse_csr()


    def matmul(self, x, y):
        """
        Matrix multiplication using the CSR format:

        Computes the product between matrix A ('.a') and a dense real-valued 
        matrix 'x'. The result of this product is stored in matrix 'y' 
    
        Notes: 
            -This method wraps C++ code and resorts to Intel MKL sparse BLAS
            routines to speedup the matrix product between '.deltas' and 'x'.
               
            -Use OpenMP environment variables to control parallelism:
            (e.g. OMP_NUM_THREADS=16 OMP_PLACES=cores ...)
            
        Args:
            x (pytorch.Tensor (torch.float)): 
                a dense matrix corresponding to the right-hand side operand.
            y (pytorch.Tensor (torch.float)): 
                a dense matrix corresponding to the resulting matrix.

        Shapes:
            - x: ('num_rows', user-defined)
            - y: ('num_rows', len(x[1]))
        """

        row_ptr_s = self.a.crow_indices()[:-1].to(int32)
        row_ptr_e = self.a.crow_indices()[1:].to(int32)
        col_ptr = self.a.col_indices().to(int32)
        val_ptr = self.a.values().to(float32)
        mkl.s_spmm_csr_int32(row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)