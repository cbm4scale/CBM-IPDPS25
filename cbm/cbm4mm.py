import torch
from cbm import cbm_mkl_cpp as cbm_cpp

class cbm4mm:
    '''
    CBM matrix representation and support multiplication with dense matrices:
    - Represents a binary matrix A in CBM format.
    - Supports matrix products A @ X, where X is a real-valued dense matrix. 

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
    '''
    
    def __init__(self, edge_index, edge_values, alpha=0):
        '''
        Stores matrix A in CBM format.

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
        
        # Get number of rows in input dataset 
        num_rows = max(edge_index[0].max(), edge_index[1].max()) + 1

        # Represent input dataset in CBM
        cbm_data = cbm_cpp.init(
            edge_index[0],  # row indices
            edge_index[1],  # column indices
            edge_values,    # value of nnz's
            num_rows,       # number of rows
            alpha)           # prunning param
        

        # Unpack resulting data
        delta_edge_index = torch.stack([cbm_data[0], cbm_data[1]])
        delta_values = cbm_data[2]
        self.mca_branches = cbm_data[3]
        self.mca_src_idx = cbm_data[4]
        self.mca_dst_idx = cbm_data[5] 

        # Convert matrix of deltas to CSR tensor (torch.float32)
        self.deltas = torch.sparse_coo_tensor(
            delta_edge_index, 
            delta_values, 
            (num_rows, num_rows)
        ).to(torch.float32).coalesce().to_sparse_csr()
        
        
        self.num_nodes = num_rows


    def matmul(self, x, y):
        """
        Matrix multiplication using the CBM format:

        Computes the product between the matrix of deltas ('.deltas') and a
        dense real-valued matrix 'x'. The result of this product is stored in 
        another dense real-valued matrix 'y'. Then 'y' is updated according to 
        the compression tree ('.mca_branches' '.mca_src_ptr', and '.mca_dst_idx) 
        obtained during the construction of the CBM format.

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
        cbm_cpp.s_spmm_update_csr_int32(
            self.deltas.crow_indices()[:-1].to(torch.int32),
            self.deltas.crow_indices()[1: ].to(torch.int32), 
            self.deltas.col_indices().to(torch.int32), 
            self.deltas.values().to(torch.float32), 
            x, 
            self.mca_branches.to(torch.int32), 
            self.mca_src_idx.to(torch.int32), 
            self.mca_dst_idx.to(torch.int32), 
            y)