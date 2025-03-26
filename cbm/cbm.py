import torch
from cbm import cbm_mkl_cpp as cbm_cpp

class cbm():
    '''
    CBM matrix representation and support multiplication with dense matrices.

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

        '.matmul' (function):
            This attribute is used to dispatch a suitable matrix mulitplication
            algorithm. The choices of matrix multiplication depends on whether
            'd_right' is 'None'.

    Shape:
        - deltas: (num_rows, num_rows)
        
    Note: 
        - the compression algorithm determines the remaining tensor shapes.
    '''

    def __init__(self, edge_index, edge_values, d_left=None, d_right=None, alpha=0):    
        '''
        Stores the product of matrices D1 * A * D2 in CBM format.

        Args:
            edge_index (torch.Tensor (dtype=torch.int32)): 
                Coordinates of nonzero elements in A.

            edge_values (torch.Tensor (dtype=torch.float)): 
                Values of the nonzero elements in A.
            
            d_left (torch.Tensor (dtype=torch.float), optional): 
                1D tensor containing the diagonal entries of matrix D1. 
                If not provided, D1 is assumed to be the identity matrix.
                For conviniency this tensor is stored in 'self.d_left'.
            
            d_right (torch.Tensor (dtype=torch.float), optional): 
                1D tensor containing the diagonal entries of matrix D2. 
                If not provided, D2 is assumed to be the identity matrix.
                This tensor can be freed after building the CBM format.

            alpha (int, optional):
                alpha value used by the compression algorithm.
                If not provided alpha is assumed to be 0.

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

        # represent input dataset in CBM
        cbm_data = cbm_cpp.init(
            edge_index[0],
            edge_index[1],      
            edge_values,
            self.num_nodes,
            alpha)

        # unpack resulting data
        new_edge_index = torch.stack([cbm_data[0], cbm_data[1]])
        new_edge_values = cbm_data[2]
        self.mca_branches = cbm_data[3]
        self.mca_src_idx = cbm_data[4]
        self.mca_dst_idx = cbm_data[5] 

        # convert matrix of deltas to CSR tensor
        self.deltas = torch.sparse_coo_tensor(
            new_edge_index.to(torch.int32), 
            new_edge_values.to(torch.float), 
            (self.num_nodes, self.num_nodes)
        ).coalesce().to_sparse_csr()

        # scale columns of '.deltas' if necessary
        if d_right is not None:
            # get column indices of '.deltas'
            column_indices = self.deltas.col_indices()
            
            # scale columns of '.deltas' according to 'd_right'
            col_scaled_values = self.deltas.values() * d_right[column_indices]

            self.deltas = torch.sparse_csr_tensor(
                self.deltas.crow_indices().to(torch.int32),
                self.deltas.col_indices().to(torch.int32),
                col_scaled_values.to(torch.float), 
                (self.num_nodes, self.num_nodes)
            ).to(torch.float32)

        # scale rows of '.deltas' if necessary
        if d_left is not None:
            self.d_left = d_left
            column_indices = self.deltas.col_indices()

            # find nodes that are "not" in the mca
            missing_nodes = set(range(self.num_nodes)) - set(column_indices)

            # get current edge values 
            row_scaled_values = self.deltas.values()

            # scale missing mca nodes (rows)
            for row_idx in missing_nodes:
                row_ptr_s = self.deltas.crow_indices()[row_idx].item()
                row_ptr_e = self.deltas.crow_indices()[row_idx + 1].item()
                row_scaled_values[row_ptr_s:row_ptr_e] *= self.d_left[row_idx]

            self.deltas = torch.sparse_csr_tensor(
                self.deltas.crow_indices().to(torch.int32),
                self.deltas.col_indices().to(torch.int32),
                row_scaled_values.to(torch.float), 
                (self.num_nodes, self.num_nodes)
            ).to(torch.float32)

        # choose suitable matmul algorithm
        if(d_left is not None):
            self.matmul = self._fused_matmul
        else:
            self.matmul = self._simple_matmul

    def _simple_matmul(self, x, y):
        """
        Simple matrix multiplication with CBM format ('.d_left' is 'None'):

        Computes the product between the matrix of deltas ('.deltas') and a
        dense real-valued matrix 'x'. The result of this product is stored in 
        another dense real-valued matrix 'y'. 'y' is then updated according to 
        the compression tree ('.mca_branches' '.mca_src_ptr', and '.mca_dst_idx) 
        obtained during the construction of the CBM format. If 'd_right' is not 
        'None', the columns of 'y' will be scaled accordingly.

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

    def _fused_matmul(self, x, y):
        """
        Fused matrix multiplication with CBM format ('.d_left' is NOT 'None'):

        Computes the product between the matrix of deltas ('.deltas') and a
        dense real-valued matrix 'x'. The result of this product is stored in 
        another dense real-valued matrix 'y'. Then 'y' is updated according to 
        the compression tree ('.mca_branches' '.mca_src_ptr', and '.mca_dst_idx) 
        obtained during the construction of the CBM format. If 'd_right' is not 
        'None', the columns of 'y' will be scaled accordingly. The same is true
        for the rows of 'y', which will be scaled by '.d_left'.
            
        Args:
            x (pytorch.Tensor (torch.float)): 
                a dense matrix corresponding to the right-hand side operand.
            y (pytorch.Tensor (torch.float)): 
                a dense matrix corresponding to the resulting matrix.

        Shapes:
            - x: ('num_rows', user-defined)
            - y: ('num_rows', len(x[1]))

        Notes: 
            -This method wraps C++ code and resorts to Intel MKL sparse BLAS
            routines to speedup the matrix product between '.deltas' and 'x'.
               
            -Use OpenMP environment variables to control parallelism:
            (e.g. OMP_NUM_THREADS=16 OMP_PLACES=cores ...)
        """
        cbm_cpp.s_spmm_fused_update_csr_int32(
            self.deltas.crow_indices()[:-1].to(torch.int32),
            self.deltas.crow_indices()[1: ].to(torch.int32),
            self.deltas.col_indices().to(torch.int32),
            self.deltas.values().to(torch.float32),
            x,
            self.mca_branches.to(torch.int32), 
            self.mca_src_idx.to(torch.int32), 
            self.mca_dst_idx.to(torch.int32),
            self.d_left.to(torch.float), 
            y)


