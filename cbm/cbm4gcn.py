import torch
from cbm.cbm4mm import cbm4mm
from cbm import cbm_mkl_cpp as cbm_cpp

class cbm4gcn(cbm4mm):
    '''
    CBM matrix representation and support multiplication with dense matrices:
    - Represents matrix D^(-1/2) @ A @ D^(-1/2) in CBM, where D is a diagonal degree matrix of A.
    - Supports matrix products (D^(-1/2) @ A @ D^(-1/2)) @ X, where X is a real-valued dense matrix. 

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
        
        '.D' (torch.Tensor):
            1D tensor corresponding to the (diagonal) degree matrix of A.

    Shape:
        - deltas: (num_rows, num_rows)
        
    Note: 
        - the compression algorithm determines the remaining tensor shapes.
        - .matmul(...) method is inherited from parent class. (no need for mods).
    '''
    def __init__(self, edge_index, edge_values, alpha=0):
        super().__init__(edge_index, edge_values, alpha)

        # For GCNConv Â = D^{⁻1/2} A D^{⁻1/2}, when D is the degree matrix of A 
        num_rows = self.deltas.size()[0]
        d = torch.zeros(num_rows,1)
        x = torch.ones(num_rows,1)

        # Resort to cbm4mm to compute the outdegree
        super().matmul(x, d)

        # Compute D^{⁻1/2} and flatten it 
        self.D = (d ** (-1/2)).view(-1)

        # Get csr column indices of matrix of deltas
        column_indices = self.deltas.col_indices()
        
        # Scale columns of matrix of deltas
        new_values = self.deltas.values() * self.D[column_indices]

        # find nodes that are "not" in the mca
        missing_nodes = set(range(num_rows)) - set(self.deltas.col_indices())

        # Scale missing mca nodes (rows)
        for row_idx in missing_nodes:
            row_ptr_s = self.deltas.crow_indices()[row_idx].item()
            row_ptr_e = self.deltas.crow_indices()[row_idx + 1].item()
            new_values[row_ptr_s:row_ptr_e] *= self.D[row_idx]


        self.deltas = torch.sparse_csr_tensor(
            self.deltas.crow_indices(),
            self.deltas.col_indices(),
            new_values, 
            (num_rows, num_rows)
        ).to(torch.float32)


    def matmul(self, x, y):
        """
        Fused matrix multiplication with CBM format:

        Computes the product between the matrix of deltas ('.deltas') and a
        dense real-valued matrix 'x'. The result of this product is stored in 
        another dense real-valued matrix 'y'. Then 'y' is updated according to 
        the compression tree ('.mca_branches' '.mca_src_ptr', and '.mca_dst_idx) 
        obtained during the construction of the CBM format. During the update, 
        the rows of matrix 'y' are scaled according to the diagonal matrix '.D'.
            
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
            self.D.to(torch.float32), 
            y)
