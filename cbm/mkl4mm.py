from torch import int32, float32, ones, sparse_coo_tensor
from cbm import cbm_mkl_cpp as mkl

#TODO: remove edge_values

class mkl4mm:

    def __init__(self, edge_index):

        # get number of rows in input dataset
        self.num_nodes = edge_index.max().item() + 1
        
        # convert edge index to csr format
        self.a = sparse_coo_tensor(edge_index.to(int32), 
                                   ones(edge_index.size(1)), 
                                   (self.num_nodes, self.num_nodes)).to_sparse_csr()


    def matmul(self, x, y):
        """
        Matrix multiplication with CSR format:

        
        Args:
            x (pytorch.Tensor): right-hand side operand matrix.
            y (pytorch.Tensor): result matrix.
        """
        row_ptr_s = self.a.crow_indices()[:-1].to(int32)
        row_ptr_e = self.a.crow_indices()[1:].to(int32)
        col_ptr = self.a.col_indices().to(int32)
        val_ptr = self.a.values().to(float32)
        mkl.s_spmm_csr_int32(row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)