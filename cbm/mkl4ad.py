from torch import int32, float32, ones, tensor, sparse_coo_tensor, sparse_csr_tensor
from cbm.mkl4mm import mkl4mm


class mkl4ad(mkl4mm):

    def __init__(self, edge_index):
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


        
