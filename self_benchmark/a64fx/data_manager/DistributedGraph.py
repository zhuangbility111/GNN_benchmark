from torch import Tensor
from torch_sparse import SparseTensor
from .CommBuffer import CommBuffer

class DistributedGraph(object):
    def __init__(self, 
                 local_adj_t: SparseTensor,
                 remote_adj_t: SparseTensor,
                 idx_nodes_send_to_others: Tensor,
                 num_nodes_send_to_others: list,
                 num_nodes_recv_from_others: list,
                 comm_buf: CommBuffer
                  ) -> None:
        
        self.local_adj_t = local_adj_t
        self.remote_adj_t = remote_adj_t

        self.idx_nodes_send_to_others = idx_nodes_send_to_others

        self.num_nodes_send_to_others = num_nodes_send_to_others
        self.num_nodes_recv_from_others = num_nodes_recv_from_others

        self.comm_buf = comm_buf

class DistributedGraphForPre(object):
    def __init__(self, 
                 local_adj_t: SparseTensor, 
                 adj_t_pre_post_aggr_from: SparseTensor, 
                 adj_t_pre_post_aggr_to: SparseTensor,
                 pre_post_aggr_from_splits: list,
                 pre_post_aggr_to_splits: list,
                 comm_buf: CommBuffer
                 ) -> None:

        self.local_adj_t = local_adj_t
        self.adj_t_pre_post_aggr_from = adj_t_pre_post_aggr_from
        self.adj_t_pre_post_aggr_to = adj_t_pre_post_aggr_to

        self.pre_post_aggr_from_splits = pre_post_aggr_from_splits
        self.pre_post_aggr_to_splits = pre_post_aggr_to_splits

        self.comm_buf = comm_buf
        # self.in_degrees = get_in_indices(local_adj_t)

    