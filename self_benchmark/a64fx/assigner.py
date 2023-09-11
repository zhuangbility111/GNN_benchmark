import torch


bits_idx = torch.tensor([8.0, 4.0, 2.0], dtype=torch.float32)


class Assigner(object):
    def __init__(self):
        pass

    @staticmethod
    def assign_node_dataformat_randomly(num_nodes, num_bits=8):
        if num_bits == -1:
            weights = torch.tensor([1.0, 1.0, 1.0])
            node_dataformat_tensor = torch.multinomial(weights, num_nodes, replacement=True)
            node_dataformat_tensor = bits_idx[node_dataformat_tensor]
        else:
            node_dataformat_tensor = torch.ones(num_nodes, dtype=torch.float32) * num_bits
        return node_dataformat_tensor
