import torch


class Assigner(object):
    def __init__(self):
        pass

    @staticmethod
    def assign_node_dataformat_randomly(node_dataformat_list, num_bits=8):
        # torch.multinomial(weights, len(node_dataformat_list), replacement=True, out=node_dataformat_list)
        node_dataformat_list.fill_(float(num_bits))
