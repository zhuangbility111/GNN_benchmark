import torch


bits_idx = torch.tensor([8.0, 4.0, 2.0], dtype=torch.float32)


class Assigner(object):
    def __init__(self, num_layers, num_send_nodes_on_forward, num_send_nodes_on_backward):
        self.num_layers = num_layers
        self.num_send_nodes_on_forward = num_send_nodes_on_forward
        self.num_send_nodes_on_backward = num_send_nodes_on_backward
        self.node_dataformat_dict = {}
        for layer in range(num_layers):
            self.node_dataformat_dict[f"forward{layer}"] = torch.zeros(
                num_send_nodes_on_forward, dtype=torch.float32
            )
            self.node_dataformat_dict[f"backward{layer}"] = torch.zeros(
                num_send_nodes_on_backward, dtype=torch.float32
            )
        Assigner.ctx = self

    def assign_node_dataformat_randomly(self, weight, num_bits=8):
        for layer in self.node_dataformat_dict.keys():
            num_nodes = self.node_dataformat_dict[layer].size(0)
            node_dataformat_tensor = torch.multinomial(weight, num_nodes, replacement=True)
            self.node_dataformat_dict[layer].copy_(bits_idx[node_dataformat_tensor])

    def get_node_dataformat(self, layer):
        return self.node_dataformat_dict[f"{layer}"]
