import torch


bits_idx = torch.tensor([8.0, 4.0, 2.0], dtype=torch.float32)


class Assigner(object):
    def __init__(
        self,
        num_bits,
        num_layers,
        assign_weight,
        assign_period,
        num_send_nodes_on_forward,
        num_send_nodes_on_backward,
    ):
        self.num_layers = num_layers
        self.num_send_nodes_on_forward = num_send_nodes_on_forward
        self.num_send_nodes_on_backward = num_send_nodes_on_backward
        self.num_bits = num_bits

        self.node_dataformat_dict = {}

        self.assign_weight = assign_weight
        self.assign_period = assign_period

        self._init_node_dataformat_dict(
            num_bits, num_layers, num_send_nodes_on_forward, num_send_nodes_on_backward
        )

        Assigner.ctx = self

    def _init_node_dataformat_dict(
        self, num_bits, num_layers, num_send_nodes_on_forward, num_send_nodes_on_backward
    ):
        # if num_bits != -1, then assign the dataformat according to the num_bits
        for layer in range(num_layers):
            if num_bits != -1:
                self.node_dataformat_dict[f"forward{layer}"] = torch.full(
                    (num_send_nodes_on_forward,), num_bits, dtype=torch.float32
                )
                self.node_dataformat_dict[f"backward{layer}"] = torch.full(
                    (num_send_nodes_on_backward,), num_bits, dtype=torch.float32
                )
            else:
                self.node_dataformat_dict[f"forward{layer}"] = torch.zeros(
                    num_send_nodes_on_forward, dtype=torch.float32
                )
                self.node_dataformat_dict[f"backward{layer}"] = torch.zeros(
                    num_send_nodes_on_backward, dtype=torch.float32
                )

    def reassign_node_dataformat(self, epoch):
        if epoch % self.assign_period == 0:
            if self.num_bits == -1:
                for layer in self.node_dataformat_dict.keys():
                    num_nodes = self.node_dataformat_dict[layer].size(0)
                    node_dataformat_tensor = torch.multinomial(
                        self.assign_weight, num_nodes, replacement=True
                    )
                    self.node_dataformat_dict[layer].copy_(bits_idx[node_dataformat_tensor])

    def get_node_dataformat(self, layer):
        return self.node_dataformat_dict[layer]
