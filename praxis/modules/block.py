import random

import hivemind
import torch
import torch.nn as nn
from hivemind import DHT
from hivemind.moe import ModuleBackend, RemoteExpert, Server, get_experts
from hivemind.moe.server.layers import name_to_block
from hivemind.utils import BatchTensorDescriptor

from .attention import PraxisAttention
from .mlp import PraxisMLP
from .router import PraxisRouter


class PraxisBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.attn = PraxisAttention(config)

        self.n_experts = config.n_experts
        self.k_best = config.k_best

        self.mlp_norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.router = PraxisRouter(
            config.n_embd,
            self.n_experts,
            self.k_best,
            config.target_temperature,
            config.annealing_steps,
        )

        experts = {}
        for i in range(self.n_experts):
            expert = name_to_block["praxis_mlp"](config)
            experts[f"expert.{i}"] = ModuleBackend(
                name=f"expert.{i}",
                module=expert,
                args_schema=(
                    BatchTensorDescriptor(
                        config.n_embd,
                    ),
                ),
                outputs_schema=BatchTensorDescriptor(
                    config.n_embd,
                ),
                max_batch_size=8192,
            )

        relay = DHTSingleton.get_instance()

        def run_server():
            server = Server(
                relay.get_dht(),
                experts,
                num_connection_handlers=4,
            )
            try:
                server.run_in_background(timeout=5.0)
            except TimeoutError as e:
                print(e)
                # TODO: fix this
                server.shutdown()
                run_server()

        run_server()

        self.dht = DHT(
            initial_peers=relay.get_visible_maddrs(),
            start=True,
            use_auto_relay=True,
            use_relay=True,
            use_ipfs=True,
            ensure_bootstrap_success=True,
        )

        dht_experts = get_experts(
            self.dht, [f"expert.{i}" for i in range(self.n_experts)]
        )
        self.experts = PraxisExpert(dht_experts, self.k_best)

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, attention_mask)
        x = residual + x
        residual = x
        x = self.mlp_norm(x)

        # expert handling
        batch_size, seq_len, input_size = x.shape
        top_k_scores, top_k_indices, balancing_loss, expert_counts, temperature = (
            self.router(x)
        )

        # Flatten x and top_k_indices for the sparse experts
        flat_x = x.reshape(-1, input_size)
        flat_top_k_indices = top_k_indices.reshape(-1, self.k_best)

        expert_outputs = self.experts(flat_x, flat_top_k_indices)
        output = expert_outputs.view(self.k_best, batch_size, seq_len, input_size)

        weighted_output = output * top_k_scores.permute(2, 0, 1).unsqueeze(-1)
        x = weighted_output.sum(dim=0)

        x = residual + x

        return x, balancing_loss, expert_counts


class PraxisExpert(nn.Module):
    def __init__(self, experts, k):
        super().__init__()
        self.experts = experts
        self.num_experts = len(experts)
        self.k = k

    def forward(self, inputs, expert_indices):
        batch_size, seq_len, input_size = (
            inputs.shape if inputs.dim() == 3 else (1, *inputs.shape)
        )
        k = self.k
        expert_indices = expert_indices.view(batch_size, seq_len, k)

        outputs = torch.zeros(batch_size * seq_len, k, input_size, device=inputs.device)

        for i in range(k):
            for expert_idx in range(self.num_experts):
                mask = expert_indices[:, :, i].reshape(-1) == expert_idx
                if mask.any():
                    expert_input = inputs.view(-1, input_size)[mask]
                    expert_output = self.experts[expert_idx](expert_input.to("cpu")).to(
                        inputs.device
                    )
                    outputs[:, i][mask] = expert_output

        return outputs.view(batch_size, seq_len, k, input_size)


# PUBLIC_INITIAL_PEERS = [
#     # IPv4 DNS addresses
#     "/dns/bootstrap1.petals.dev/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY",
#     "/dns/bootstrap2.petals.dev/tcp/31338/p2p/QmQGTqmM7NKjV6ggU1ZCap8zWiyKR89RViDXiqehSiCpY5",
#     # IPv6 DNS addresses
#     "/dns6/bootstrap1.petals.dev/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY",
#     "/dns6/bootstrap2.petals.dev/tcp/31338/p2p/QmQGTqmM7NKjV6ggU1ZCap8zWiyKR89RViDXiqehSiCpY5",
#     # Reserved IPs
#     "/ip4/159.89.214.152/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY",
#     "/ip4/159.203.156.48/tcp/31338/p2p/QmQGTqmM7NKjV6ggU1ZCap8zWiyKR89RViDXiqehSiCpY5",
# ]


class DHTSingleton:
    """
    Ensures that we only initialize the global DHT once.
    """

    _instance = None

    @staticmethod
    def get_instance():
        if DHTSingleton._instance is None:
            DHTSingleton()
        return DHTSingleton._instance

    def __init__(self):
        if DHTSingleton._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            DHTSingleton._instance = self
            self.dht = self._initialize_dht()

    def _initialize_dht(self):
        dht_kwargs = dict(
            # initial_peers=PUBLIC_INITIAL_PEERS,
            use_auto_relay=True,
            use_relay=True,
            use_ipfs=True,
            ensure_bootstrap_success=True,
            # parallel_rpc=4,
            # client_mode=False,
            # identity_path="./data/id.key",
        )

        print("Waiting for the DHT to initialize")
        # dht = DHT(start=True, daemon=True, await_ready=True, **dht_kwargs)
        dht = DHT(
            start=True,
            initial_peers=None,
            use_auto_relay=True,
            use_relay=True,
            use_ipfs=True,
        )

        return dht

    def get_visible_maddrs(self):
        return self.dht.get_visible_maddrs()

    def get_dht(self):
        return self.dht
