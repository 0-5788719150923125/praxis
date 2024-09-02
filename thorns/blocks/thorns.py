import torch.nn as nn
import hivemind

from hivemind import DHT
from hivemind.moe.client.expert import create_remote_experts
from hivemind.moe import Server
from hivemind.moe.expert_uid import ExpertInfo
from hivemind.moe.server.dht_handler import DHTHandlerThread, get_experts
from hivemind.moe.server import background_server
from hivemind.moe.server.dht_handler import declare_experts, get_experts
from ..layers.attention import ThornsAttention
from ..layers.mlp import ThornsMLP
from functools import partial
from contextlib import ExitStack

# dht = hivemind.DHT(
#     # initial_peers=["/ip4/127.0.0.1/tcp/TODO/COPYFULL_ADDRESS/FROM_ONE_OF_THE_SERVERS"],
#     initial_peers=None,
#     client_mode=False,
#     use_ipfs=True,
#     use_relay=True,
#     use_auto_relay=True,
#     start=False,
# )
# dht.run_in_background(await_ready=True)


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


class ThornsBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # for attr in dir(dht):
        #     if not attr.startswith("__"):  # Exclude dunder methods
        #         print(attr)
        # print(dht.get("initial_peers"))
        # return
        self.hid_dim = config.n_embd
        self.norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.attn = ThornsAttention(config)
        self.mlp = ThornsMLP(hid_dim=config.n_embd)
        # print(config)
        # with background_server(
        #     expert_cls="ffn",
        #     num_experts=2,
        #     device="cpu",
        #     hidden_dim=self.hid_dim,
        #     num_handlers=2,
        #     # custom_module_path=CUSTOM_EXPERTS_PATH,
        # ) as server:
        #     print(server)
        # dht = DHT(initial_peers=server_peer_info.addrs, start=True)
        # self.expert0, self.expert1 = create_remote_experts(
        #     [
        #         ExpertInfo(uid="expert.0", peer_id=server_peer_info.peer_id),
        #         ExpertInfo(uid="expert.1", peer_id=server_peer_info.peer_id),
        #     ],
        #     dht=dht,
        # )

        # self.exit_stack = ExitStack()
        # self.server_peer_info = self.exit_stack.enter_context(
        #     background_server(
        #         expert_cls="thorns",
        #         num_experts=2,
        #         device="cpu",
        #         hidden_dim=config.n_embd,
        #         num_handlers=2,
        #         # custom_module_path=CUSTOM_EXPERTS_PATH,
        #     )
        # )
        # self.dht = DHT(initial_peers=self.server_peer_info.addrs, start=True)
        # self.expert0, self.expert1 = create_remote_experts(
        #     [
        #         ExpertInfo(uid="expert.0", peer_id=self.server_peer_info.peer_id),
        #         ExpertInfo(uid="expert.1", peer_id=self.server_peer_info.peer_id),
        #     ],
        #     dht=self.dht,
        # )
        # print(server)
        # dht = DHT(initial_peers=server.addrs, start=True)
        # self.expert1, self.expert2 = create_remote_experts(
        #     [
        #         ExpertInfo(uid="expert.0", peer_id=server.peer_id),
        #         ExpertInfo(uid="expert.1", peer_id=server.peer_id),
        #     ],
        #     dht=dht,
        # )
        # self.exit_stack = ExitStack()
        # # self.server_peer_info = self.exit_stack.enter_context(
        # #     background_server(
        # #         expert_cls="perceptron",
        # #         num_experts=2,
        # #         device="cpu",
        # #         hidden_dim=hid_dim,
        # #         num_handlers=2,
        # #         custom_module_path=CUSTOM_EXPERTS_PATH,
        # #     )
        # # )
        # self.server = self.exit_stack.enter_context(
        #     background_server, expert_cls="thorn", num_experts=1, device="cpu", hidden_dim=config.n_embd
        # )
        # with self.server() as server:
        #     print(server)
        # self.server = Server.create(
        #     num_experts=1,
        #     expert_cls="thorn",
        #     # dht=dht,
        #     # expert_cls="ffn",
        #     hidden_dim=config.n_embd,
        #     # host_maddrs=host_maddrs if host_maddrs is not None else ["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
        #     # host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
        #     # announce_maddrs=["/ip4/127.0.0.1/tcp/4010", "/ip4/127.0.0.1/udp/4010/quic"],
        #     # initial_peers=PUBLIC_INITIAL_PEERS,
        #     use_ipfs=True,
        #     use_relay=True,
        #     use_auto_relay=True,
        #     start=False,
        # )
        # self.server.dht.run_in_background(await_ready=True)
        # self.server.run()
        # self.server.join()
        # print(self.server)
        # self.mlp = get_experts(self.server.dht, ["expert.0"])[0]
        # print(self.mlp)

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.norm(x)
        x = self.attn(x, attention_mask)
        x = residual + x
        residual = x
        x = self.norm(x)

        # with background_server(
        #     expert_cls="ffn",
        #     num_experts=2,
        #     device="cpu",
        #     hidden_dim=self.hid_dim,
        #     num_handlers=2,
        #     # custom_module_path=CUSTOM_EXPERTS_PATH,
        # ) as server:
        #     print(server)
        # dht = DHT(initial_peers=server.addrs, start=True)
        # self.expert1, self.expert2 = create_remote_experts(
        #     [
        #         ExpertInfo(uid="expert.0", peer_id=server.peer_id),
        #         ExpertInfo(uid="expert.1", peer_id=server.peer_id),
        #     ],
        #     dht=dht,
        # )
        # batch = torch.randn(1, self.hid_dim)
        # print(self.expert1(batch))

        x = self.mlp(x)
        x = residual + x
        return x
