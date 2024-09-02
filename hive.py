import torch
import hivemind
from hivemind.moe import Server
from hivemind.moe.server.dht_handler import get_experts

server = Server.create(
    num_experts=1,
    expert_cls="ffn",
    hidden_dim=64,
    use_ipfs=True,
    use_relay=True,
    use_auto_relay=True,
    start=True,
)

expert0 = get_experts(server.dht, ["expert.0"])[0]
print(expert0)

x = torch.randn(3, 512)
prediction = expert0(x)
print(prediction)

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

# dht = hivemind.DHT(
#     # initial_peers=["/ip4/127.0.0.1/tcp/TODO/COPYFULL_ADDRESS/FROM_ONE_OF_THE_SERVERS"],
#     # initial_peers=PUBLIC_INITIAL_PEERS,
#     initial_peers=[
#         "/ip4/127.0.0.1/tcp/38227/p2p/12D3KooWLFkwquyzs6ic3jChLwXJrdAiSYhSS5uL2496Lz5iRw5h"
#     ],
#     use_auto_relay=True,
#     use_ipfs=False,
#     use_relay=True,
#     client_mode=True,
#     start=True,
#     ensure_bootstrap_success=True,
# )

# # note: client_mode=True means that your peer will operate in a "client-only" mode:
# # this means that it can request other peers, but will not accept requests in return

# expert1, expert4 = hivemind.moe.get_experts(dht, ["expert.1", "expert.4"])
# assert (
#     expert1 is not None and expert4 is not None
# ), "experts not found. Please double-check initial peers"

# # generate dummy data
# x = torch.randn(3, 512)
# y = 0.01 * x.sum(dim=-1, keepdim=True)

# # local torch module
# proj_out = torch.nn.Sequential(torch.nn.Linear(512, 3))
# opt = torch.optim.SGD(proj_out.parameters(), lr=0.01)

# for i in range(100):
#     prediction = proj_out(expert1(expert4(x)))
#     loss = torch.mean(abs(prediction - y))
#     print(loss.item())
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
