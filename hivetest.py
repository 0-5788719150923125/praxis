import torch
import hivemind
from hivemind import get_dht_time
from hivemind.dht.node import DHTNode
from hivemind.moe.client.beam_search import MoEBeamSearcher
from hivemind.moe.expert_uid import ExpertInfo, is_valid_prefix, is_valid_uid, split_uid
from hivemind.moe.server.dht_handler import declare_experts, get_experts
import random
import time

import os

import torch

from hivemind.dht import DHT
from hivemind.moe.client.expert import create_remote_experts
from hivemind.moe.expert_uid import ExpertInfo
from hivemind.moe.server import background_server

# CUSTOM_EXPERTS_PATH = os.path.join(os.path.dirname(__file__), "custom_networks.py")


def test_custom_expert(hid_dim=16):
    with background_server(
        expert_cls="ffn",
        num_experts=2,
        device="cpu",
        hidden_dim=hid_dim,
        num_handlers=2,
        # custom_module_path=CUSTOM_EXPERTS_PATH,
    ) as server_peer_info:
        dht = DHT(initial_peers=server_peer_info.addrs, start=True)
        expert0, expert1 = create_remote_experts(
            [
                ExpertInfo(uid="expert.0", peer_id=server_peer_info.peer_id),
                ExpertInfo(uid="expert.1", peer_id=server_peer_info.peer_id),
            ],
            dht=dht,
        )

        for batch_size in (1, 4):
            batch = torch.randn(batch_size, hid_dim)

            output0 = expert0(batch)
            output1 = expert1(batch)
            print(output0)
            loss = output0.sum()
            loss.backward()
            loss = output1.sum()
            loss.backward()

        return server_peer_info


if __name__ == "__main__":
    info = test_custom_expert()
    print(info)

# def test_store_get_experts(n_peers=10):
#     peers = [hivemind.DHT(start=True)]
#     initial_peers = peers[0].get_visible_maddrs()
#     peers += [
#         hivemind.DHT(initial_peers=initial_peers, start=True)
#         for _ in range(n_peers - 1)
#     ]

#     first_peer = random.choice(peers)
#     other_peer = random.choice(peers)

#     expert_uids = [f"my_expert.{i}" for i in range(50)]
#     batch_size = 10
#     for batch_start in range(0, len(expert_uids), batch_size):
#         declare_experts(
#             first_peer,
#             expert_uids[batch_start : batch_start + batch_size],
#             get_dht_time() + 30,
#         )

#     found = get_experts(other_peer, random.sample(expert_uids, 5) + ["foo", "bar"])
#     assert all(
#         res is not None for res in found[:-2]
#     ), "Could not find some existing experts"
#     assert all(res is None for res in found[-2:]), "Found non-existing experts"

#     x = torch.randn(3, 512)
#     print(found[0](x))

#     other_expert = "my_other_expert.1337"
#     declare_experts(other_peer, [other_expert], get_dht_time() + 30)
#     first_notfound, first_found = get_experts(first_peer, ["foobar", other_expert])
#     assert isinstance(first_found, hivemind.RemoteExpert)
#     assert first_found.peer_id == other_peer.peer_id
#     assert first_notfound is None

#     # test graceful shutdown
#     first_peer.shutdown()
#     other_peer.shutdown()
#     time.sleep(1.0)
#     remaining_peer1 = random.choice([peer for peer in peers if peer.is_alive()])
#     remaining_peer2 = random.choice([peer for peer in peers if peer.is_alive()])
#     assert all(
#         declare_experts(
#             remaining_peer1, ["new_expert.1"], expiration_time=get_dht_time() + 30
#         )
#     )
#     assert (
#         get_experts(remaining_peer2, ["new_expert.1"])[0].peer_id
#         == remaining_peer1.peer_id
#     )
#     print("success")


# if __name__ == "__main__":
#     test_store_get_experts()
