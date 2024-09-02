import asyncio
import ctypes
import multiprocessing as mp
import threading
import time

import numpy as np

# import pytest
import torch

from hivemind.dht import DHT
from hivemind.moe.client.expert import RemoteExpert, create_remote_experts
from hivemind.moe.client.moe import DUMMY, RemoteMixtureOfExperts, _RemoteCallMany
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.moe.client.switch_moe import RemoteSwitchMixtureOfExperts
from hivemind.moe.expert_uid import ExpertInfo
from hivemind.moe.server import (
    ModuleBackend,
    Server,
    background_server,
    declare_experts,
)
from hivemind.moe.server.layers import name_to_block
from hivemind.p2p.p2p_daemon_bindings.control import P2PHandlerError
from hivemind.utils import BatchTensorDescriptor, MPFuture, get_dht_time
from hivemind.moe.server.dht_handler import declare_experts, get_experts


def test_thing():
    HID_DIM = 16

    experts = {}
    for i in range(4):
        expert = name_to_block["ffn"](HID_DIM)
        experts[f"expert.{i}"] = ModuleBackend(
            name=f"expert.{i}",
            module=expert,
            # optimizer=torch.optim.Adam(expert.parameters()),
            args_schema=(BatchTensorDescriptor(HID_DIM),),
            outputs_schema=BatchTensorDescriptor(HID_DIM),
            max_batch_size=16,
        )

    dht = DHT(start=True)
    server = Server(dht, experts, num_connection_handlers=1)
    server.start()
    try:
        server.ready.wait()
        client_side_dht = DHT(initial_peers=dht.get_visible_maddrs(), start=True)

        # expert = experts.get("expert.0")
        expert = get_experts(client_side_dht, ["expert.0"])[0]
        print(expert)

        # dmoe = RemoteMixtureOfExperts(
        #     in_features=16,
        #     grid_size=(3,),
        #     dht=client_side_dht,
        #     k_best=3,
        #     uid_prefix="expert.",
        #     detect_anomalies=True,
        # )

        input = torch.randn(1, 16)

        output = expert(input)
        print(output)
        # with pytest.raises(ValueError):
        #     inf_loss.backward()

    finally:
        server.shutdown()


if __name__ == "__main__":
    test_thing()
