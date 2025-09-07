import logging
import random
import time
import warnings
from ipaddress import ip_address
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

# Filter protobuf version warnings that come from hivemind
warnings.filterwarnings(
    "ignore",
    message=".*Protobuf gencode version.*is exactly one major version older.*",
)

import hivemind
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind import DHT
from hivemind.moe import ModuleBackend, Server, get_experts
from hivemind.moe.server.layers import name_to_block
from hivemind.moe.server.layers.custom_experts import register_expert_class
from hivemind.p2p import P2PDaemonError, P2PHandlerError
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils import BatchTensorDescriptor
from hivemind.utils.networking import log_visible_maddrs
from torch import Tensor

from praxis.orchestration import RemoteExpert
from praxis.routers import ROUTER_REGISTRY
from praxis.utils import PREFIXES, SUFFIXES

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class PraxisManagement:
    """
    A helper class, with convenience methods for Hivemind swarm management.

    When using Hivemind, there are certain limitations:

    1. All inputs to the `forward()` method of an expert must be Tensors.
    2. No inputs are allowed to be empty (None) types.
    3. All inputs must be of a constant shape.
    3. All inputs/outputs must be a part of the computation graph (i.e. returning detached aux_loss tensors is invalid).

    Essentially, Hivemind experts have static inputs/outputs - in contrast to the "dynamic" nature of Pytorch.
    """

    __version__ = "0.1.0"

    def __init__(self, config: ConfigType) -> None:
        """
        Initialize hivemind management system.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.config = config

        self.pool_size = 3
        self.expert_uids = []
        self.active_remote_experts = []

        # request = requests.get("https://api.ipify.org")
        # request.raise_for_status()

        # address = request.text
        # print(f"Received public IP address of this machine: {address}")
        # version = ip_address(address).version
        # announce_maddrs = [f"/ip{version}/{address}/tcp/0"]
        self.use_ipfs = False
        self.dht = DHT(
            initial_peers=PUBLIC_INITIAL_PEERS,
            # initial_peers=IPFS_INITIAL_PEERS + config.initial_peers,
            # initial_peers=PUBLIC_INITIAL_PEERS + config.initial_peers,
            # initial_peers=config.initial_peers,
            host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
            # announce_maddrs=announce_maddrs,
            # dht_mode="auto",
            # force_reachability="public",
            start=True,
            use_relay=True,
            use_auto_relay=True,
            use_ipfs=self.use_ipfs,
            ensure_bootstrap_success=False,
            # quic=True,
        )
        # TODO: the mere act of using this method prevents bootstrap freezing
        visible_maddrs_str = [str(a) for a in self.dht.get_visible_maddrs()]
        self.backends = {}
        self.active_local_experts = []

        router_cls = ROUTER_REGISTRY.get("mixture_of_depths")
        self.router = router_cls(
            config, experts=self.active_local_experts + self.active_remote_experts
        )

        self.running = False
        self.thread = None
        self.task_manager(interval=30)

    def task_manager(self, interval: int) -> None:
        """
        Initialize background task manager for expert discovery.

        Args:
            interval: Time interval between expert searches in seconds
        """
        self.running = True
        self.thread = Thread(target=self._run_task, args=(interval,))
        self.thread.daemon = True
        self.thread.start()

    def _run_task(self, interval: int) -> None:
        """
        Background task loop for searching experts.

        Args:
            interval: Time interval between expert searches in seconds
        """
        while self.running:
            self._search_for_experts()
            time.sleep(interval)

    @property
    def local_experts(self) -> List[nn.Module]:
        """
        Get list of active local experts.

        Returns:
            List of registered local expert modules
        """
        return self.active_local_experts

    @property
    def remote_experts(self) -> List[nn.Module]:
        """
        Get list of active remote experts.

        Returns:
            List of discovered remote expert modules
        """
        return self.active_remote_experts

    def get_visible_maddrs(self) -> None:
        """
        Log visible multiaddresses for this node.
        """
        log_visible_maddrs(self.dht.get_visible_maddrs(), only_p2p=self.use_ipfs)

    def register_expert(
        self, config: ConfigType, expert_cls: str = "hivemind_expert"
    ) -> nn.Module:
        """
        Register a new expert module with hivemind.

        Args:
            config: Model configuration
            expert_cls: Type of expert to register

        Returns:
            Registered expert module
        """
        assert expert_cls in name_to_block

        hidden_schema = BatchTensorDescriptor(
            4,
            # compression=CompressionType.QUANTILE_8BIT
        )
        attention_schema = hidden_schema
        router_weights = hidden_schema

        expert_uid, _ = self._generate_unique_name(self.pool_size)
        self.expert_uids.append(expert_uid)
        self.backends[expert_uid] = ModuleBackend(
            name=expert_uid,
            module=name_to_block[expert_cls](config),
            args_schema=(
                hidden_schema,
                attention_schema,
                router_weights,
            ),
            outputs_schema=(hidden_schema),
            optimizer=None,
            scheduler=None,
            min_batch_size=1,
            max_batch_size=8192,
        )
        module = self.backends[expert_uid].module
        self.active_local_experts.append(module)
        return module

    def serve_experts(self) -> None:
        """
        Start serving registered experts to the hivemind network.
        """
        device = self.config.device_map or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        thread = Server(
            self.dht,
            self.backends,
            num_connection_handlers=len(self.active_local_experts) * 8,
            device=device,
            stats_report_interval=None,
            update_period=30,
            expiration=None,
            start=True,
        )

    def handle_failure(self, expert: nn.Module) -> None:
        """
        Handle a remote expert failure by removing it from the active list.

        Args:
            expert: The expert module that failed
        """
        self.active_remote_experts.remove(expert)
        if expert.block.uid in self.expert_uids:
            print("removing:", expert.block.uid)
            self.expert_uids.remove(expert.block.uid)

    def _generate_random_name(self, k: int) -> str:
        """
        Generate a random expert name.

        Args:
            k: Pool size parameter affecting name variety

        Returns:
            Generated random name string
        """
        return random.choice(PREFIXES[:k]) + "~" + random.choice(SUFFIXES[:k]) + ".0"

    def _generate_unique_name(
        self, k: int = 3, run_once: bool = False
    ) -> Tuple[str, Optional[RemoteExpert]]:
        """
        Generate a unique expert name not currently in use.

        Args:
            k: Pool size parameter affecting name variety
            run_once: Whether to run search only once regardless of conflicts

        Returns:
            Tuple containing the unique name and an expert instance (or None)
        """
        attempts = 0
        MAX_ATTEMPTS = 100
        while True:
            if attempts >= MAX_ATTEMPTS:
                self.pool_size += 1
                print(
                    f"Exceeded max attempts when trying to generate a unique expert name. "
                    f"Expanding the pool size to: {self.pool_size}"
                )
                attempts = 0  # Reset attempts after increasing pool size
            new_name = self._generate_random_name(self.pool_size)
            attempts += 1
            if new_name in self.expert_uids:
                continue
            new_expert = get_experts(self.dht, [new_name])[0]
            if isinstance(new_expert, RemoteExpert) and not run_once:
                continue
            else:
                return new_name, new_expert

    def _search_for_experts(self) -> None:
        """
        Search the hivemind network for additional experts.
        """
        _, new_expert = self._generate_unique_name(self.pool_size, run_once=True)
        if new_expert is None:
            return
        if new_expert.uid not in self.expert_uids:
            expert = RemoteExpert(
                self.config,
                HivemindWrapper(new_expert),
                self.router,
            )
            self.active_remote_experts.append(expert)
            self.expert_uids.append(new_expert.uid)
            uid = new_expert.uid.split(".")[0]
            messages = [
                f"{uid} has joined the swarm!",
                f"Please welcome {uid} into the fold!",
                f"{uid} slid into the hivemind!",
            ]
            print(random.choice(messages))


class HivemindWrapper:
    """
    Ensures that gradients are not computed for remote experts.
    """

    def __init__(self, module: RemoteExpert) -> None:
        """
        Initialize wrapper for remote expert.

        Args:
            module: Remote expert to wrap
        """
        self.module = module

    @torch.no_grad()
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward pass wrapper that ensures no gradients are computed.

        Args:
            *args: Positional arguments to pass to the wrapped module
            **kwargs: Keyword arguments to pass to the wrapped module

        Returns:
            Output from the wrapped module's forward method
        """
        return self.module.forward(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call operator that invokes the forward method.

        Args:
            *args: Positional arguments to pass to forward
            **kwargs: Keyword arguments to pass to forward

        Returns:
            Output from the forward method
        """
        return self.forward(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """
        Attribute access that falls through to the wrapped module.

        Args:
            name: Attribute name to access

        Returns:
            The requested attribute from the wrapped module
        """
        return getattr(self.module, name)


PUBLIC_INITIAL_PEERS = [
    # IPv4 DNS addresses
    "/dns/bootstrap1.petals.dev/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY",
    "/dns/bootstrap2.petals.dev/tcp/31338/p2p/QmQGTqmM7NKjV6ggU1ZCap8zWiyKR89RViDXiqehSiCpY5",
    # IPv6 DNS addresses
    "/dns6/bootstrap1.petals.dev/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY",
    "/dns6/bootstrap2.petals.dev/tcp/31338/p2p/QmQGTqmM7NKjV6ggU1ZCap8zWiyKR89RViDXiqehSiCpY5",
    # Reserved IPs
    "/ip4/159.89.214.152/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY",
    "/ip4/159.203.156.48/tcp/31338/p2p/QmQGTqmM7NKjV6ggU1ZCap8zWiyKR89RViDXiqehSiCpY5",
]

IPFS_INITIAL_PEERS = [
    # From running `ipfs bootstrap list` with Kubo
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt",
    "/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ",
    "/ip4/104.131.131.82/udp/4001/quic-v1/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ",
    "/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN",
    "/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa",
    "/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb",
    "/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt",
    "/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ",
    "/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ",
]
