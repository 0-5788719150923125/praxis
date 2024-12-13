import logging
import random
import time
from ipaddress import ip_address
from threading import Thread
from typing import Optional

import hivemind
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind import DHT
from hivemind.moe import ModuleBackend, RemoteExpert, Server, get_experts
from hivemind.moe.server.layers import name_to_block
from hivemind.moe.server.layers.custom_experts import register_expert_class
from hivemind.p2p import P2PDaemonError, P2PHandlerError
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils import BatchTensorDescriptor
from hivemind.utils.networking import log_visible_maddrs
from torch import Tensor

from praxis.modules.experts import PraxisExpert
from praxis.modules.router import PraxisMixtureOfDepths


class PraxisManagement:
    """
    A helper class, with convenience methods for Hivemind swarm management.
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig"):
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

        self.router = PraxisMixtureOfDepths(config)

        self.running = False
        self.thread = None
        self.task_manager(interval=30)

    def task_manager(self, interval):
        self.running = True
        self.thread = Thread(target=self._run_task, args=(interval,))
        self.thread.daemon = True
        self.thread.start()

    def _run_task(self, interval):
        while self.running:
            self._search_for_experts()
            time.sleep(interval)

    @property
    def local_experts(self):
        return self.active_local_experts

    @property
    def remote_experts(self):
        return self.active_remote_experts

    def get_visible_maddrs(self):
        log_visible_maddrs(self.dht.get_visible_maddrs(), only_p2p=self.use_ipfs)

    def register_expert(
        self, config: "AutoConfig", expert_cls: str = "hivemind_expert"
    ):
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

    def serve_experts(self):
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

    def handle_failure(self, expert):
        self.active_remote_experts.remove(expert)
        if expert.block.uid in self.expert_uids:
            print("removing:", expert.block.uid)
            self.expert_uids.remove(expert.block.uid)

    def _generate_random_name(self, k: int):
        return random.choice(PREFIXES[:k]) + "~" + random.choice(SUFFIXES[:k]) + ".0"

    def _generate_unique_name(self, k=3, run_once=False):
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

    def _search_for_experts(self):
        _, new_expert = self._generate_unique_name(self.pool_size, run_once=True)
        if new_expert is None:
            return
        if new_expert.uid not in self.expert_uids:
            expert = PraxisExpert(
                self.config,
                HivemindWrapper(new_expert),
                self.router,
                is_remote=True,
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

    def __init__(self, module):
        self.module = module

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __getattr__(self, name):
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

PREFIXES = [
    "doz",
    "mar",
    "bin",
    "wan",
    "sam",
    "lit",
    "sig",
    "hid",
    "fid",
    "lis",
    "sog",
    "dir",
    "wac",
    "sab",
    "wis",
    "sib",
    "rig",
    "sol",
    "dop",
    "mod",
    "fog",
    "lid",
    "hop",
    "dar",
    "dor",
    "lor",
    "hod",
    "fol",
    "rin",
    "tog",
    "sil",
    "mir",
    "hol",
    "pas",
    "lac",
    "rov",
    "liv",
    "dal",
    "sat",
    "lib",
    "tab",
    "han",
    "tic",
    "pid",
    "tor",
    "bol",
    "fos",
    "dot",
    "los",
    "dil",
    "for",
    "pil",
    "ram",
    "tir",
    "win",
    "tad",
    "bic",
    "dif",
    "roc",
    "wid",
    "bis",
    "das",
    "mid",
    "lop",
    "ril",
    "nar",
    "dap",
    "mol",
    "san",
    "loc",
    "nov",
    "sit",
    "nid",
    "tip",
    "sic",
    "rop",
    "wit",
    "nat",
    "pan",
    "min",
    "rit",
    "pod",
    "mot",
    "tam",
    "tol",
    "sav",
    "pos",
    "nap",
    "nop",
    "som",
    "fin",
    "fon",
    "ban",
    "mor",
    "wor",
    "sip",
    "ron",
    "nor",
    "bot",
    "wic",
    "soc",
    "wat",
    "dol",
    "mag",
    "pic",
    "dav",
    "bid",
    "bal",
    "tim",
    "tas",
    "mal",
    "lig",
    "siv",
    "tag",
    "pad",
    "sal",
    "div",
    "dac",
    "tan",
    "sid",
    "fab",
    "tar",
    "mon",
    "ran",
    "nis",
    "wol",
    "mis",
    "pal",
    "las",
    "dis",
    "map",
    "rab",
    "tob",
    "rol",
    "lat",
    "lon",
    "nod",
    "nav",
    "fig",
    "nom",
    "nib",
    "pag",
    "sop",
    "ral",
    "bil",
    "had",
    "doc",
    "rid",
    "moc",
    "pac",
    "rav",
    "rip",
    "fal",
    "tod",
    "til",
    "tin",
    "hap",
    "mic",
    "fan",
    "pat",
    "tac",
    "lab",
    "mog",
    "sim",
    "son",
    "pin",
    "lom",
    "ric",
    "tap",
    "fir",
    "has",
    "bos",
    "bat",
    "poc",
    "hac",
    "tid",
    "hav",
    "sap",
    "lin",
    "dib",
    "hos",
    "dab",
    "bit",
    "bar",
    "rac",
    "par",
    "lod",
    "dos",
    "bor",
    "toc",
    "hil",
    "mac",
    "tom",
    "dig",
    "fil",
    "fas",
    "mit",
    "hob",
    "har",
    "mig",
    "hin",
    "rad",
    "mas",
    "hal",
    "rag",
    "lag",
    "fad",
    "top",
    "mop",
    "hab",
    "nil",
    "nos",
    "mil",
    "fop",
    "fam",
    "dat",
    "nol",
    "din",
    "hat",
    "nac",
    "ris",
    "fot",
    "rib",
    "hoc",
    "nim",
    "lar",
    "fit",
    "wal",
    "rap",
    "sar",
    "nal",
    "mos",
    "lan",
    "don",
    "dan",
    "lad",
    "dov",
    "riv",
    "bac",
    "pol",
    "lap",
    "tal",
    "pit",
    "nam",
    "bon",
    "ros",
    "ton",
    "fod",
    "pon",
    "sov",
    "noc",
    "sor",
    "lav",
    "mat",
    "mip",
    "fip",
]

SUFFIXES = [
    "zod",
    "nec",
    "bud",
    "wes",
    "sev",
    "per",
    "sut",
    "let",
    "ful",
    "pen",
    "syt",
    "dur",
    "wep",
    "ser",
    "wyl",
    "sun",
    "ryp",
    "syx",
    "dyr",
    "nup",
    "heb",
    "peg",
    "lup",
    "dep",
    "dys",
    "put",
    "lug",
    "hec",
    "ryt",
    "tyv",
    "syd",
    "nex",
    "lun",
    "mep",
    "lut",
    "sep",
    "pes",
    "del",
    "sul",
    "ped",
    "tem",
    "led",
    "tul",
    "met",
    "wen",
    "byn",
    "hex",
    "feb",
    "pyl",
    "dul",
    "het",
    "mev",
    "rut",
    "tyl",
    "wyd",
    "tep",
    "bes",
    "dex",
    "sef",
    "wyc",
    "bur",
    "der",
    "nep",
    "pur",
    "rys",
    "reb",
    "den",
    "nut",
    "sub",
    "pet",
    "rul",
    "syn",
    "reg",
    "tyd",
    "sup",
    "sem",
    "wyn",
    "rec",
    "meg",
    "net",
    "sec",
    "mul",
    "nym",
    "tev",
    "web",
    "sum",
    "mut",
    "nyx",
    "rex",
    "teb",
    "fus",
    "hep",
    "ben",
    "mus",
    "wyx",
    "sym",
    "sel",
    "ruc",
    "dec",
    "wex",
    "syr",
    "wet",
    "dyl",
    "myn",
    "mes",
    "det",
    "bet",
    "bel",
    "tux",
    "tug",
    "myr",
    "pel",
    "syp",
    "ter",
    "meb",
    "set",
    "dut",
    "deg",
    "tex",
    "sur",
    "fel",
    "tud",
    "nux",
    "rux",
    "ren",
    "wyt",
    "nub",
    "med",
    "lyt",
    "dus",
    "neb",
    "rum",
    "tyn",
    "seg",
    "lyx",
    "pun",
    "res",
    "red",
    "fun",
    "rev",
    "ref",
    "mec",
    "ted",
    "rus",
    "bex",
    "leb",
    "dux",
    "ryn",
    "num",
    "pyx",
    "ryg",
    "ryx",
    "fep",
    "tyr",
    "tus",
    "tyc",
    "leg",
    "nem",
    "fer",
    "mer",
    "ten",
    "lus",
    "nus",
    "syl",
    "tec",
    "mex",
    "pub",
    "rym",
    "tuc",
    "fyl",
    "lep",
    "deb",
    "ber",
    "mug",
    "hut",
    "tun",
    "byl",
    "sud",
    "pem",
    "dev",
    "lur",
    "def",
    "bus",
    "bep",
    "run",
    "mel",
    "pex",
    "dyt",
    "byt",
    "typ",
    "lev",
    "myl",
    "wed",
    "duc",
    "fur",
    "fex",
    "nul",
    "luc",
    "len",
    "ner",
    "lex",
    "rup",
    "ned",
    "lec",
    "ryd",
    "lyd",
    "fen",
    "wel",
    "nyd",
    "hus",
    "rel",
    "rud",
    "nes",
    "hes",
    "fet",
    "des",
    "ret",
    "dun",
    "ler",
    "nyr",
    "seb",
    "hul",
    "ryl",
    "lud",
    "rem",
    "lys",
    "fyn",
    "wer",
    "ryc",
    "sug",
    "nys",
    "nyl",
    "lyn",
    "dyn",
    "dem",
    "lux",
    "fed",
    "sed",
    "bec",
    "mun",
    "lyr",
    "tes",
    "mud",
    "nyt",
    "byr",
    "sen",
    "weg",
    "fyr",
    "mur",
    "tel",
    "rep",
    "teg",
    "pec",
    "nel",
    "nev",
    "fes",
]
