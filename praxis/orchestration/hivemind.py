import asyncio
import logging as logger
import os
import random
import time
from pathlib import Path
from typing import Optional

import hivemind
import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind import DHT
from hivemind.moe import ModuleBackend, RemoteExpert, Server, get_experts
from hivemind.moe.server import declare_experts
from hivemind.moe.server.layers import (
    add_custom_models_from_file,
    name_to_block,
    name_to_input,
    schedule_name_to_scheduler,
)
from hivemind.p2p import P2PDaemonError, P2PHandlerError
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils import BatchTensorDescriptor, TensorDescriptor, get_dht_time
from hivemind.utils.tensor_descr import DUMMY_BATCH_SIZE, BatchTensorDescriptor
from torch import Tensor

from praxis import PraxisConfig
from praxis.modules.experts import PraxisBlock
from praxis.modules.router import PraxisMixtureOfDepths


class PraxisServer(Server):

    @classmethod
    def create(
        cls,
        expert_uids: list = None,
        expert_cls="ffn",
        config: PraxisConfig = None,
        num_handlers=None,
        min_batch_size=1,
        max_batch_size=4096,
        device=None,
        initial_peers=(),
        compression=CompressionType.NONE,
        stats_report_interval: Optional[int] = None,
        update_period: float = 30,
        expiration: Optional[float] = None,
        *,
        start: bool,
        **kwargs,
    ) -> Server:

        assert expert_cls in name_to_block

        hidden_dim = config.num_dims
        dht = DHT(initial_peers=initial_peers, start=True, **kwargs)
        visible_maddrs_str = [str(a) for a in dht.get_visible_maddrs()]
        logger.info(
            f"Running DHT node on {visible_maddrs_str}, initial peers = {initial_peers}"
        )

        assert expert_uids is not None, "Please provide expert_uids"

        num_experts = len(expert_uids)
        num_handlers = num_handlers if num_handlers is not None else num_experts * 8
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        hidden_schema = BatchTensorDescriptor(
            config.num_dims,
        )
        attention_schema = BatchTensorDescriptor(
            1,
        )
        bit_tensor_schema = BatchTensorDescriptor(
            1,
        )

        # initialize experts
        backends = {}
        cls.experts = []
        for expert_uid in expert_uids:
            expert = name_to_block[expert_cls](config)
            backends[expert_uid] = ModuleBackend(
                name=expert_uid,
                module=expert,
                args_schema=(
                    hidden_schema,
                    attention_schema,
                    bit_tensor_schema,
                ),
                outputs_schema=(hidden_schema),
                optimizer=None,
                scheduler=None,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
            )
            cls.experts.append(backends[expert_uid].module)

        return cls(
            dht,
            backends,
            num_connection_handlers=num_handlers,
            device=device,
            stats_report_interval=stats_report_interval,
            update_period=update_period,
            expiration=expiration,
            start=start,
        )


class PraxisSwarm:
    def __init__(self, config: PraxisConfig):
        super().__init__()

        self.expert_uids = []
        [
            self.expert_uids.append(self._generate_unique_name())
            for _ in range(config.num_layers)
        ]

        server = PraxisServer.create(
            expert_uids=self.expert_uids,
            expert_cls="praxis_expert",
            start=True,
            daemon=True,
            initial_peers=PUBLIC_INITIAL_PEERS,
            host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
            config=config,
            device=config.device_map,
        )
        self.dht = server.dht
        self.experts = server.experts

    def get_experts(self):
        return self.experts

    def get_visible_maddrs(self):
        return self.dht.get_visible_maddrs()

    def handle_failure(self, expert):
        self.experts.remove(expert)
        print("removing:")
        if expert.uid in self.expert_uids:
            print(expert.uid)
            self.expert_uids.remove(expert.uid)

    def _generate_unique_name(self, k=3):
        new_name = (
            random.choice(PREFIXES[:k]) + "~" + random.choice(SUFFIXES[:k]) + ".0"
        )
        if new_name not in self.expert_uids:
            return new_name
        else:
            return self._generate_unique_name(k)

    def _search_for_experts(self, chance=0.1):
        if random.random() < chance:
            new_name = self._generate_unique_name()
            new_expert = get_experts(self.dht, [new_name])[0]
            if new_expert is not None and new_expert.uid not in self.expert_uids:
                self.experts.append(new_expert)
                self.expert_uids.append(new_expert.uid)
                print(
                    f"A new expert joined the swarm! ({new_expert.uid.split('.')[0]})"
                )


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
