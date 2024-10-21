import asyncio
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
from hivemind.moe.server.layers import name_to_block
from hivemind.p2p import P2PDaemonError, P2PHandlerError
from hivemind.utils import BatchTensorDescriptor, get_dht_time
from torch import Tensor

from praxis import PraxisConfig
from praxis.modules.experts import PraxisBlock
from praxis.modules.router import PraxisMixtureOfDepths


class PraxisHivemind(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        # self.experts = nn.ModuleList()
        self.experts = []
        self.dht = DHT(
            # initial_peers=config.initial_peers,
            initial_peers=PUBLIC_INITIAL_PEERS,
            host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
            start=True,
            use_auto_relay=True,
            use_relay=True,
            use_ipfs=False,
            ensure_bootstrap_success=True,
            daemon=True,
            await_ready=True,
            # identity_path=os.path.join(os.getcwd(), "id.key"),
        )
        schema = BatchTensorDescriptor(
            config.num_dims,
        )
        attn_schema = BatchTensorDescriptor(
            1,
        )
        bit_tensor = BatchTensorDescriptor(
            1,
        )
        backends = {}
        self.local_experts = []
        for i in range(config.num_layers):
            expert_name = self._generate_unique_name(k=10)
            self.local_experts.append(expert_name)
            expert = ModuleBackend(
                name=expert_name,
                module=name_to_block["praxis_expert"](config),
                args_schema=(
                    schema,
                    attn_schema,
                    bit_tensor,
                ),
                outputs_schema=schema,
                max_batch_size=64,  # should match the `target_batch_size`
                start=True,
                # timeout=5,
            )
            backends[expert_name] = expert
            self.experts.append(expert.module)
        server = Server(
            self.dht,
            backends,
            num_connection_handlers=4 * config.num_layers,
            device=config.device_map,
        )
        server.run_in_background(timeout=5.0)

    def get_experts(self):
        return self.experts

    def get_visible_maddrs(self):
        return self.dht.get_visible_maddrs()

    def handle_failure(self, expert):
        self.experts.remove(expert)
        print("removing:")
        if expert.uid in self.local_experts:
            print(expert.uid)
            self.local_experts.remove(expert.uid)

    def _generate_unique_name(self, k=100):
        new_name = (
            random.choice(PREFIXES[:3]) + "~" + random.choice(SUFFIXES[:3]) + ".0"
        )
        if new_name not in self.local_experts:
            return new_name
        else:
            return self._generate_unique_name(k)

    def _search_for_experts(self, chance=0.5):
        if random.random() < chance:
            new_name = self._generate_unique_name(k=10)
            new_expert = get_experts(self.dht, [new_name])[0]
            if new_expert is not None and new_expert.uid not in self.local_experts:
                self.experts.append(new_expert)
                self.local_experts.append(new_expert.uid)
                print(
                    f"A new expert joined the swarm! ({new_expert.uid.split('.')[0]})"
                )


# class PraxisDecoder(nn.Module):
#     """
#     A module that wraps the entire decoder block (and all intermediate layers)
#     in a single class.
#     """

#     def __init__(self, config: PraxisConfig):
#         super().__init__()
#         self.shuffle = config.shuffle
#         self.checkpoint_layers = self._checkpoint_strategy(
#             config.memory_profile, config.num_layers
#         )
#         self.dht = False
#         self.experts = nn.ModuleList()
#         if config.hivemind:
#             self.dht = DHT(
#                 # initial_peers=config.initial_peers,
#                 initial_peers=PUBLIC_INITIAL_PEERS,
#                 host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
#                 start=True,
#                 use_auto_relay=True,
#                 use_relay=True,
#                 use_ipfs=False,
#                 ensure_bootstrap_success=True,
#                 daemon=True,
#                 await_ready=True,
#                 # identity_path=os.path.join(os.getcwd(), "id.key"),
#             )
#             schema = BatchTensorDescriptor(
#                 config.num_dims,
#             )
#             attn_schema = BatchTensorDescriptor(
#                 1,
#             )
#             router_weights = BatchTensorDescriptor(
#                 1,
#             )
#             token_indices = BatchTensorDescriptor(
#                 None,
#             )
#             self.backends = {}
#             self.local_experts = []
#             for i in range(config.num_layers):
#                 expert_name = self._generate_unique_name(k=10)
#                 self.local_experts.append(expert_name)
#                 expert = ModuleBackend(
#                     name=expert_name,
#                     module=name_to_block["praxis_block"](config),
#                     args_schema=(
#                         schema,
#                         attn_schema,
#                         router_weights,
#                         token_indices,
#                     ),
#                     outputs_schema=schema,
#                     max_batch_size=64,  # should match the `target_batch_size`
#                     start=True,
#                 )
#                 self.backends[expert_name] = expert
#                 self.experts.append(expert.module)
#             server = Server(
#                 self.dht,
#                 self.backends,
#                 num_connection_handlers=4 * config.num_layers,
#                 device=config.device_map,
#             )
#             server.run_in_background(timeout=5.0)
#         else:
#             [self.experts.append(PraxisBlock(config)) for _ in range(config.num_layers)]

#         self.routers = (
#             nn.ModuleList(
#                 PraxisMixtureOfDepths(config) for _ in range(config.num_layers // 2)
#             )
#             if config.sparse
#             else None
#         )

#     def forward(self, inputs: Tensor, attention_mask: Tensor):
#         if self.shuffle:
#             random.shuffle(self.experts)

#         if self.dht:
#             self._search_for_experts()

#         hidden_states = inputs
#         aux_losses = []

#         for i, expert in enumerate(self.experts):
#             router = (
#                 self.routers[(i - 1) // 2] if self.routers and i % 2 != 0 else None
#             )  # select odd layers
#             gradient_checkpointing = True if i in self.checkpoint_layers else False
#             hidden_states, aux_loss = self._create_forward(
#                 expert,
#                 router,
#                 hidden_states,
#                 attention_mask,
#                 gradient_checkpointing,
#             )
#             aux_losses.append(aux_loss)
#             # try:
#             # hidden_states, aux_loss = self._create_forward(
#             #     expert,
#             #     router,
#             #     hidden_states,
#             #     attention_mask,
#             #     gradient_checkpointing,
#             # )
#             # aux_losses.append(aux_loss)
#             # if isinstance(expert, RemoteExpert):
#             #     hidden_states = expert(
#             #         hidden_states.to("cpu"), attention_mask.to("cpu")
#             #     ).to("cpu")
#             #     hidden_states = hidden_states.to(inputs.device)
#             # else:
#             #     hidden_states = expert(hidden_states, attention_mask)

#             # except P2PDaemonError as e:
#             # except Exception as e:
#             #     print(e)
#             #     print("pruning bad expert")
#             #     self.experts.pop(i)
#             #     if expert.uid in self.local_experts:
#             #         self.local_experts.remove(expert.uid)

#         return hidden_states, sum(aux_losses)

#     def _generate_unique_name(self, k=100):
#         new_name = (
#             random.choice(PREFIXES[:3]) + "~" + random.choice(SUFFIXES[:3]) + ".0"
#         )
#         if new_name not in self.local_experts:
#             return new_name
#         else:
#             return self._generate_unique_name(k)

#     def _search_for_experts(self, chance=0.5):
#         if random.random() < chance:
#             new_name = self._generate_unique_name(k=10)
#             new_expert = get_experts(self.dht, [new_name])[0]
#             if new_expert is not None and new_expert.uid not in self.local_experts:
#                 self.experts.append(new_expert)
#                 self.local_experts.append(new_expert.uid)
#                 print(
#                     f"A new expert joined the swarm! ({new_expert.uid.split('.')[0]})"
#                 )

#     def _checkpoint_strategy(self, strategy="speed", num_layers=0):
#         if strategy == "aggressive":
#             # every layer
#             return [i for i in range(num_layers)]
#         elif strategy == "balanced":
#             # every fourth layer
#             return [i for i in range(num_layers) if i % 4 == 0]
#         else:
#             # no gradient checkpointing
#             return []

#     def _create_forward(
#         self,
#         expert: nn.Module,
#         router: Optional[nn.Module],
#         hidden_states: Tensor,
#         attention_mask: Tensor,
#         gradient_checkpointing=False,
#     ):
#         def custom_forward(*inputs):
#             return router(expert, *inputs) if router else expert(*inputs)

#         if gradient_checkpointing and self.training:
#             return torch.utils.checkpoint.checkpoint(
#                 custom_forward, hidden_states, attention_mask, use_reentrant=False
#             )
#         else:
#             return custom_forward(hidden_states, attention_mask)


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
