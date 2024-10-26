import logging
import random
from typing import Optional

import hivemind
import hivemind.moe
import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind import DHT
from hivemind.moe import RemoteExpert
from hivemind.moe.client.expert import _RemoteModuleCall
from hivemind.moe.server.layers import name_to_block, name_to_input
from hivemind.p2p import P2PDaemonError, P2PHandlerError
from hivemind.utils import BatchTensorDescriptor
from hivemind.utils.nested import nested_compare, nested_flatten, nested_pack
from torch import Tensor
from torch.autograd.function import once_differentiable

from praxis import PraxisConfig
from praxis.modules.experts import PraxisBlock
from praxis.modules.router import PraxisMixtureOfDepths


class FaultTolerantRemoteModuleCall(_RemoteModuleCall):
    """A monkey-patch of _RemoteModuleCall that handles failures gracefully"""

    @staticmethod
    def forward(ctx, dummy, uid, stub, info, *inputs):
        try:
            return _RemoteModuleCall.forward(ctx, dummy, uid, stub, info, *inputs)
        except Exception as e:
            # Return zeros matching input shape
            return tuple(
                torch.zeros_like(inputs[0])
                for _ in nested_flatten(info["outputs_schema"])
            )

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        try:
            return _RemoteModuleCall.backward(ctx, *grad_outputs)
        except Exception as e:
            # Return zero grads matching input shapes
            return (
                torch.empty(0, requires_grad=True),
                None,
                None,
                None,
                *(
                    torch.zeros_like(x) if x.requires_grad else None
                    for x in ctx.saved_tensors
                ),
            )


# Apply the patch
hivemind.moe.client.expert._RemoteModuleCall = FaultTolerantRemoteModuleCall
from hivemind.moe import ModuleBackend, RemoteExpert, Server, get_experts


class PraxisSwarm:
    """
    A helper class, with convenience methods for Hivemind swarm management.
    """

    def __init__(self, config: PraxisConfig):
        super().__init__()

        self.expert_uids = []
        self.active_remote_experts = []
        self.dht = DHT(
            initial_peers=PUBLIC_INITIAL_PEERS,
            host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
            start=True,
        )
        # TODO: the mere act of using this method prevents bootstrap freezing
        visible_maddrs_str = [str(a) for a in self.dht.get_visible_maddrs()]
        self.backends = {}
        self.active_local_experts = []
        # self.server, self.dht, self.active_local_experts = PraxisServer.create(
        #     expert_uids=self.expert_uids,
        #     expert_cls="praxis_expert",
        #     start=True,
        #     daemon=True,
        #     use_relay=True,
        #     use_auto_relay=True,
        #     use_ipfs=False,
        #     ensure_bootstrap_success=True,
        #     initial_peers=PUBLIC_INITIAL_PEERS,
        #     # initial_peers=IPFS_INITIAL_PEERS,
        #     host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
        #     config=config,
        # )

    @property
    def local_experts(self):
        return self.active_local_experts

    @property
    def remote_experts(self):
        return self.active_remote_experts

    def get_visible_maddrs(self):
        return self.dht.get_visible_maddrs()

    def is_remote(self, expert: Optional[RemoteExpert]):
        return isinstance(expert, RemoteExpert)

    def register_expert(
        self, config: PraxisConfig, expert_cls: str = "hivemind_expert"
    ):
        assert expert_cls in name_to_block
        expert = name_to_block[expert_cls](config)

        # hidden_schema = BatchTensorDescriptor(
        #     config.num_dims,
        # )
        # attention_schema = BatchTensorDescriptor(
        #     1,
        # )
        dummy_inputs = BatchTensorDescriptor.from_tensor(
            torch.zeros([4, 1, config.num_dims])
        )
        dummy_attention = BatchTensorDescriptor.from_tensor(torch.zeros([4, 1]))

        expert_uid = self._generate_unique_name()
        self.expert_uids.append(expert_uid)
        self.backends[expert_uid] = ModuleBackend(
            name=expert_uid,
            module=expert,
            # args_schema=(hidden_schema, attention_schema),
            # outputs_schema=(hidden_schema),
            args_schema=(dummy_inputs, dummy_attention),
            outputs_schema=(dummy_inputs),
            optimizer=None,
            scheduler=None,
            min_batch_size=1,
            max_batch_size=8192,
        )
        module = self.backends[expert_uid].module
        self.active_local_experts.append(module)
        return module

    def serve_experts(self, config: PraxisConfig):
        device = config.device_map or ("cuda" if torch.cuda.is_available() else "cpu")
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
        if expert.uid in self.expert_uids:
            print("removing:", expert.uid)
            self.expert_uids.remove(expert.uid)

    def _generate_unique_name(self, k=3):
        new_name = (
            random.choice(PREFIXES[:k]) + "~" + random.choice(SUFFIXES[:k]) + ".0"
        )
        if new_name not in self.expert_uids:
            return new_name
        else:
            return self._generate_unique_name(k)

    def _search_for_experts(self, chance=0.5):
        if random.random() < chance:
            new_name = self._generate_unique_name()
            new_expert = get_experts(self.dht, [new_name])[0]
            if new_expert is not None and new_expert.uid not in self.expert_uids:
                self.active_remote_experts.append(new_expert)
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

IPFS_INITIAL_PEERS = [
    # From running `ipfs bootstrap list` with Kubo
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt",
    "/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ",
    "/ip4/104.131.131.82/udp/4001/quic-v1/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ",
    # "/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN",
    # "/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa",
    # "/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb",
    # "/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt",
    # "/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ",
    # "/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ",
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
