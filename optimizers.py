import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_optimizer import create_optimizer
from pytorch_optimizer.optimizer import TRAC, Lookahead, OrthoGrad
from torch.optim import Optimizer


def get_optimizer_profile(name="AdamW", shuffle=False, no_schedule=False):
    profiles = {k.lower(): v for k, v in OPTIMIZER_PROFILES.items()}
    profile = {**profiles.get(name.lower()), "wd_ban_list": WD_BAN_LIST}
    profile["weight_decay"] = 0 if shuffle else profile.get("weight_decay", None)
    no_schedule = profile.get("no_schedule", no_schedule)
    if "no_schedule" in profile:
        del profile["no_schedule"]
    return profile, no_schedule


def get_optimizer(model, trac=False, ortho=False, lookahead=False, *args, **kwargs):
    if kwargs.get("optimizer_name") == "DoWG":
        del kwargs["optimizer_name"]
        del kwargs["wd_ban_list"]
        del kwargs["weight_decay"]
        optimizer = DoWG(model.parameters(), **kwargs)
    else:
        optimizer = create_optimizer(model, *args, **kwargs)
    if trac:
        optimizer = TRAC(optimizer, num_coefs=128)
    if ortho:
        optimizer = OrthoGrad(optimizer)
    if lookahead:
        optimizer = Lookahead(optimizer, k=5, alpha=0.5, pullback_momentum="none")
    return optimizer


# Most optimizer settings can be found here:
# https://pytorch-optimizers.readthedocs.io/en/latest/optimizer
OPTIMIZER_PROFILES = {
    "AdamG": dict(
        optimizer_name="AdamG",
        lr=1.0,
        weight_decay=0.1,
        weight_decouple=True,
        p=0.2,
        q=0.24,
        betas=(0.95, 0.999, 0.95),
        no_schedule=True,
    ),
    "AdamW": dict(
        optimizer_name="AdamW",
        lr=1e-3,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    ),
    "AdEMAMix": dict(
        optimizer_name="AdEMAMix",
        lr=0.001,
        weight_decay=0.1,
        weight_decouple=True,
        betas=(0.9, 0.95, 0.9999),
        alpha=5.0,
        cautious=True,
    ),
    "Grams": dict(
        optimizer_name="Grams",
        lr=0.001,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    ),
    "Lion": dict(
        optimizer_name="Lion",
        lr=0.000333,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        r=0.98,
        use_gc=True,
        adanorm=True,
        cautious=True,
    ),
    "Prodigy": dict(
        optimizer_name="Prodigy",
        lr=1.0,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        beta3=0.98,
        growth_rate=float("inf"),
        d_coef=0.1,
        bias_correction=True,
        safeguard_warmup=False,
        no_schedule=True,
    ),
    "DoWG": dict(
        optimizer_name="DoWG",
        eps=1e-6,
        no_schedule=True,
    ),
    "SOAP": dict(
        optimizer_name="SOAP",
        lr=0.00333,
        weight_decay=0.1,
        betas=(0.95, 0.99),
        shampoo_beta=0.98,
        precondition_frequency=10,
        max_precondition_dim=10000,
        normalize_gradient=False,
        correct_bias=True,
        precondition_1d=False,
        merge_dims=False,
    ),
}

WD_BAN_LIST = [
    "bias",
    "edge_embeddings",
    "spatial_embeddings",
    "Embedding",
    "BatchNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "GroupNorm",
    "LayerNorm",
    "RMSNorm",
    "InstanceNorm",
    "InstanceNorm1d",
    "InstanceNorm3d",
    "InstanceNorm2d",
    "PReLU",
    "SinLU",
    "NMDA",
]


class DoWG(Optimizer):
    """Implements DoWG optimization algorithm.

    Args:
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4). Also used as the default squared distance estimate.
    """

    def __init__(self, params, eps=1e-4):
        defaults = dict(eps=eps)
        self.eps = eps
        super(DoWG, self).__init__(params, defaults)
        # Ensure each parameter group has an lr
        for param_group in self.param_groups:
            if "lr" not in param_group:
                param_group["lr"] = 1.0  # Set default learning rate

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        state = self.state
        device = self.param_groups[0]["params"][0].device

        # Initialize state variables if needed
        if "rt2" not in state:
            state["rt2"] = torch.Tensor([self.eps]).to(device)
        if "vt" not in state:
            state["vt"] = torch.Tensor([0]).to(device)

        grad_sq_norm = torch.Tensor([0]).to(device)
        curr_d2 = torch.Tensor([0]).to(device)

        for idx, group in enumerate(self.param_groups):
            group_state = state[idx]

            # Check if x0 needs to be initialized or updated
            needs_reset = False
            if "x0" not in group_state:
                needs_reset = True
            else:
                # Check if sizes match
                for p, p0 in zip(group["params"], group_state["x0"]):
                    if p.size() != p0.size():
                        needs_reset = True
                        break

            if needs_reset:
                group_state["x0"] = [p.data.clone().detach() for p in group["params"]]

            # Only include parameters that have gradients
            grad_sq_norm += (
                torch.stack(
                    [
                        (p.grad.data**2).sum()
                        for p in group["params"]
                        if p.grad is not None
                    ]
                ).sum()
                if any(p.grad is not None for p in group["params"])
                else torch.Tensor([0]).to(device)
            )

            curr_d2 += torch.stack(
                [
                    ((p.data - p0) ** 2).sum()
                    for p, p0 in zip(group["params"], group_state["x0"])
                ]
            ).sum()

        state["rt2"] = torch.max(state["rt2"], curr_d2)
        state["vt"] += state["rt2"] * grad_sq_norm
        rt2, vt = state["rt2"], state["vt"]

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    gt_hat = rt2 * p.grad.data
                    denom = torch.sqrt(vt).add_(group["eps"])
                    p.data.addcdiv_(gt_hat, denom, value=-1.0)

        return loss
