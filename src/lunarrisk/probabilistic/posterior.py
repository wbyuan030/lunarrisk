"""Conditional posteriors over the four attributes (P2.1 A1): P1 point, P2 diagonal, P3 joint.

All three variants share one front end and differ only in the head, so the comparison isolates the
*distributional* assumption rather than capacity:

* ``point``    - 4 outputs, the attribute means (rules consume them as a delta posterior).
* ``diagonal`` - 8 outputs, ``mu`` plus ``log sigma`` per attribute.
* ``joint``    - 8 outputs plus a rank-``k`` factor ``U``: ``Sigma = diag(sigma^2) + U U^T``, which is
  symmetric positive definite by construction and equals the diagonal model exactly when ``U = 0``.

The posterior lives in the unconstrained chain from ``transforms``; ``nll_internal`` returns the
Gaussian NLL in that space together with the Jacobian term, and both numbers are reported so nobody
can quote internal-space NLL as a physical density.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

N_ATTRIBUTES = 4
PARAM_COUNT_FIXED_MASK = 98_464       # P1.8 fusion front end: encoder(3ch, 7 scalars) + 4-output head
PARAM_COUNT_ENCODER = 97_920          # same encoder without the head
VARIANTS = ("point", "diagonal", "joint")


class SmallEncoder(nn.Module):
    """Same architecture as the P1.8 fixed-mask front end (4 conv blocks, 128 pooled features)."""

    def __init__(self, in_channels: int = 1, n_scalars: int = 0, width: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.width = width + n_scalars
        self.n_scalars = n_scalars

    def forward(self, x: torch.Tensor, scalars: torch.Tensor | None = None) -> torch.Tensor:
        features = self.pool(self.features(x)).flatten(1)
        if scalars is not None and self.n_scalars:
            features = torch.cat([features, scalars], dim=1)
        return features


class ConditionalPosterior(nn.Module):
    """Shared encoder plus a head whose mean output count depends on the variant."""

    def __init__(self, variant: str = "diagonal", in_channels: int = 3, n_scalars: int = 7,
                 rank: int = 2, width: int = 128) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.variant = variant
        self.rank = rank if variant == "joint" else 0
        self.in_channels = in_channels
        self.n_scalars = n_scalars
        self.base_width = width
        self.encoder = SmallEncoder(in_channels, n_scalars, width)
        n_out = N_ATTRIBUTES
        if variant in ("diagonal", "joint"):
            n_out += N_ATTRIBUTES
        if self.rank:
            n_out += N_ATTRIBUTES * self.rank
        self.head = nn.Linear(self.encoder.width, n_out)

    def forward(self, x: torch.Tensor, scalars: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.encoder(x, scalars))

    def parameters_dict(self, out: torch.Tensor) -> dict:
        """Split the raw head output into named tensors."""
        mu = out[..., :N_ATTRIBUTES]
        result = {"mu": mu}
        if self.variant in ("diagonal", "joint"):
            log_sigma = out[..., N_ATTRIBUTES:2 * N_ATTRIBUTES]
            # keep sigma in a numerically safe band; exp of -6 is 2.5e-3, exp of 2 is 7.4
            result["log_sigma"] = log_sigma
            result["sigma"] = torch.exp(torch.clamp(log_sigma, -6.0, 2.0))
        if self.rank:
            raw = out[..., 2 * N_ATTRIBUTES:]
            result["U"] = raw.view(*raw.shape[:-1], N_ATTRIBUTES, self.rank)
        return result

    def covariance(self, params: dict) -> torch.Tensor:
        """Full covariance per sample: diag(sigma^2) (+ U U^T for the joint variant)."""
        if self.variant == "point":
            raise ValueError("the point variant has no covariance")
        sigma = params["sigma"]
        cov = torch.diag_embed(sigma ** 2)
        if self.rank:
            u = params["U"]
            cov = cov + u @ u.transpose(-1, -2)
        return cov

    def nll(self, out: torch.Tensor, z: torch.Tensor):
        """Gaussian NLL in the internal chain, plus the Jacobian-corrected physical value."""
        params = self.parameters_dict(out)
        mu = params["mu"]
        if self.variant == "point":
            residual = (z - mu) ** 2
            return residual.sum(dim=-1), residual.sum(dim=-1)
        cov = self.covariance(params)
        # Cholesky keeps the density well defined and surfaces non-PSD errors early
        chol = torch.linalg.cholesky(cov)
        diff = (z - mu).unsqueeze(-1)
        solved = torch.linalg.solve_triangular(chol, diff, upper=False)
        quad = (solved ** 2).sum(dim=(-1, -2))
        log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        internal = 0.5 * (quad + log_det + N_ATTRIBUTES * math.log(2 * math.pi))
        jacobian = jacobian_term(mu)
        return internal, internal - jacobian


def jacobian_term(z: torch.Tensor) -> torch.Tensor:
    """log|det J| of the transform, evaluated at ``z`` (see transforms.log_det_jacobian)."""
    eps = 1e-6
    z1, z2, z3, z4 = z[..., 0], z[..., 1], z[..., 2], z[..., 3]
    p10 = torch.clamp(torch.sigmoid(z1), eps, 1 - eps)
    ratio = torch.clamp(torch.sigmoid(z2), eps, 1 - eps)
    return (torch.log(p10) + torch.log(1 - p10) + torch.log(p10)
            + torch.log(ratio) + torch.log(1 - ratio) + z3 + z4)


def condition_posterior(variant: str = "diagonal", *, rank: int = 2, in_channels: int = 3,
                        n_scalars: int = 7, width: int = 128) -> ConditionalPosterior:
    """Defaults reproduce the P1.8 fusion front end (3 channels, 7 scalars, 98,464 parameters in
    total for the point head). The diagonal and joint variants add outputs only in the head, so the
    encoder capacity is identical across variants and the comparison isolates the distribution."""
    return ConditionalPosterior(variant, in_channels=in_channels, n_scalars=n_scalars, rank=rank,
                                width=width)


def export_state(model: ConditionalPosterior, *, target_eps: float = 1e-6, seeds=None,
                 transforms=None) -> dict:
    """Everything needed to re-run inference: weights, variant, transforms, eps, seeds."""
    import hashlib
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    buffer = b"".join(v.numpy().tobytes() for _, v in sorted(state.items()))
    return {"variant": model.variant, "rank": model.rank, "width": model.base_width,
            "in_channels": model.in_channels, "n_scalars": model.n_scalars,
            "state_dict": state, "target_eps": target_eps, "seeds": seeds or [],
            "transforms": transforms or {"type": "logit_chain", "eps": target_eps},
            "weights_sha256": hashlib.sha256(buffer).hexdigest()}


def posterior_from_state(payload: dict) -> ConditionalPosterior:
    model = condition_posterior(payload["variant"], rank=payload["rank"] and payload["rank"],
                                width=payload.get("width", 128),
                                in_channels=payload.get("in_channels", 3),
                                n_scalars=payload.get("n_scalars", 7))
    model.load_state_dict(payload["state_dict"])
    return model


def sample_internal(model: ConditionalPosterior, x, scalars=None, *, draws: int = 256,
                    seed: int = 0) -> torch.Tensor:
    """Seeded draws of the internal chain, shape (draws, batch, 4).

    Uses an explicit ``torch.Generator`` rather than ``MultivariateNormal.sample``, because that method
    draws from the *global* RNG and would make "same seed, same posterior" silently false.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        out = model(x, scalars)
        params = model.parameters_dict(out)
        if model.variant == "point":
            return params["mu"].unsqueeze(0).repeat(draws, 1, 1)
        chol = torch.linalg.cholesky(model.covariance(params))
        # noise is drawn on CPU and moved afterwards: a CPU generator cannot fill a CUDA tensor, and
        # this way a fixed seed yields identical draws on CPU and GPU
        noise = torch.randn((draws,) + params["mu"].shape, generator=generator).to(chol.device)
        # (draws, batch, 4) x (batch, 4, 4): the batch is a shared leading dim, not a draw index
        correlated = torch.einsum("sni,nij->snj", noise, chol.transpose(-1, -2))
        return params["mu"].unsqueeze(0) + correlated


def nll_internal(model: ConditionalPosterior, x, z, scalars=None):
    """Convenience wrapper used by tests and the training loop."""
    return model.nll(model(x, scalars), z)
