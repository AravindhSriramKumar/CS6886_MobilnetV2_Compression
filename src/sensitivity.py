"""
HAWQ-V2-style layer sensitivity via Hutchinson Hessian-trace estimation (Q3).

Reference idea (Dong et al., "HAWQ-V2: Hessian Aware trace-Weighted
Quantization"): a layer's sensitivity to quantization is the *average Hessian
trace* of the loss w.r.t. that layer's weights, times the quantization
perturbation it would suffer:

    Omega_l(b) = Tr(H_l)/n_l  *  || W_l - Q_b(W_l) ||_2^2

Tr(H_l) is estimated with Hutchinson's method, which needs no explicit
Hessian: for a Rademacher vector v (entries +-1),

    E[ v^T H v ] = Tr(H)

and Hv is obtained from a second backward pass through the gradient
(double backprop), so the whole thing costs a handful of forward/backward
passes on a small batch subset.

Everything here is plain autograd -- no quantization/compression library.
"""

import torch
import torch.nn as nn

from quantize import quantize_dequantize, weight_scales


def quantizable_layers(model):
    """{name: module} for every Conv2d/Linear, matching `quantize.wrap_model`."""
    return {name: m for name, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))}


def _rademacher_like(t, generator):
    return torch.randint(0, 2, t.shape, device=t.device, dtype=t.dtype,
                         generator=generator).mul_(2).sub_(1)


def hessian_traces(model, loader, device, num_batches=8, num_samples=8,
                   seed=42, verbose=True):
    """Hutchinson estimate of Tr(H_l) for each quantizable layer's weight.

    Returns {layer_name: average Hessian trace per weight element}.
    """
    model.eval()          # BN in inference mode, dropout off -> deterministic H
    layers = quantizable_layers(model)
    names = list(layers)
    params = [layers[n].weight for n in names]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)

    criterion = nn.CrossEntropyLoss()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    acc = {n: 0.0 for n in names}
    n_draws = 0

    for bi, (x, y) in enumerate(loader):
        if bi >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        grads = torch.autograd.grad(loss, params, create_graph=True)

        for _ in range(num_samples):
            vs = [_rademacher_like(p, gen) for p in params]
            # d/dW (g . v) = H v   (v is constant w.r.t. W)
            hvs = torch.autograd.grad(
                grads, params, grad_outputs=vs, retain_graph=True)
            for n, v, hv in zip(names, vs, hvs):
                acc[n] += torch.sum(v * hv).item()
            n_draws += 1
        del grads
        if verbose:
            print(f"  hutchinson batch {bi+1}/{num_batches}", flush=True)

    traces = {n: acc[n] / max(n_draws, 1) for n in names}
    # Per-element (average) trace -- HAWQ-V2 normalizes by layer size so that
    # big layers are not sensitive merely by virtue of having more parameters.
    avg_traces = {n: traces[n] / layers[n].weight.numel() for n in names}

    for p in model.parameters():
        p.requires_grad_(True)
    return traces, avg_traces


def quant_perturbation(model, bits):
    """|| W - Q_b(W) ||_2^2 per layer, using the Q2 per-output-channel scheme."""
    out = {}
    for name, m in quantizable_layers(model).items():
        w = m.weight.detach()
        if bits == 32:
            out[name] = 0.0
            continue
        s = weight_scales(w, bits).view([-1] + [1] * (w.dim() - 1))
        out[name] = float(((w - quantize_dequantize(w, s, bits)) ** 2).sum())
    return out


def sensitivity_scores(model, avg_traces, bits):
    """HAWQ-V2 Omega_l(b) = |avg_trace_l| * ||W - Q_b(W)||^2.

    The absolute value matters in practice: the trained model sits near, but
    not exactly at, a minimum (and the Hutchinson estimate is stochastic), so
    a few layers come back with a small negative trace estimate. A large
    negative curvature is still a direction along which the loss moves fast
    under a weight perturbation, so magnitude is the right sensitivity signal;
    taking the raw signed value would rank those layers as the *safest* to
    crush to 2 bits, which is exactly wrong.
    """
    pert = quant_perturbation(model, bits)
    return {n: abs(avg_traces[n]) * pert[n] for n in avg_traces}


def rank_layers(scores):
    """Layer names ordered most-sensitive first."""
    return [n for n, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
