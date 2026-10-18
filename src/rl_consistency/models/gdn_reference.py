"""Independent FP64 gated-delta-rule accuracy reference, CPU and small fixtures only.

This serial mathematical recurrence is not a production kernel or a bitwise
reference for FLA, vLLM or RL-Kernel. It deliberately keeps all arithmetic in
FP64 and preserves autograd, without copying their chunk/rounding schedules.
Contract: packed [tokens, heads, channels], decay in log space, beta in [0,1],
Q/K normalization by sqrt(sum(x*x) + eps), query scale 1/sqrt(key dimension).
See the Qwen3.8 skill reference for sources and excluded projection/conv stages.
"""
from __future__ import annotations

import math
from ..common import require


def gated_delta_reference(query, key, value, log_decay, beta, *, cu_seqlens,
                          initial_state=None, normalize_qk=True, eps=1e-6):
    import torch
    tensors = (query, key, value, log_decay, beta)
    require(all(isinstance(t, torch.Tensor) and t.device.type == "cpu" and t.is_floating_point()
                and bool(torch.isfinite(t).all()) for t in tensors), "finite CPU floating tensors required")
    require(query.ndim == key.ndim == value.ndim == 3 and query.shape == key.shape,
            "expected packed [tokens, heads, channels] Q/K with matching shapes")
    tokens, key_heads, key_dim = key.shape
    require(tokens > 0 and key_heads > 0 and key_dim > 0 and value.shape[0] == tokens
            and value.shape[1] > 0 and value.shape[2] > 0 and value.shape[1] % key_heads == 0,
            "invalid packed dimensions or GQA head grouping")
    heads, value_dim = value.shape[1:]
    require(log_decay.shape == beta.shape == (tokens, heads), "decay/beta shape mismatch")
    require(bool((log_decay <= 0).all()) and bool(((beta >= 0) & (beta <= 1)).all()),
            "log decay must be <=0 and beta in [0,1]")
    require(type(normalize_qk) is bool and type(eps) in (int,float) and math.isfinite(eps) and eps > 0,
            "invalid normalization contract")
    require(isinstance(cu_seqlens, (tuple,list)) and len(cu_seqlens) >= 2
            and all(type(n) is int for n in cu_seqlens) and cu_seqlens[0] == 0 and cu_seqlens[-1] == tokens
            and all(a < b for a,b in zip(cu_seqlens,cu_seqlens[1:])), "invalid packed sequence boundaries")
    q, k, v, decay, rate = [t.to(torch.float64) for t in tensors]
    if normalize_qk:
        q = q / (q.square().sum(-1, keepdim=True) + eps).sqrt()
        k = k / (k.square().sum(-1, keepdim=True) + eps).sqrt()
    q = q.repeat_interleave(heads // key_heads, dim=1) / math.sqrt(key_dim)
    k = k.repeat_interleave(heads // key_heads, dim=1)
    state_shape = (len(cu_seqlens)-1, heads, key_dim, value_dim)
    if initial_state is None:
        initial = v.new_zeros(state_shape)
    else:
        require(isinstance(initial_state, torch.Tensor) and initial_state.device.type == "cpu"
                and initial_state.is_floating_point() and initial_state.shape == state_shape
                and bool(torch.isfinite(initial_state).all()), "invalid initial recurrent state")
        initial = initial_state.to(torch.float64)
    outputs, final = [], []
    for sequence, (start, end) in enumerate(zip(cu_seqlens,cu_seqlens[1:])):
        state = initial[sequence]
        for token in range(start, end):
            decayed = state * decay[token].exp()[:, None, None]
            prediction = torch.einsum("hk,hkv->hv", k[token], decayed)
            correction = rate[token, :, None] * (v[token] - prediction)
            state = decayed + torch.einsum("hk,hv->hkv", k[token], correction)
            outputs.append(torch.einsum("hk,hkv->hv", q[token], state))
        final.append(state)
    return torch.stack(outputs), torch.stack(final)
