# Qwen3.8-27B Hybrid Dense onboarding

Contents: [scope/sources](#scope-and-source-map) · [prepare](#prepare-and-inspect) ·
[proxy](#inventory-and-proxy) · [numerical tasks](#numerical-integration-tasks) ·
[acceptance](#additional-acceptance).

## Scope and source map

Use this guide for `Qwen/Qwen3.8-27B`, initially the BF16 **text-only** RL path.
The checkpoint is a native vision-language model; this scope does not certify
its visual encoder, images/video, MTP training, speculative decoding, FP8 or MoE.
Do not identify checkpoint weights from architecture dimensions: Qwen3.5 and
Qwen3.8 can share architecture code. Bind the actual model revision and manifests.

Inspected sources (2026-10-18):

- [Official config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json):
  `Qwen3_5ForConditionalGeneration`, nested `text_config`, `qwen3_5_text`.
- [vime model args](https://github.com/vllm-project/vime/blob/d7e40fe8ed9fd9ab61ecb0ec2259b3cc4568af36/scripts/models/qwen3.5-27B.sh)
  and [Qwen3.8 example](https://github.com/vllm-project/vime/blob/d7e40fe8ed9fd9ab61ecb0ec2259b3cc4568af36/examples/coding_agent_rl/run_qwen38_27b_sunabako_2plus2.sh).
- Same vime revision: `vime_plugins/models/qwen3_5.py`, `hf_attention.py`,
  `qwen_gdn_backend.py`; `vime/backends/megatron_utils/{hf_to_megatron,megatron_to_hf}/qwen3_5.py`.
- [Transformers mathematical recurrence](https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/src/transformers/models/qwen3_5/modeling_qwen3_5.py):
  normalization, gate and state semantics; inspect the actual installed revision.

The vime example uses 16 training GPUs and CUDA-specific setup. It is evidence
of an integration route, not a ROCm consistency recipe. Do not run upstream
scripts with broad `pkill`, cluster shutdown or hard-coded remote hosts.
The recorded vime SHA is a reviewed source reference, not a required checkout
change or a substitute for worker provenance.

## Prepare and inspect

Run `prepare-model` as documented in `docs/qwen38.md` using the actual local HF
config. Inspect `inventory.json`, `proxy-plan.json`, `proxy/config.json`,
`vime-model-args.json`, `recipe.json` and `preparation.json`.
These are preparation artifacts, not gate evidence or a loadable checkpoint.
Keep `certified=false` and every GPU gate `not-run` until real execution exists.
The Qwen3-8B `run`/`rocm-check` bridge must not be reused for this model.

## Inventory and proxy

Preserve 64 text layers with repeating L/L/L/F: 48 Gated DeltaNet layers and
16 full-attention layers. Keep hidden=5120, FFN=17408, full Q/KV heads=24/4,
head dim=256, vocabulary=248320. GDN has 16 key heads, 48 value heads,
K/V head dim=128, convolution width=4 and FP32 recurrent state.

Use the first **8 layers** by default, preserving two complete cycles and the
L→L, L→F and F→L transitions. The adapter permits contiguous prefixes of
8,12,...,60 layers. Keep all non-depth config fields, positional semantics,
dtype and dimensions unchanged. Four layers omit F→L and are insufficient.

Materialize selected original language-layer tensors plus embedding, final norm
and head; keep original values/dtypes and audit tensor names against the actual
checkpoint index and converters. Preserve dormant vision/MTP configuration;
explicitly exclude those execution paths. Do not reinitialize proxy weights,
shrink width, or alter the full-model checkpoint. The generated layer map alone
does not create a checkpoint or prove C2. C2 requires a materialized weight
manifest and actual source/proxy configs.

## Numerical integration tasks

| Boundary | Inspect and align |
|---|---|
| Common Dense | RMSNorm `(1 + weight)` semantics, residual rounding, SwiGLU, GEMM and ordered TP/DP reductions |
| Full attention | Q/K normalization, partial RoPE=0.25, theta=1e7, mRoPE metadata, explicit `output_gate_type=swish`, paged KV and training/prefill/decode order |
| GDN projections | Separate QKV, Z, B and A projections; repeat K/Q heads into 48 value heads in the same order |
| Short convolution | Width=4, history order, activation, packed boundaries, state reset and decode updates |
| GDN recurrence | Q/K L2 normalization with eps=1e-6; query scale; sigmoid beta; `g=-exp(A_log)*softplus(a+dt_bias)`; decay then delta update |
| Gated norm/output | Norm-before-SiLU gate, learned scale and cast positions; output projection and residual |
| State ownership | Convolution history, FP32 recurrent matrix and KV cache tracked by request, layer, logical position and weight version |

Inspect actual implementations. The vime GDN currently uses FLA by default;
FlashQLA has NVIDIA/CUDA runtime requirements and must not be selected for ROCm.
FLA availability alone proves no AMD support or bitwise property. The fallback
from Qwen's RMSNorm to `torch.nn.RMSNorm` must be treated as a semantic change
until its weight convention is reconciled and tested.

The inspected vime HF-attention wrapper gathers sequence-parallel activations
with split-only backward for replicated GDN; CP gather has a different gradient
reduction because each rank keeps distinct output chunks. Preserve these
semantics and test gradient scaling when TP/SP/CP changes. Start candidate CP=1;
do not silently adopt Qwen3-8B's CP2 contract. Record every loaded provider;
the pinned RL-Kernel source baseline has no complete Hybrid strict route.

## Additional acceptance

- C0: collect both engines' actual module/kernel routes and `model_config`.
- C2: supply `model_config`, `proxy_config` and real materialized proxy coverage.
  Source descriptors must match the adapter, not invented all-attention layers.
- C3: require common BI/PD cases plus `gdn-chunk-recurrent`, `conv-state-reset`,
  `gdn-state-reset`, `packed-boundaries`, `cache-reorder` and `prefix-cache`.
  Compare logical tokens across chunk/recurrent execution, independent packed
  requests, mixed scheduling and reused cache slots. Capture raw state at the
  first mismatch as well as logprobs; do not compare unrelated state layouts.
- C5: execute independent old-logprob and grad-enabled actor paths. Training's
  chunk kernel and inference's recurrent path must be observed separately.
- C6/C7: cover convolution and GDN backward, A/B/Z gates, norm and collectives.
  Check native mathematical gradients independently of same-order bitwise tests.
- C8/C9: verify theta0/theta1/theta2; invalidate/rebuild recurrence, convolution
  and KV caches after weight sync. Add `gdn-long-context` and
  `state-rebuild-after-sync`. Resume must reconstruct equivalent request state
  or restore it where the serving contract persists it; never reuse stale state.

Use `rl_consistency.models.gdn_reference.gated_delta_reference` on small CPU
fixtures for an independent FP64 recurrence and gradients, including packed
resets and grouped heads. It accepts already projected Q/K/V, log-decay and beta;
it does **not** implement projections, convolution, gate construction, gated
RMSNorm or the full model. It intentionally omits the optimized kernel's rounding
schedule, so use it for accuracy (C7), not as a bitwise provider or baseline.

Publish no verified recipe until the missing strict providers, capture, full
model, update/recovery, accuracy and performance evidence have been completed.
