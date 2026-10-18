# Qwen3-30B-A3B MoE integration

## Inspect and prepare

Use model ID `Qwen/Qwen3-30B-A3B`, family `qwen3_moe`, and the actual flat
`Qwen3MoeForCausalLM` HF config. Do not substitute Qwen3 Dense, Qwen3.8 Hybrid,
an Instruct-2507 variant or a quantized checkpoint. Run `prepare-model` as shown
in `docs/qwen3-moe.md`; the default four-layer prefix preserves all widths,
128 experts, top-8 and each router. Materialize source weights separately.
Prepared configs and CPU tests never pass GPU gates.

Read `docs/evidence-format.md` and `docs/moe-routing.md` before implementing a
producer. Use the candidate `qwen3-30b-a3b-vime-{cuda,rocm}` matching the backend.
Keep BF16, no token dropping, independent routing and no speculative decoding.
Pin the actual config/revision, worker sources, loaded binaries and process
groups. Preparation binds config hashes only, not source or weight identities.

## Reuse boundaries and numerical work

Inspect Qwen-owned `config.json` and vime commit
`d7e40fe8ed9fd9ab61ecb0ec2259b3cc4568af36`:

| Source | What to inspect |
|---|---|
| `scripts/models/qwen3-30B-A3B.sh` | 48 layers, H2048, 32 Q / 4 KV heads, D128, expert width 768, softmax top-8, FP32 router, all-to-all, grouped GEMM |
| `scripts/run-qwen3-30B-A3B.sh` | Native eight-GPU TP4/EP8/ETP1 training, TP8 rollout; CUDA-specific setup and destructive cleanup commands must not be run wholesale |
| `tests/test_qwen3_30B_A3B.py` | Native integration uses routing replay; FP8 rollout and DeepEP are enabled by default in this test |
| `tests/test_qwen3_30B_A3B_r3.py` | Rollout routing replay diagnostic, not independent-router evidence |
| `tests/test_qwen3_30B_A3B_training_recovery.py` | Native recovery reference; it does not establish our full optimizer/RNG/data-cursor resume contract |

Resolve these paths under `https://github.com/vllm-project/vime/blob/<commit>/`.
Inspect the pinned Transformers/vLLM/Megatron code actually loaded on the host,
including router casting and expert checkpoint mappings; model config alone
does not define every arithmetic detail.

Reuse RL-Kernel's RMSNorm, QK norm, RoPE, attention, GEMM, SwiGLU and logprob
providers only after checking these shapes and numerical contracts. This MoE
uses ordinary RMSNorm, not the Hybrid model's 1+weight norm. The inspected
RL-Kernel baseline is not a complete MoE runtime. Implement and test:

1. Router GEMM → FP32 softmax → top-8 → selected-weight renormalization. The
   strict candidate chooses lower expert ID for exact score ties. Do not assume
   native `topk` supplies that tie rule. Capture normalized FP32 bytes before
   cast and audit the actual applied-weight cast, multiplication and accumulation.
2. Stable logical `(sample, position, layer, slot)` mappings through dispatch,
   all-to-all and unpermute. Cover empty experts, skew, packing, permutations
   and capacity stress; verify actual token/slot counts and reject dropping.
   Native `--moe-token-drop-policy probs` is not evidence that no tokens dropped.
3. Expert gate/up/down GEMMs and SwiGLU with batch-invariant reductions. Changing
   expert batch size must not change the arithmetic contract. Combine in fixed
   ascending slot order; audit kernel reductions rather than claiming that
   native `index_add_`, grouped GEMM or a recorded slot list is sufficient.
4. Record actual dense/expert TP, EP, DP groups and rank-to-expert ownership.
   Megatron's dense TP4/DP2 and expert TP1/EP8/DP1 overlap on eight ranks;
   TP×EP is not an independent-axis GPU count. Rollout TP8/EP1 shards each
   expert differently. Fix both sides' reductions and weight redistribution.
   Establish ROCm support per selected all-to-all/fused provider; do not assume
   CUDA NCCL/DeepEP launch settings apply unchanged.

## Gates and failure isolation

At C0/C2/C9 supply the actual `model_config`; at C2 also supply `proxy_config`
and a materialized weight manifest. Keep all mandatory MoE cases in the recipe.
At C3/C5/C8/C9 submit `routing_pairs` for every output case. Bind each ledger
to the corresponding output's weights, version, inputs, producer and executed
callsite. Include all proxy layers, or all 48 layers at C9, for every captured
token and all scored tokens. Declare complete prompt/decode and rank coverage
in the fixture; the validator cannot infer omitted unscored prompt tokens.

Run both native old-logprob recomputation and grad-enabled actor forward.
Do not copy rollout expert IDs/weights into training to obtain a passing
comparison. Save R3 experiments separately; both training routing replay and
rollout routing replay fail this independent-routing contract.

For a mismatch, locate the earliest divergence among router inputs/logits,
softmax, selected IDs/weights, dispatch mapping, expert outputs and combine.
Equal final logprobs do not excuse different routes. At C7 use independent
mathematical router/expert/combine references and gradient checks away from
top-k discontinuities; test exact-tie selection separately. Account for expert
and router parameters in gradient coverage. The first training objective uses
aux coefficient zero; retain the HF config value without silently using it as
the training setting. A different objective needs a new contract and run.

After each real update, compare independently computed routes and outputs at
theta0/theta1/theta2. Validate full-model context, EP behavior, synchronized
router/expert weights and first restored forward/update. Include separate
uninterrupted/resumed routing artifacts. Profile all-to-all, permute, grouped
GEMM and combine against the frozen BF16 native baseline at equal work.

Report strict provider implementation, capture integration, proxy weights and
real GPU gates as pending until completed. Do not use the Qwen3-8B launch bridge
for this model or promote an onboarding bundle to a runnable verified recipe.
