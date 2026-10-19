# Implementation roadmap

The default product is now the independent shared runtime; existing engine bridges remain optional compatibility tools.

| Milestone | Deliverable | Acceptance |
|---|---|---|
| Independent reference runtime (implemented) | Shared-prefix generation, independent old-logprob/actor, real backward, local safetensors and optional SGD updates for Dense/Hybrid/MoE | Real tiny FP32/BF16 architecture tests and failure tests; full-model GPU acceptance pending |
| Independent GPU release (pending) | Actual full-checkpoint fixtures on pinned ROCm/CUDA environments, accuracy and memory/performance measurements | Reviewed retained artifacts; no claim from tiny CPU tests |
| Faster independent providers (pending) | Cache execution, invariant batching and distributed operators | Separate explicit contracts and model/environment matrices |
| Foundation (implemented) | Skill, recipe binding, evidence checks, CLI, HTML, Qwen3 launch bridge | CPU failure-path tests and skill validation |
| Capture primitives (implemented) | Explicit worker trace writer, loaded-file fingerprints, existing sidecar diagnostic import | CPU PyTorch serialization, dtype, mask and distributed-layout failure-path tests |
| Native divergence locator (implemented CPU tooling) | Explicit module hooks and semantic DAG, binary snapshots, repeat checks, local replay and HTML matrix for unregistered models | Reconvergence, parallel frontiers, missing coverage, unstable repeats, autograd and replay failure tests; actual engine/GPU capture still required |
| ROCm diagnostic runner (implemented, GPU run pending) | Profile resolution, pinned launch, raw comparison collection, JSON/HTML verdict | CPU orchestration/failure tests; real MI300X results still required |
| Workflow v2 (implemented validators) | C0–C10 gate roles, dual-path receipts, master updates and theta0/1/2 | New schemas reject old-numbered evidence; raw GPU producers remain necessary |
| Qwen3 producers | Actual vLLM/Megatron worker capture and source/binary/weight manifests | C0/C5 evidence from both engines and all relevant ranks |
| Reduced-model path | Model-derived layer selector and real inference harness | C2–C4 on hardware; no dimensional shrink |
| Training and recovery | Independent forward/gradient references, update and restart fixtures | C6–C9 with retained raw evidence |
| First verified recipes | Separate CUDA and ROCm measured runs with frozen native baselines | C0–C10 plus human baseline/precision/release review |
| Second model (CPU preparation implemented) | Qwen3.8-27B Hybrid Dense config/proxy adapter, candidates, state cases and FP64 GDN reference | Preparation and CPU gradient tests implemented; strict providers, proxy weights and GPU acceptance pending |
| Registry | Authenticated reviews, signed artifact identities, compatible recipe lookup | Untrusted `verified` labels rejected; revocation tested |
| Broader scope | MoE/PD, new frameworks/hardware, cross-topology/hardware contracts | Separate comparison matrices and evidence for each advertised relation |

No time or hardware-performance estimates are asserted by this scaffold. Model-specific GPU acceptance remains pending.
