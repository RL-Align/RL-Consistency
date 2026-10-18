# Implementation roadmap

| Milestone | Deliverable | Acceptance |
|---|---|---|
| Foundation (this change) | Skill, recipe binding, evidence checks, CLI, HTML, Qwen3 launch bridge | CPU failure-path tests and skill validation |
| Capture primitives (implemented) | Explicit worker trace writer, loaded-file fingerprints, existing sidecar diagnostic import | CPU PyTorch serialization, dtype, mask and distributed-layout failure-path tests |
| ROCm diagnostic runner (implemented, GPU run pending) | Profile resolution, pinned launch, raw comparison collection, JSON/HTML verdict | CPU orchestration/failure tests; real MI300X results still required |
| Workflow v2 (implemented validators) | PR #61551 gate roles, dual-path receipts, master updates and theta0/1/2 | New schemas reject old-numbered evidence; raw GPU producers remain necessary |
| Qwen3 producers | Actual vLLM/Megatron worker capture and source/binary/weight manifests | C0/C5 evidence from both engines and all relevant ranks |
| Reduced-model path | Model-derived layer selector and real inference harness | C2–C4 on hardware; no dimensional shrink |
| Training and recovery | Independent forward/gradient references, update and restart fixtures | C6–C9 with retained raw evidence |
| First verified recipes | Separate CUDA and ROCm measured runs with frozen native baselines | C0–C10 plus human baseline/precision/release review |
| Second model (CPU preparation implemented) | Qwen3.8-27B Hybrid Dense config/proxy adapter, candidates, state cases and FP64 GDN reference | Preparation and CPU gradient tests implemented; strict providers, proxy weights and GPU acceptance pending |
| Registry | Authenticated reviews, signed artifact identities, compatible recipe lookup | Untrusted `verified` labels rejected; revocation tested |
| Broader scope | MoE/PD, new frameworks/hardware, cross-topology/hardware contracts | Separate comparison matrices and evidence for each advertised relation |

No time or hardware-performance estimates are asserted by this scaffold. Model-specific GPU acceptance remains pending.
