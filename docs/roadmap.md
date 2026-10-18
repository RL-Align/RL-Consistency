# Implementation roadmap

| Milestone | Deliverable | Acceptance |
|---|---|---|
| Foundation (this change) | Skill, recipe binding, evidence checks, CLI, HTML, Qwen3 launch bridge | CPU failure-path tests and skill validation |
| Capture primitives (implemented) | Explicit worker trace writer, loaded-file fingerprints, existing sidecar diagnostic import | CPU PyTorch serialization, dtype, mask and distributed-layout failure-path tests |
| Qwen3 producers | Actual vLLM/Megatron worker capture and source/binary/weight manifests | C0/C5 evidence from both engines and all relevant ranks |
| Reduced-model path | Model-derived layer selector and real inference harness | C2–C4 on hardware; no dimensional shrink |
| Training and recovery | Independent forward/gradient references, update and restart fixtures | C6–C9 with retained raw evidence |
| First verified recipes | Separate CUDA and ROCm measured runs with frozen native baselines | C0–C10 plus human baseline/precision/release review |
| Second model | A second real model adapter without copying the workflow | Full recipe acceptance and uncovered semantic cases documented |
| Registry | Authenticated reviews, signed artifact identities, compatible recipe lookup | Untrusted `verified` labels rejected; revocation tested |
| Broader scope | MoE/PD, new frameworks/hardware, cross-topology/hardware contracts | Separate comparison matrices and evidence for each advertised relation |

No time or hardware-performance estimates are asserted by this scaffold. Model-specific GPU acceptance remains pending.
