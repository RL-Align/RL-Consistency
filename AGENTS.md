# Maintaining RL-Consistency

This repository delivers a portable Markdown skill suite. Keep tracked deliverables as `.md` files. Implement model code, kernels, capture adapters and executable tests in the user's target repositories when applying the skills, not in this skill repository.

- Keep all tracked documentation, headings, task prompts, examples and code comments in English.
- Read `skills/train-inference-consistency/SKILL.md` and the canonical stage map in `references/gates.md` before changing the workflow. Preserve C0–C10 meanings. Default to consistency-only acceptance: C1/C4 performance work is optional and must not block correctness delivery.
- Keep the three-directory suite self-contained under `skills/`. Install the sibling directories together at the same revision; make every required instruction reachable through relative Markdown links inside the suite. `train-inference-consistency/` owns the shared references, models and templates; specialist entrypoints must link to them rather than duplicate acceptance rules. Root README files are user-facing entry points, not hidden dependencies.
- Keep role boundaries explicit: diagnosis/verification does not itself authorize production changes; an existing end-to-end implementation request permits routine role transitions. Preserve bounded task outcomes without claiming complete C0–C10 acceptance. Do not require multi-agent execution.
- Write concrete actions, evidence requirements, failure handling and handoff records. Allow slow common arithmetic implementations as final deliveries when correctness is verified. Do not prescribe a mandatory engine, kernel repository or toolkit CLI, or silently remove explicitly required device/cache/batch/topology behavior.
- Keep architecture guidance separate from measured support. Never infer acceptance from documentation coverage, synthetic fixtures, historical CPU tests or an agent's self-reported success.
- Validate relative links, Markdown-only payload, gate coverage and template completeness. For substantial workflow changes, exercise the skill with fresh agents on bounded task-local fixtures; report the actual scope of those trials.
- Keep runtime history on `archive/runtime-v0.3`; do not reintroduce runtime dependencies into the skill. Do not reference external project PRs as requirements.
