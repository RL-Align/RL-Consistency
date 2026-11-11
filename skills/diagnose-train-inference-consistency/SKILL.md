---
name: diagnose-train-inference-consistency
description: Diagnose training–inference numerical drift or verify a consistency change in an existing framework. Use for independent captures, first-divergence investigation, comparator checks and evidence-based acceptance. A diagnosis or verification request does not by itself authorize production arithmetic changes.
---

# Diagnose and Verify Train–Inference Consistency

Measure the actual paths, localize differences and report what the evidence establishes. This skill supports both pre-change diagnosis and post-change verification. It may implement task-local capture, comparators, reference checks and tests; keep product arithmetic, objective, routing and configured execution behavior unchanged unless the user also authorized implementation.

## Bind the task

This skill requires the sibling `train-inference-consistency/` directory from the same suite revision. Read [suite roles and handoffs](../train-inference-consistency/references/skill-suite.md), [numerical contract](../train-inference-consistency/references/numerical-contract.md) and [evidence rules](../train-inference-consistency/references/evidence.md). The references, templates and model guides there are the single authority; do not copy or invent alternative acceptance rules. A missing sibling is a packaging blocker, not a reason to improvise a contract.

Inspect the user's requested scope and existing [task brief](../train-inference-consistency/templates/task-brief.md), [ledger](../train-inference-consistency/templates/run-ledger.md) and [handoff](../train-inference-consistency/templates/handoff.md). If absent, create task-local records from inspected facts. Identify actual callsites and loaded implementations using [C0](../train-inference-consistency/references/c0-inventory.md). Do not require full-model GPU execution merely to inspect a bounded supplied capture; report what that narrower evidence cannot establish.

## Diagnose a mismatch

Follow [diagnostics.md](../train-inference-consistency/references/diagnostics.md). In particular:

1. Freeze the failing input, weight version, scoring semantics and necessary state. Define expected logical samples/tokens/nodes/ranks in the [comparison matrix](../train-inference-consistency/templates/comparison-matrix.md). Validate provenance, nonempty coverage, unique identities and alignment before comparing numbers.
2. Capture each actual side independently and check same-side repetition. Keep rollout, old-logprob recomputation and grad-enabled actor separate when those branches are under investigation. Inspect actor graph connectivity rather than accepting an inference surrogate.
3. Compare original-dtype payload bytes. Exercise the comparator with a controlled mismatch and missing/duplicate entries; zero-tolerance numeric comparison is not a substitute for a byte comparison. Keep mathematical tolerance checks separate.
4. Trace the semantic dependency graph to observed divergent frontiers. Missing or opaque predecessors remain unknown; the first sorted row or final output alone cannot identify a cause.
5. Replay a candidate with identical inputs and equivalent state using the actual provider where available. Report observed difference, confirmed cause and unresolved hypothesis separately. Preserve the original failing execution and instrumentation limitations.

Finish a diagnosis-only task with findings and a reproducible handoff. If product repair is already authorized, continue using [implementation and integration](../implement-train-inference-consistency/SKILL.md); do not ask again solely to change roles. Never silently change precision, disable required cache behavior or copy scores/routes to obtain a passing comparison.

## Verify a submitted implementation

Bind the exact changed files/source identity and inspect the actual loaded worker paths. Do not accept the implementer's summary or existing `pass` labels without inspecting the underlying evidence and its applicability.

Read the relevant gates from the [canonical stage map](../train-inference-consistency/references/gates.md) and apply [workflow dependencies and invalidation](../train-inference-consistency/references/workflow.md). Rerun the original failure and the cases affected by the change. Execute independent reference checks for changed arithmetic/backward and the relevant downstream integration checks. A training-loop or checkpoint test is appropriate when included in the authorized verification scope; it is not automatically required for a local comparator investigation.

For complete C0–C10 acceptance, verify all active prerequisites: C2 proxy, C3 inference cases, both C5 training branches, C6 gradients, C7 independent accuracy, C8 effective updates, C9 full model and resume, then C10 delivery. C1/C4 are not-required by default. A local passing check with missing prerequisites is a local result, not a canonical gate pass. Never waive mandatory gates to close a bounded verification task.

If verification fails, retain the failure and identify the next repair or missing observation. Product fixes remain with the implementation role when authorized; otherwise report the finding without making them. Reference/capture corrections must be recorded and old artifacts retained, rather than rewriting evidence to match the expected verdict.

## Return evidence, not only a verdict

Update the [gate record](../train-inference-consistency/templates/gate-record.md) and shared handoff with executed commands/exit codes, evidence paths and digests, expected/observed coverage, first observed frontier, comparator/reference results and precise limitations. Distinguish `fail`, `blocked`, `stale` and unexecuted work. State the next executable reproduction or the target file/function requiring work.

Use the [release report](../train-inference-consistency/templates/release-report.md) only for the scope actually assessed. No GPU access means target GPU acceptance is unverified; a CPU fixture tests the procedure only. Performance is not evaluated unless requested. Verification can be completed by reporting a failure; it does not have to manufacture a pass.
