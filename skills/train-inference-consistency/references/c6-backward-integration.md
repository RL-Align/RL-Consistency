# C6 — Integrate backward and verify gradient coverage

**Entry:** C5 independent forward parity. This gate proves backward integration/coverage; mathematical gradient acceptance follows at C7.

## Implement backward deliberately

1. Extend the operator inventory with each module's backward source: reusable framework autograd, glue around a strict forward, or a custom backward. Identify saved tensors/state, layouts, precision and gradient accumulation/reduction. Include loss/logprob, router/expert paths, convolution/recurrence, normalization and collectives.
2. Define the differentiated function. If strict forward changes rounding, fusion, quantization or discrete choices, state how backward relates to it. Document every straight-through or surrogate-gradient choice and obtain the relevant precision decision; do not silently substitute a different mathematical operation.
3. Preserve the actual forward state needed by backward, including pre/post-cast intermediates, route decisions and recurrent state where required. Recomputing it through a different numerical path needs explicit validation. Check activation checkpointing/recomputation if in scope.
4. Implement missing gradient glue/custom kernels and target-project tests. Verify actor outputs remain connected to the intended trainable parameters, no diagnostic detach/no-grad path replaces them, and state ownership/lifetimes are correct.
5. Inspect distributed gradient scaling, accumulation and reductions against the declared loss normalization and process groups. Sequence gathers with split-only backward and gathers requiring reduction are not interchangeable; verify actual replicated/sharded semantics.
6. Execute real forward/backward on required cases. Check finite gradients, expected active parameter coverage and loss normalization. For conditional MoE paths, distinguish legitimately inactive experts from lost gradients and exercise enough fixtures to cover required experts/routes.
7. Only when performance work is requested, measure native/strict forward, backward, optimizer and synchronization costs at equal work. In consistency-only mode, omit profiling and cost budgets. Slow backward is acceptable if required cases actually complete with valid gradients.

## Artifacts and exit

Retain the per-module backward map, gradient/STE contract, saved-state description, target code/tests, observed gradient coverage/finiteness and completed execution records. Pass when all required modules have executable gradient paths, expected saved-state behavior and complete gradient coverage. Timing/budget evidence is required only for an explicitly requested performance track.

Do not claim gradients are mathematically correct merely because they are finite or nonzero; C7 supplies independent accuracy. If forward semantics or provider choices change, repeat affected C5/C3 checks. Missing gradients or unresolved surrogate semantics remain blockers/failures. Missing timing or slow execution alone cannot block consistency-only acceptance; unfinished execution still cannot pass.
