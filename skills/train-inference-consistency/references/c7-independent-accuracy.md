# C7 — Validate independent mathematical accuracy

**Entry:** C6 integration and predeclared precision/gradient contract and tolerances. Keep bitwise equality and mathematical accuracy as separate results.

## Build and exercise references

1. Implement a reference from the intended mathematical definition using independent code and suitable higher precision (often FP64 on tractable fixtures). Do not call the tested forward/backward or reproduce its fixed reduction tree as the only accuracy oracle. Document precision limitations and which operation subset the reference covers.
2. Compare forward outputs, intermediate mathematical quantities, input gradients and parameter gradients for all required operator families. Exercise realistic shapes where feasible and small isolated fixtures for expensive high-precision checks; record the coverage distinction.
3. For gradients, use independently differentiated reference code and/or numerical finite differences on appropriate smooth cases. Check conditioning and finite-difference step sensitivity. Test discrete router ties, clipping and other discontinuities separately rather than applying a smooth derivative oracle across a decision boundary.
4. Validate loss/mask normalization and distributed gradient scaling, including communication reference behavior. An explicit fixed-tree implementation checks a collective's contract; a mathematically independent high-precision sum checks its accuracy. These serve different purposes.
5. Compare a controlled optimizer step from identical parameters, gradients and optimizer state against an independent reference for the claimed update semantics. Include master/deployment casts and quantization where applicable; do not infer optimizer correctness from changed weights alone.
6. Report max absolute/relative error and appropriate additional metrics with reference definitions, denominator policy, tolerance source and complete case results. Retain every failing fixture. Choose tolerances before observing acceptance results; a change requires a new reviewed precision contract and a new run.

## Cover numerical conditioning, not only tensor shapes

A reduced-depth proxy can produce diffuse logits while the full model produces probabilities extremely close to one, even when all layer dimensions and source weights are preserved. Include concentrated-distribution fixtures for log-softmax, entropy, loss and their gradients. Check the declared relative/denominator policy against independently differentiated references; a small absolute error does not waive a failed predeclared relative criterion.

When a full-model fixture exposes such a gap, preserve its actual logits, masks, loss and gradients, reproduce the failure with the original provider, and carry that fixture into the repaired provider's regressions. Trace precision through normalization, exp/products/reductions, casts and backward, not only the final score dtype. Higher internal precision may be a correctness repair, but it changes the numerical contract: rerun affected invariance, independent training branches, accuracy and update evidence before restoring their pass status. Longer or easier fixtures cannot replace the original failure case.

## Artifacts and exit

Retain reference source/provenance, tested source/build, fixtures, raw outputs/gradients/update state, coverage and error reports, tolerance/STE decisions and failures. Pass only when the declared forward, gradient and update accuracy scope meets its predeclared criteria independently of bitwise parity.

If both strict paths agree but fail accuracy, repair the shared numerical/gradient problem. Do not loosen a threshold or classify the discrepancy as harmless solely because reduction orders differ. Revisit C0/C3/C5/C6 as the fix requires and rerun impacted checks.

SFT/quality/reward behavior needs its own task evaluation. Operator accuracy does not establish quality improvement; bitwise parity does not establish either accuracy or reward improvement.
