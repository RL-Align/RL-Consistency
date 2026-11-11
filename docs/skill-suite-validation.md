# Skill suite validation — 2026-10-26

This records bounded validation of the three-entrypoint split on `codex/split-consistency-skills`, based on repository commit `920dd5b`. It is not a model consistency or GPU acceptance report.

## Structural checks

- All three entrypoints passed the bundled skill-creator `quick_validate.py` frontmatter/name checks.
- Relative file links and local heading anchors resolved; every skill could reach all shared Markdown resources inside the installed suite.
- The suite was copied outside the repository for the behavioral trials, retaining only the three sibling skill directories. No root documentation was required by the trial agents.
- C0–C10 gate/ledger coverage was retained; canonical gate meanings and numerical acceptance were not relaxed. C1/C4 remain not-required by default.
- Repository deliverables remain English Markdown. `git diff --check` passed.

## Fresh-agent behavioral trials

Three fresh agents received the copied suite and separate temporary CPU fixtures, with no surrounding conversation or intended answer. They could write only to their own fixture directories; no external services, GPU execution or production repositories were used. The parent inspected generated artifacts and reran the executable checks.

| Entrypoint | Fixture | Observed behavior |
|---|---|---|
| Diagnosis and verification | Two float64 score captures and a numeric comparator; first values were positive and negative zero | Found the byte mismatch despite numeric equality, preserved the original comparator/captures, and reported unknown producer provenance and upstream cause. Byte verifier exited 1 as expected; 10 comparator controls passed. No canonical model pass claimed. |
| Implementation and integration | Independent scalar rollout/actor reductions with different accumulation orders; squared loss and analytical gradients | Reproduced the failure, repaired the common arithmetic with independently executed `math.fsum` calls, preserved the supplied verifier, and passed its two cases plus seven local regression cases and three finite-difference checks. Reported CPU-only scope and missing model/update/resume acceptance. |
| Coordination | A predecessor's release claim supported only by matching JSON scores, no old-logprob, no gradient-enabled actor and no updates | Rejected release sufficiency, performed local numeric/negative-control checks, retained the original handoff and reported the canonical correctness gates blocked. Kept performance not evaluated. |

The initial implementation trial generated a regression command that reused an evidence output directory. This exposed an actionable procedural gap despite the shared append-only evidence rule. The implementation skill now explicitly requires a fresh attempt directory or new output path and a second-invocation preservation check.

In the follow-up, the agent repaired its task-local harness and executed two successful fresh attempts. It verified 51 historical evidence files remained unchanged and the second attempt preserved all 33 files from the first. The parent reran the repaired harness and independently checked preservation of all pre-existing evidence files. This is a correction driven by the trial; the initial behavior was not treated as fully satisfactory.

## Retained local evidence and limits

The local trial root is `/private/tmp/consistency-skill-trials-hcl7ulih`. It contains the copied suite, fixture tasks, diagnosis execution records, implementation before/after captures and repeatability records, and the coordinator's audit/ledger. These temporary local artifacts are not shipped as a runtime in this repository and may be removed by host cleanup.

`tested-suite-manifest.json` records SHA-256 for every tested skill Markdown file. Its SHA-256 is `37d8b9c18082572c2cec3a34c5310602840053b5a4b96dd068bd7744c2517546`. The final tested copy was checked against the working suite; the targeted reproduction instruction was exercised in the implementation follow-up.

The trials demonstrate bounded routing, role boundaries, evidence handling and one small repair. The scalar fixture has analytical gradients, not a target framework's autograd integration. No model weights, accelerator providers, real optimizer loop, distributed path, full-model acceptance, checkpoint resume or performance were validated. Broader framework trials remain separate work.
