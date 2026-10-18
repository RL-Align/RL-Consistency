# Human review and recipe release

Record baseline review after native tuning, precision review after the numerical scope/tolerances and training budget are fixed, recipe review after proxy C5/C7/C8 and before full-model C9 acceptance, and release review only after all required gates pass. Each record binds the recipe and gate attempt IDs. A rerun invalidates decisions whose bound evidence changed.

The local `review` CLI is a declaration recorder, not identity verification or a cryptographic signature. The agent must have the actual human decision before calling it. Final maintainers must inspect real worker traces, manifests, independent reference provenance and benchmark raw data. JSON producer assertions alone cannot prove that GPU work happened.

The initial release has no verified recipes, automatic promotion, hosted trust store or signed registry. Do not edit a candidate's status field to simulate certification. A future verified registry must bind the recipe hash, full evidence digest, authenticated reviewer decision and artifact origin, and revoke records when the numerical/environment identity changes.

Do not promise universal consistency, reward improvement or performance gains. Advertise only validated model/backend/framework/precision/topology/input ranges and list unsupported cases explicitly.
