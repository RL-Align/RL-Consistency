# Configuration fixture provenance

`qwen38-27b-config.json` preserves the Qwen-owned configuration retrieved on
2026-10-18 from https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json
(model repository license: Apache-2.0). Only formatting has been normalized.
It contains architecture metadata, not weights or proof of model identity.
Tests use it as a CPU input; it is not a GPU validation artifact.

`qwen3-30b-a3b-config.json` records the Qwen-owned configuration retrieved on
2026-10-18 from https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json
(model repository license: Apache-2.0). JSON formatting and equivalent numeric
spellings are normalized. It is architecture metadata only; acceptance must
bind the user's actual config, model revision and complete weights manifest.
