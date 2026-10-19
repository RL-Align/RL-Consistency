"""Real tiny provider architectures, never evidence for full Qwen model acceptance."""
from contextlib import redirect_stdout
from importlib.util import find_spec
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl_consistency.common import ContractError, read_json
from rl_consistency.runtime.models import catalog, validate_profile, validate_provider_environment

AVAILABLE = find_spec("torch") is not None and find_spec("transformers") is not None


def tiny_model(family, dtype=None):
    import torch
    from transformers import (Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM,
                              Qwen3_5TextConfig, Qwen3_5ForCausalLM)
    torch.manual_seed(17)
    base = dict(vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
                attn_implementation="eager", experts_implementation="eager")
    if family == "qwen3":
        model = Qwen3ForCausalLM(Qwen3Config(**base))
    elif family == "qwen3_moe":
        model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**base, num_experts=4, num_experts_per_tok=2,
                                                moe_intermediate_size=16))
    else:
        config = Qwen3_5TextConfig(**base, layer_types=["linear_attention", "full_attention"],
                                  linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
                                  linear_num_key_heads=2, linear_num_value_heads=4,
                                  rope_parameters=dict(rope_type="default", rope_theta=10000.,
                                                       partial_rotary_factor=0.5, mrope_section=[1, 1, 0]))
        model = Qwen3_5ForCausalLM(config)
    return model.to(dtype=dtype or torch.float32).eval()


REQUESTS = [{"id": "a", "input_ids": [1, 2, 3]}, {"id": "b", "input_ids": [4, 5]}]


class RuntimeCatalogTests(unittest.TestCase):
    def test_catalog_has_three_uncertified_independent_profiles(self):
        self.assertEqual(len(catalog()), 3)
        self.assertTrue(all(not r["rl_kernel_required"] and not r["certified"] for r in catalog()))

    def test_profiles_bind_actual_architecture(self):
        fixtures = Path(__file__).parent / "fixtures"
        self.assertEqual(validate_profile("Qwen/Qwen3.8-27B", read_json(fixtures / "qwen38-27b-config.json")),
                         "qwen3_5_text")
        self.assertEqual(validate_profile("Qwen/Qwen3-30B-A3B", read_json(fixtures / "qwen3-30b-a3b-config.json")),
                         "qwen3_moe")
        with self.assertRaises(ContractError):
            validate_profile("Qwen/Qwen3-8B", {"model_type": "qwen3", "hidden_size": 32})
        with self.assertRaises(ContractError):
            validate_profile("unknown", {})

    def test_optional_hybrid_provider_is_rejected(self):
        with patch("rl_consistency.runtime.models.find_spec", side_effect=lambda x: object() if x == "fla" else None):
            with self.assertRaisesRegex(ContractError, "fla"):
                validate_provider_environment("qwen3_5_text")


@unittest.skipUnless(AVAILABLE, "optional Torch + Transformers runtime required")
class SharedRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.old_threads)

    def test_three_real_architectures_two_effective_updates(self):
        from rl_consistency.runtime.runner import run_loaded
        for family in ("qwen3", "qwen3_moe", "qwen3_5_text"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as d:
                result = run_loaded(tiny_model(family), REQUESTS, Path(d) / "run",
                                    identity={"fixture": "tiny-architecture", "family": family}, updates=2)
                self.assertEqual(result["verdict"], "equal_observed", result.get("error"))
                self.assertEqual(len(result["rounds"]), 3)
                self.assertEqual(len({r["state_sha256"] for r in result["rounds"]}), 3)
                self.assertFalse(result["certified"])
                self.assertFalse(result["full_model_gpu_verified"])
                for r in result["rounds"]:
                    check = read_json(Path(d) / "run" / r["check"]["path"])
                    self.assertEqual(len(check["comparisons"]), 8)
                    self.assertGreater(check["parameters_with_nonzero_gradient"], 0)
                    if family == "qwen3_moe":
                        self.assertEqual(len(check["comparisons"][0]["observation"]["routing"]), 2)

    def test_bfloat16_generation_and_actor_on_all_architectures(self):
        import torch
        from rl_consistency.runtime.execution import SharedPrefixRuntime
        for family in ("qwen3", "qwen3_moe", "qwen3_5_text"):
            with self.subTest(family=family):
                runtime = SharedPrefixRuntime(tiny_model(family, torch.bfloat16))
                trajectories = runtime.generate(REQUESTS, max_new_tokens=1)
                self.assertGreater(runtime.check(trajectories)["parameters_with_nonzero_gradient"], 0)

    def test_outer_no_grad_does_not_change_execution_and_grouping(self):
        import torch
        from rl_consistency.runtime.execution import SharedPrefixRuntime
        runtime = SharedPrefixRuntime(tiny_model("qwen3"))
        calls = []
        handle = runtime.model.register_forward_pre_hook(
            lambda module, args: calls.append((torch.is_grad_enabled(), module.training)))
        try:
            alone = runtime.generate(REQUESTS[:1], max_new_tokens=2)
            with torch.no_grad():
                grouped = runtime.generate(REQUESTS, max_new_tokens=2)
                runtime.check(grouped)
            self.assertEqual(alone[0].observations, grouped[0].observations)
            self.assertEqual(len(calls), 14)  # Actual independent provider forwards in all phases.
            self.assertTrue(all(grad and not training for grad, training in calls))
        finally:
            handle.remove()

    def test_actual_safetensors_roundtrip_all_architectures(self):
        from rl_consistency.runtime.execution import SharedPrefixRuntime, state_hash
        from rl_consistency.runtime.models import load_checkpoint
        for family in ("qwen3", "qwen3_moe", "qwen3_5_text"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as d:
                model = tiny_model(family)
                model.save_pretrained(d)
                restored, identity = load_checkpoint(d, family)
                self.assertEqual(state_hash(restored), state_hash(model))
                self.assertIn("model.safetensors", identity["checkpoint_files"])
                runtime = SharedPrefixRuntime(restored)
                self.assertGreater(runtime.check(runtime.generate(REQUESTS, max_new_tokens=1))
                                   ["parameters_with_nonzero_gradient"], 0)

    def test_hybrid_nested_checkpoint_text_load(self):
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        from rl_consistency.runtime.models import load_checkpoint
        from rl_consistency.runtime.execution import tensor_bytes
        model = tiny_model("qwen3_5_text")
        config = Qwen3_5Config(text_config=model.config.to_dict(), vision_config={
            "depth": 1, "hidden_size": 32, "intermediate_size": 48, "num_heads": 4,
            "out_hidden_size": 32, "num_position_embeddings": 16})
        full = Qwen3_5ForConditionalGeneration(config).eval()
        with tempfile.TemporaryDirectory() as d:
            full.save_pretrained(d)
            restored, _ = load_checkpoint(d, "qwen3_5_text")
            for name, tensor in restored.model.state_dict().items():
                self.assertEqual(tensor_bytes(tensor), tensor_bytes(full.model.language_model.state_dict()[name]))
            self.assertEqual(tensor_bytes(restored.lm_head.weight), tensor_bytes(full.lm_head.weight))

    def test_cli_loads_registered_local_fixture_and_prompt(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from rl_consistency.cli import main
        from rl_consistency.runtime.models import PROFILES
        model = tiny_model("qwen3")
        dimensions = {key: getattr(model.config, key) for key in PROFILES["Qwen/Qwen3-8B"][1]}
        with tempfile.TemporaryDirectory() as d, patch.dict(PROFILES, {"tiny-qwen3-fixture": ("qwen3", dimensions)}):
            root = Path(d)
            checkpoint = root / "checkpoint"
            model.save_pretrained(checkpoint)
            backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]").save_pretrained(checkpoint)
            fixture = Path(__file__).parent.parent / "examples/runtime-tokens.json"
            for name, arguments in (("fixture", ["--fixture", str(fixture)]), ("prompt", ["--prompt", "hello world"])):
                with self.subTest(input=name), redirect_stdout(io.StringIO()) as output:
                    code = main(["runtime-check", "--model", "tiny-qwen3-fixture", "--checkpoint", str(checkpoint),
                                 "--device", "cpu", "--dtype", "float32", "--max-new-tokens", "1",
                                 "--output", str(root / name), *arguments])
                    self.assertEqual(code, 0, output.getvalue())
                    self.assertEqual(json.loads(output.getvalue())["verdict"], "equal_observed")
                    report = read_json(root / name / "result.json")
                    self.assertEqual(report["identity"]["model_profile"], "tiny-qwen3-fixture")
                    if name == "prompt":
                        self.assertEqual(report["requests"][0]["input_ids"], [1, 2])
                        self.assertIn("tokenizer.json", report["identity"]["tokenizer_files"])

    def test_missing_checkpoint_weights_rejected(self):
        from safetensors.torch import load_file, save_file
        from rl_consistency.runtime.models import load_checkpoint
        with tempfile.TemporaryDirectory() as d:
            tiny_model("qwen3").save_pretrained(d)
            path = Path(d) / "model.safetensors"
            weights = load_file(path)
            del weights["model.layers.0.self_attn.q_proj.weight"]
            save_file(weights, path)
            with self.assertRaisesRegex(ContractError, "missing_keys"):
                load_checkpoint(d, "qwen3")

    def test_weight_change_rejected_before_recompute(self):
        import torch
        from rl_consistency.runtime.execution import SharedPrefixRuntime
        runtime = SharedPrefixRuntime(tiny_model("qwen3"))
        rollout = runtime.generate(REQUESTS, max_new_tokens=1)
        with torch.no_grad():
            next(runtime.model.parameters()).add_(0.01)
        with self.assertRaisesRegex(ContractError, "weight versions"):
            runtime.check(rollout)

    def test_actual_layer_drift_is_located_and_cannot_update(self):
        from rl_consistency.runtime.execution import SharedPrefixRuntime, Divergence
        runtime = SharedPrefixRuntime(tiny_model("qwen3"))
        rollout = runtime.generate(REQUESTS, max_new_tokens=1)
        handle = runtime.model.model.layers[1].register_forward_hook(lambda m, a, output: output + 0.125)
        try:
            with self.assertRaises(Divergence) as failure:
                runtime.check(rollout)
            self.assertEqual(failure.exception.detail["first_observed_layer"], 1)
            self.assertEqual(failure.exception.detail["phase"], "old_logprob")
            self.assertTrue(all(p.grad is None for p in runtime.model.parameters()))
        finally:
            handle.remove()
        self.assertTrue(all(not layer._forward_hooks for layer in runtime.layers))

    def test_routing_compared_even_when_final_logits_match(self):
        from rl_consistency.runtime.execution import SharedPrefixRuntime, Divergence
        runtime = SharedPrefixRuntime(tiny_model("qwen3_moe"))
        rollout = runtime.generate(REQUESTS, max_new_tokens=1)
        key = next(iter(rollout[0].observations[0]["routing"]))
        rollout[0].observations[0]["routing"][key]["experts"]["data_hex"] = "00"
        with self.assertRaises(Divergence) as failure:
            runtime.check(rollout)
        self.assertEqual(failure.exception.detail["field"], "routing")

    def test_modes_context_and_token_errors_fail(self):
        import torch
        from rl_consistency.runtime.execution import SharedPrefixRuntime
        runtime = SharedPrefixRuntime(tiny_model("qwen3"))
        with torch.inference_mode(), self.assertRaisesRegex(ContractError, "inference_mode"):
            runtime.generate(REQUESTS, max_new_tokens=1)
        runtime.model.train()
        with self.assertRaisesRegex(ContractError, "eval mode"):
            runtime.generate(REQUESTS, max_new_tokens=1)
        runtime.model.eval()
        for requests in ([], [{"id": "a", "input_ids": []}], [{"id": "a", "input_ids": [32]}], REQUESTS * 2):
            with self.subTest(requests=requests), self.assertRaises(ContractError):
                runtime.generate(requests, max_new_tokens=1)

    def test_signed_zero_bytes_and_global_settings_restored(self):
        import torch
        from rl_consistency.runtime.execution import deterministic_execution, tensor_bytes
        self.assertNotEqual(tensor_bytes(torch.tensor([-0.])), tensor_bytes(torch.tensor([0.])))
        before = torch.are_deterministic_algorithms_enabled()
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with deterministic_execution():
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                raise RuntimeError("injected")
        self.assertEqual(before, torch.are_deterministic_algorithms_enabled())

    def test_failed_run_has_report_and_no_passing_round(self):
        from rl_consistency.runtime.runner import run_loaded
        with tempfile.TemporaryDirectory() as d:
            result = run_loaded(tiny_model("qwen3"), [], Path(d) / "run", identity={"fixture": "tiny"})
            self.assertEqual(result["verdict"], "error")
            self.assertEqual(result["rounds"], [])
            self.assertEqual(read_json(Path(d) / "run/result.json"), result)
            with self.assertRaises(FileExistsError):
                run_loaded(tiny_model("qwen3"), REQUESTS, Path(d) / "run", identity={})
