"""Synthetic CPU fixtures: test rejection paths, never claim GPU acceptance."""
import contextlib
import copy
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rl_consistency.adapters import plan
from rl_consistency.cli import main
from rl_consistency.common import ContractError, digest, read_json
from rl_consistency.contracts import validate_recipe, unbound_fields
from rl_consistency.gates import evaluate
from rl_consistency.model_setup import inspect, prepare
from rl_consistency.models import get_adapter
from rl_consistency.models.qwen3_moe import MODEL_ID, Qwen3MoeAdapter, MOE_CASES
from rl_consistency.models.routing import SCHEMA
from helpers import recipe, evidence, fixture

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tests/fixtures/qwen3-30b-a3b-config.json"


def moe_recipe(bound=False):
    r = read_json(ROOT / "src/rl_consistency/recipes/qwen3-30b-a3b-vime-rocm.json")
    if bound:
        old = recipe("rocm")
        for key in ("revision", "weights_sha256", "master_weights_sha256", "tokenizer_sha256", "config_sha256"):
            r["model"][key] = old["model"][key]
        r["model"]["config_canonical_sha256"] = digest(read_json(CONFIG))
        for key in ("inputs", "environment", "acceptance"):
            r[key] = old[key]
    return r


def routing_trace(r, output, layers):
    value = {"schema": SCHEMA, "independent": True, "routing_replay": False,
             "contract_sha256": digest(r["routing"]), "implementation_sha256": "a" * 64,
             "records": [{"sample": "s", "position": 0, "layer": layer,
                          "experts": list(range(8)), "weights_hex": struct.pack("<8f", *([0.125] * 8)).hex(),
                          "delivered_slots": list(range(8)), "returned_experts": list(range(8)),
                          "combine_slots": list(range(8))} for layer in layers]}
    for key in ("producer", "weights_sha256", "weight_version", "inputs_sha256", "execution_path"):
        if key in output:
            value[key] = copy.deepcopy(output[key])
    return value


def moe_fixture(r, gate):
    e, artifacts = fixture(r, gate)
    layers = list(range(48)) if gate == "C9" else r["model"]["proxy_source_layers"]
    if gate in ("C0", "C9"):
        e["model_config"] = "source.json"
        artifacts["source.json"] = read_json(CONFIG)
    e["routing_pairs"] = []
    for pair in e.get("pairs", []):
        route = {"case": pair["case"]}
        for side in ("left", "right"):
            route[side] = "routing-" + pair[side]
            artifacts[route[side]] = routing_trace(r, artifacts[pair[side]], layers)
        e["routing_pairs"].append(route)
    if gate == "C9":
        artifacts["full.json"]["layers"] = 48
        checkpoint = artifacts["checkpoint.json"]
        for kind in ("uninterrupted", "resumed"):
            checkpoint[kind + "_routing"] = kind + "-routing.json"
            artifacts[checkpoint[kind + "_routing"]] = routing_trace(r, artifacts[checkpoint[kind + "_trace"]], layers)
    return e, artifacts


class MoePreparationTests(unittest.TestCase):
    def test_official_config_and_inventory_without_subprocess(self):
        with patch("subprocess.run", side_effect=AssertionError("no GPU launcher")):
            result = inspect(MODEL_ID, CONFIG)
        self.assertEqual(result["layer_counts"], {"full_attention_moe": 48})
        self.assertEqual(result["layers"][0]["dimensions"]["num_experts"], 128)
        self.assertFalse(result["certified"])
        self.assertFalse(result["integration"]["strict_launcher"])

    def test_config_rejects_changed_shapes_precision_and_routing(self):
        for key, value in (("hidden_size", 4096), ("num_experts", 64), ("num_experts_per_tok", 4),
                           ("num_hidden_layers", 48.0), ("decoder_sparse_step", 2), ("mlp_only_layers", [0]),
                           ("norm_topk_prob", False), ("torch_dtype", "float16"), ("dtype", "float32"),
                           ("num_shared_experts", 1), ("quantization_config", {"quant_method": "fp8"}),
                           ("architectures", ["Qwen3ForCausalLM"]), ("rope_scaling", {"type": "yarn"}),
                           ("attention_bias", 0), ("use_sliding_window", True)):
            with self.subTest(key=key):
                config = read_json(CONFIG); config[key] = value
                with self.assertRaises(ContractError): Qwen3MoeAdapter(config)

    def test_model_identity_is_not_inferred_from_similar_config(self):
        with self.assertRaises(ContractError): get_adapter("Qwen/Qwen3-30B-A3B-Instruct-2507", read_json(CONFIG))
        r = moe_recipe(); r["model"]["id"] = "Qwen/Qwen3.8-27B"
        with self.assertRaises(ContractError): validate_recipe(r)

    def test_proxy_changes_only_depth_preserves_all_experts_and_metadata(self):
        config = read_json(CONFIG); config["custom_metadata"] = {"keep": True}
        proxy = Qwen3MoeAdapter(config).reduced_spec(list(range(4)))
        restored = copy.deepcopy(proxy["config"]); restored["num_hidden_layers"] = 48
        self.assertEqual(restored, config)
        self.assertEqual(proxy["config"]["num_experts"], 128)
        self.assertEqual(proxy["config"]["num_experts_per_tok"], 8)
        self.assertEqual(proxy["weight_layer_map"], [{"source": i, "destination": i} for i in range(4)])
        self.assertFalse(proxy["checkpoint_materialized"])

    def test_prepare_defaults_and_pending_identity_for_both_backends(self):
        for backend in ("cuda", "rocm"):
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "prepared"
                result = prepare(MODEL_ID, CONFIG, backend, out)
                self.assertEqual(result["model_id"], MODEL_ID)
                self.assertEqual(result["proxy_layers"], 4)
                self.assertEqual(set(result["gates"].values()), {"not-run"})
                r = read_json(out / "recipe.json")
                self.assertEqual(r["model"]["config_canonical_sha256"], digest(read_json(CONFIG)))
                self.assertIn("model.weights_sha256", unbound_fields(r))
                self.assertFalse(result["runnable_diagnostic"])
                with self.assertRaises(FileExistsError): prepare(MODEL_ID, CONFIG, backend, out)
                with self.assertRaises(ContractError): validate_recipe(r, bound=True)

    def test_invalid_depth_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as d:
            for count in (0, 3, 48, True, 100000000):
                with self.assertRaises(ContractError): prepare(MODEL_ID, CONFIG, "rocm", Path(d) / "new", proxy_layers=count)
                self.assertFalse((Path(d) / "new").exists())
        with self.assertRaises(ContractError): Qwen3MoeAdapter(read_json(CONFIG)).reduced_spec([1, 2, 3, 4])

    def test_model_args_preserve_native_moe_parameters_and_proxy_frequency(self):
        args = Qwen3MoeAdapter(read_json(CONFIG)).vime_model_args(6)
        for key, expected in (("--num-experts", "128"), ("--moe-router-topk", "8"),
                              ("--moe-ffn-hidden-size", "768"), ("--moe-router-dtype", "fp32"),
                              ("--num-layers", "6"), ("--moe-layer-freq", "[1,1,1,1,1,1]")):
            self.assertEqual(args[args.index(key) + 1], expected)
        self.assertNotIn("--use-routing-replay", args)

    def test_candidate_cannot_weaken_moe_contract(self):
        for gate, cases in MOE_CASES.items():
            r = moe_recipe(); r["coverage"][gate].remove(cases[0])
            with self.assertRaises(ContractError): validate_recipe(r)
        for change in (lambda r: r["model"]["execution_scope"].update(routing_replay=True),
                       lambda r: r["routing"].update(tie_break="unspecified"),
                       lambda r: r["operators"].remove("moe_combine"),
                       lambda r: r["target"]["training"].pop("expert_tp"),
                       lambda r: r["moe_training"].update(token_capacity_factor=1)):
            r = moe_recipe(); change(r)
            with self.assertRaises(ContractError): validate_recipe(r)

    def test_cli_prepare_and_dense_launcher_guard(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["prepare-model", "--model", MODEL_ID, "--config", str(CONFIG),
                                   "--backend", "rocm", "--output", d + "/new"]), 0)
            self.assertEqual(json.loads(out.getvalue())["proxy_layers"], 4)
        with self.assertRaisesRegex(ContractError, "Qwen3-8B only"):
            plan(moe_recipe(), "/rlk", "/profile", "/python")


class MoeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.r = moe_recipe(bound=True)

    def test_c2_preserves_experts_and_requires_materialized_weights(self):
        proxy = Qwen3MoeAdapter(read_json(CONFIG)).reduced_spec(list(range(4)))
        e = evidence(self.r, "C2")
        e.update(model_config="source", proxy_config="proxy", coverage="coverage")
        a = {"source": read_json(CONFIG), "proxy": proxy["config"], "coverage": {
            **{k: proxy[k] for k in ("source_layers", "selected_layers", "reduced_layers", "boundary_operators")},
            "source_config_sha256": self.r["model"]["config_sha256"], "checkpoint_materialized": True,
            "weight_manifest_sha256": "a" * 64}}
        self.assertTrue(evaluate("C2", e, self.r, a.__getitem__)["passed"])
        for change in (lambda a: a["proxy"].update(num_experts=8),
                       lambda a: a["coverage"].update(checkpoint_materialized=False)):
            broken = copy.deepcopy(a); change(broken)
            with self.assertRaises(ContractError): evaluate("C2", e, self.r, broken.__getitem__)

    def test_all_routing_gates_accept_complete_synthetic_receipts(self):
        for gate in ("C3", "C5", "C8", "C9"):
            e, a = moe_fixture(self.r, gate)
            result = evaluate(gate, e, self.r, a.__getitem__)
            self.assertTrue(result["passed"])
            self.assertTrue(result["detail"]["routing_comparisons"])

    def test_missing_or_replayed_routes_cannot_pass_matching_outputs(self):
        for change in (lambda e, a: e.pop("routing_pairs"),
                       lambda e, a: a["routing-repeat-right.json"].update(routing_replay=True),
                       lambda e, a: a["routing-repeat-right.json"].update(independent=False)):
            e, a = moe_fixture(self.r, "C3"); change(e, a)
            with self.assertRaises(ContractError): evaluate("C3", e, self.r, a.__getitem__)

    def test_expert_choice_or_weight_bytes_drift_fails_even_when_logprobs_match(self):
        for change in (lambda row: row.update(experts=[8,1,2,3,4,5,6,7], returned_experts=[8,1,2,3,4,5,6,7]),
                       lambda row: row.update(weights_hex=struct.pack("<8f", 0.25,0,0.125,0.125,0.125,0.125,0.125,0.125).hex()),
                       lambda row: row.update(experts=list(reversed(range(8))), returned_experts=list(reversed(range(8))))):
            e, a = moe_fixture(self.r, "C3"); change(a["routing-repeat-right.json"]["records"][0])
            result = evaluate("C3", e, self.r, a.__getitem__)
            self.assertFalse(result["passed"])
            self.assertTrue(all(x["passed"] for x in result["detail"]["comparisons"]))

    def test_invalid_ids_weights_dispatch_and_combine_receipts_fail_closed(self):
        for row in (None, [], "malformed"):
            e, a = moe_fixture(self.r, "C3")
            a["routing-repeat-right.json"]["records"][0] = row
            with self.assertRaises(ContractError): evaluate("C3", e, self.r, a.__getitem__)
        for change in (lambda row: row.update(experts=[False,1,2,3,4,5,6,7]),
                       lambda row: row.update(experts=[128,1,2,3,4,5,6,7]),
                       lambda row: row.update(experts=[0]*8),
                       lambda row: row.update(weights_hex="00"),
                       lambda row: row.update(weights_hex=struct.pack("<8f", *([float("nan")]*8)).hex()),
                       lambda row: row.update(weights_hex=struct.pack("<8f", *([0.25]*8)).hex()),
                       lambda row: row.update(combine_slots=list(reversed(range(8)))),
                       lambda row: row.update(delivered_slots=[0,1,2,3,4,5,6,6]),
                       lambda row: row.update(returned_experts=list(reversed(range(8))))):
            e, a = moe_fixture(self.r, "C3"); change(a["routing-repeat-right.json"]["records"][0])
            with self.assertRaises(ContractError): evaluate("C3", e, self.r, a.__getitem__)

    def test_full_layer_and_scored_token_coverage_is_mandatory(self):
        for change in (lambda t: t["records"].pop(),
                       lambda t: t["records"].append(copy.deepcopy(t["records"][0])),
                       lambda t: [row.update(position=1) for row in t["records"]]):
            e, a = moe_fixture(self.r, "C9"); change(a["routing-theta-0-right.json"])
            with self.assertRaises(ContractError): evaluate("C9", e, self.r, a.__getitem__)

    def test_updated_weights_and_actor_callsite_must_match_output_capture(self):
        for gate, name, change in (("C8", "theta-1", lambda t: t.update(weights_sha256="1"*64)),
                                    ("C8", "theta-2", lambda t: t.update(weight_version="0")),
                                    ("C5", "actor-forward", lambda t: t["execution_path"].update(grad_enabled=False))):
            e, a = moe_fixture(self.r, gate); change(a[f"routing-{name}-right.json"])
            with self.assertRaises(ContractError): evaluate(gate, e, self.r, a.__getitem__)

    def test_resume_requires_independent_routing_equality(self):
        e, a = moe_fixture(self.r, "C9")
        a["resumed-routing.json"]["records"][0].update(experts=[8,1,2,3,4,5,6,7], returned_experts=[8,1,2,3,4,5,6,7])
        self.assertFalse(evaluate("C9", e, self.r, a.__getitem__)["passed"])
        del a["checkpoint.json"]["resumed_routing"]
        with self.assertRaises(KeyError): evaluate("C9", e, self.r, a.__getitem__)
