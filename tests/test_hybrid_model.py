"""CPU preparation/evidence tests; no model weights or GPU acceptance results."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rl_consistency.adapters import plan
from rl_consistency.cli import main
from rl_consistency.common import ContractError, digest, read_json, write_json
from rl_consistency.contracts import validate_recipe, unbound_fields
from rl_consistency.gates import evaluate
from rl_consistency.model_setup import inspect, prepare
from rl_consistency.models import get_adapter
from rl_consistency.models.qwen38 import MODEL_ID, Qwen38Adapter, HYBRID_CASES
from helpers import recipe, evidence, fixture

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tests/fixtures/qwen38-27b-config.json"


def hybrid_recipe(bound=False):
    r = read_json(ROOT / "src/rl_consistency/recipes/qwen3-8-27b-vime-rocm.json")
    if bound:
        old = recipe("rocm")
        for k in ("revision", "weights_sha256", "master_weights_sha256", "tokenizer_sha256", "config_sha256"):
            r["model"][k] = old["model"][k]
        r["model"]["config_canonical_sha256"] = digest(read_json(CONFIG))
        r["inputs"] = old["inputs"]
        r["environment"] = old["environment"]
        r["acceptance"] = old["acceptance"]
    return r


class HybridModelTests(unittest.TestCase):
    def test_official_nested_config_is_recognized_without_gpu(self):
        with patch("subprocess.run", side_effect=AssertionError("no subprocess")):
            observed = inspect(MODEL_ID, CONFIG)
        self.assertEqual(observed["layer_counts"], {"linear_attention":48, "full_attention":16})
        self.assertFalse(observed["integration"]["strict_launcher"])
        self.assertFalse(observed["certified"])

    def test_wrong_architecture_shape_precision_or_semantics_are_rejected(self):
        for key, value in (("hidden_size",4096), ("linear_num_value_heads",32), ("dtype","float16"),
                           ("output_gate_type","sigmoid"), ("attn_output_gate",1),
                           ("num_hidden_layers",64.0), ("mamba_ssm_dtype","bfloat16"), ("num_experts",16)):
            with self.subTest(key=key):
                config = read_json(CONFIG); config["text_config"][key] = value
                with self.assertRaises(ContractError): Qwen38Adapter(config)
        for mutate in (lambda c:c.update(model_type="qwen3"),
                       lambda c:c.update(quantization_config={"quant_method":"fp8"}),
                       lambda c:c["text_config"]["rope_parameters"].update(rope_theta=1000000),
                       lambda c:c["text_config"]["layer_types"].__setitem__(0,"full_attention")):
            c=read_json(CONFIG); mutate(c)
            with self.assertRaises(ContractError): Qwen38Adapter(c)

    def test_model_identity_is_explicit_not_guessed_from_qwen35_shape(self):
        with self.assertRaises(ContractError): get_adapter("Qwen/Qwen3.5-27B",read_json(CONFIG))

    def test_proxy_preserves_all_non_depth_config_and_source_weights_map(self):
        c=read_json(CONFIG); c["extra_metadata"]={"retain":True}
        adapter=Qwen38Adapter(c); p=adapter.reduced_spec(list(range(8)))
        self.assertEqual(p["config"]["text_config"]["layer_types"], c["text_config"]["layer_types"][:8])
        restored=copy.deepcopy(p["config"])
        restored["text_config"]["num_hidden_layers"]=64
        restored["text_config"]["layer_types"]=c["text_config"]["layer_types"]
        self.assertEqual(restored,c)
        self.assertEqual(p["weight_layer_map"],[{"source":i,"destination":i} for i in range(8)])
        self.assertFalse(p["checkpoint_materialized"])

    def test_incomplete_cycles_reordering_or_full_model_are_not_proxy(self):
        for selected in ([0,1,2,3], list(range(5)), list(range(64)), list(range(1,9)), [0,1,2,3,4,5,7,6]):
            with self.assertRaises(ContractError): Qwen38Adapter(read_json(CONFIG)).reduced_spec(selected)

    def test_prepare_binds_only_observed_config_and_never_marks_gpu_success(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"new"
            result=prepare(MODEL_ID,CONFIG,"rocm",root)
            r=read_json(root/"recipe.json")
            self.assertEqual(r["model"]["config_canonical_sha256"],digest(read_json(CONFIG)))
            self.assertIn("model.weights_sha256",unbound_fields(r))
            self.assertIsNone(r["environment"]["sources"]["vime"])
            self.assertEqual(set(result["gates"].values()),{"not-run"})
            self.assertFalse(result["runnable_diagnostic"])
            self.assertTrue((root/"proxy/config.json").exists())
            with self.assertRaises(FileExistsError): prepare(MODEL_ID,CONFIG,"rocm",root)
            with self.assertRaises(ContractError): validate_recipe(r,bound=True)

    def test_candidate_must_preserve_hybrid_scope_operators_and_cases(self):
        for gate,cases in HYBRID_CASES.items():
            r=hybrid_recipe(); r["coverage"][gate].remove(cases[0])
            with self.assertRaises(ContractError): validate_recipe(r)
        r=hybrid_recipe(); r["operators"].remove("gated_delta_rule")
        with self.assertRaises(ContractError): validate_recipe(r)
        r=hybrid_recipe(); r["model"]["execution_scope"]["vision_training"]=True
        with self.assertRaises(ContractError): validate_recipe(r)
        r=hybrid_recipe(); r["model"]["family"]="qwen3"
        with self.assertRaises(ContractError): validate_recipe(r)

    def test_qwen3_launch_bridge_cannot_launch_hybrid_recipe(self):
        with patch("subprocess.run",side_effect=AssertionError("must not launch")):
            with self.assertRaisesRegex(ContractError,"Qwen3-8B only"):
                plan(hybrid_recipe(),"/rlk","/profile","/python")

    def test_prepare_invalid_depth_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as d:
            for count in (0,4,9,64,True,100000000):
                with self.assertRaises(ContractError): prepare(MODEL_ID,CONFIG,"rocm",Path(d)/"new",proxy_layers=count)
                self.assertFalse((Path(d)/"new").exists())

    def test_cli_export_inspect_prepare(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["export-recipe","qwen3-8-27b-vime-rocm","--output",d+"/recipe.json"]),0)
            self.assertEqual(main(["inspect-model","--model",MODEL_ID,"--config",str(CONFIG)]),0)
            self.assertEqual(main(["prepare-model","--model",MODEL_ID,"--config",str(CONFIG),
                                   "--backend","cuda","--output",d+"/prepared"]),0)
            self.assertIn("prepared-not-executed",out.getvalue())


class HybridGateTests(unittest.TestCase):
    def setUp(self):
        self.r=hybrid_recipe(bound=True)
        self.config=read_json(CONFIG)
        self.proxy=Qwen38Adapter(self.config).reduced_spec(list(range(8)))

    def c2(self):
        e=evidence(self.r,"C2")
        e.update(model_config="source.json",proxy_config="proxy.json",coverage="coverage.json")
        coverage={k:copy.deepcopy(self.proxy[k]) for k in
                  ("source_layers","selected_layers","reduced_layers","boundary_operators")}
        coverage.update(source_config_sha256=self.r["model"]["config_sha256"],
                        checkpoint_materialized=True,weight_manifest_sha256="a"*64)
        return e,{"source.json":copy.deepcopy(self.config),"proxy.json":copy.deepcopy(self.proxy["config"]),
                  "coverage.json":coverage}

    def test_representative_proxy_can_pass_machine_checks_with_complete_receipts(self):
        e,a=self.c2()
        self.assertTrue(evaluate("C2",e,self.r,a.__getitem__)["passed"])

    def test_proxy_dimension_change_incomplete_transition_or_no_weights_fails(self):
        for edit in (lambda a:a["proxy.json"]["text_config"].update(hidden_size=256),
                     lambda a:a["coverage.json"].update(selected_layers=[0,1,2,3]),
                     lambda a:a["coverage.json"].update(checkpoint_materialized=False),
                     lambda a:a["source.json"]["text_config"].update(output_gate_type="sigmoid")):
            e,a=self.c2(); edit(a)
            with self.assertRaises(ContractError): evaluate("C2",e,self.r,a.__getitem__)

    def test_c0_and_c9_cannot_omit_actual_model_config(self):
        for gate in ("C0","C9"):
            e,a=fixture(self.r,gate)
            with self.assertRaises(KeyError): evaluate(gate,e,self.r,a.__getitem__)

    def test_added_hybrid_case_mismatch_fails_even_when_common_cases_match(self):
        e,a=fixture(self.r,"C3")
        self.assertTrue(evaluate("C3",e,self.r,a.__getitem__)["passed"])
        a["gdn-chunk-recurrent-right.json"]["records"][0]["hex"]="000000c0"
        self.assertFalse(evaluate("C3",e,self.r,a.__getitem__)["passed"])
