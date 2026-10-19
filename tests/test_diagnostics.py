import copy
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

from rl_consistency.cli import main
from rl_consistency.common import ContractError, digest, read_json, write_json
from rl_consistency.diagnostics import diagnose, run_matrix
from rl_consistency.diagnostics.schema import validate_plan


def plan(parents=None):
    parents = parents or {"z-root": [], "layer.10": ["z-root"], "layer.2": ["layer.10"]}
    return {"schema": "diagnostic-plan.v1", "identity": {
        "weights_sha256": digest("synthetic weights"), "fixture_sha256": digest("synthetic fixture"),
        "semantics_sha256": digest("synthetic semantics"), "weight_version": "test-only"},
        "nodes": [{"id": node, "parents": dependencies, "kind": "module",
                   "scope": {"sample": "s", "positions": [1], "layer": node,
                             "boundary": "out", "call": 0, "shard": "full"}}
                  for node, dependencies in parents.items()]}


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.plan = plan()
        self.counter = 0

    def tearDown(self):
        self.tmp.cleanup()

    def trace(self, side, *, outputs=None, inputs=None, missing=(), state_complete=True, state=0, dtype="float32"):
        self.counter += 1
        root = self.root / str(self.counter)
        root.mkdir()
        outputs, inputs = outputs or {}, inputs or {}
        records = []
        for node in self.plan["nodes"]:
            key = node["id"]
            if key in missing:
                continue
            ports = {}
            for group, values in (("inputs", inputs), ("outputs", outputs)):
                value = values.get(key, 1)
                raw = struct.pack({"float32": "<f", "int64": "<q"}[dtype], value)
                filename = f"{len(records)}-{group}.bin"
                (root / filename).write_bytes(raw)
                import hashlib
                ports[group] = {"x": {"dtype": dtype, "shape": [1],
                                      "sha256": hashlib.sha256(raw).hexdigest(), "path": filename}}
            records.append({"node": key, **ports,
                            "state": {"sha256": digest(state), "complete": state_complete, "keys": ["state"]}})
        value = {"schema": "diagnostic-trace.v1", "capture_id": str(self.counter),
                 "side": side, "plan_sha256": digest(self.plan), "identity": self.plan["identity"],
                 "provenance": {"provider": side, "manifest": {"synthetic": True}},
                 "execution": {"phase": "forward", "mode": "cpu-test", "observer_effects": []},
                 "records": records, "missing": list(missing)}
        path = root / "trace.json"
        write_json(path, value)
        return path

    def modify(self, path, change):
        value = read_json(path)
        change(value)
        write_json(path, value)

    def test_dependency_order_detects_reconvergence_and_not_lexical_order(self):
        # The final output agrees; the intermediate difference must survive.
        a, b = self.trace("left"), self.trace("right", outputs={"layer.10": 2})
        result = diagnose(self.plan, a, b)
        self.assertEqual(result["frontier"], ["layer.10"])
        self.assertEqual([r["node"] for r in result["rows"]], ["z-root", "layer.10", "layer.2"])
        self.assertEqual(result["rows"][-1]["status"], "equal")
        self.assertEqual(result["rows"][1]["localization"], "same_input_state_candidate")
        self.assertEqual(result["rows"][1]["repeatability"]["left"], "unverified")
        self.assertFalse(result["certified"])

    def test_parallel_frontiers_and_later_local_candidate(self):
        self.plan = plan({"root": [], "branch-b": ["root"], "branch-a": ["root"],
                          "join": ["branch-a", "branch-b"], "later": ["join"]})
        result = diagnose(self.plan, self.trace("left"),
                          self.trace("right", outputs={"branch-a": 2, "branch-b": 2, "later": 2}))
        self.assertEqual(set(result["frontier"]), {"branch-a", "branch-b"})
        self.assertEqual(result["rows"][-1]["localization"], "same_input_state_candidate")

    def test_missing_ancestor_blocks_earliest_claim(self):
        result = diagnose(self.plan, self.trace("left", missing=["z-root"]),
                          self.trace("right", outputs={"layer.10": 2}))
        self.assertEqual(result["rows"][1]["frontier"], "upstream_unresolved")
        self.assertEqual(result["rows"][1]["unresolved_ancestors"], ["z-root"])

    def test_changed_inputs_are_not_automatically_called_propagated_error(self):
        result = diagnose(self.plan, self.trace("left"),
                          self.trace("right", outputs={"layer.10": 2}, inputs={"layer.10": 2}))
        self.assertEqual(result["rows"][1]["localization"], "input_or_state_changed")

    def test_unknown_state_is_not_local_confirmation(self):
        result = diagnose(self.plan, self.trace("left", state_complete=False),
                          self.trace("right", outputs={"layer.10": 2}))
        self.assertEqual(result["rows"][1]["state"], "unknown")
        self.assertEqual(result["rows"][1]["localization"], "unresolved_context")

    def test_equal_requires_independent_repeats(self):
        a, b = self.trace("left"), self.trace("right")
        self.assertEqual(diagnose(self.plan, a, b)["verdict"], "inconclusive")
        result = diagnose(self.plan, a, b, left_repeat=self.trace("left"), right_repeat=self.trace("right"))
        self.assertEqual(result["verdict"], "equal_observed")
        self.assertFalse(result["certified"])
        with self.assertRaises(ContractError):
            diagnose(self.plan, a, b, left_repeat=a)
        with self.assertRaises(ContractError):
            diagnose(self.plan, a, a)

    def test_within_side_instability_and_context_change(self):
        a, b = self.trace("left"), self.trace("right")
        result = diagnose(self.plan, a, b, left_repeat=self.trace("left", outputs={"z-root": 2}),
                          right_repeat=self.trace("right", inputs={"z-root": 2}))
        self.assertEqual(result["rows"][0]["repeatability"], {"left": "unstable", "right": "incomparable"})
        self.assertEqual(result["verdict"], "inconclusive")

    def test_fresh_worker_pid_is_allowed_but_changed_code_is_not(self):
        a, b, ar, br = [self.trace(s) for s in ("left", "right", "left", "right")]
        for index, path in enumerate((a, b, ar, br)):
            self.modify(path, lambda x, index=index: x["provenance"].update(manifest={
                "schema": "rl-consistency.worker.v1", "pid": index, "hostname": "same-host",
                "loaded_artifacts": [{"sha256": digest("code")}]}))
        self.assertEqual(diagnose(self.plan, a, b, left_repeat=ar, right_repeat=br)["verdict"], "equal_observed")
        self.modify(ar, lambda x: x["provenance"]["manifest"]["loaded_artifacts"][0].update(sha256=digest("different-code")))
        self.assertEqual(diagnose(self.plan, a, b, left_repeat=ar, right_repeat=br)["verdict"], "inconclusive")

    def test_identity_mismatch_is_incomparable(self):
        a, b = self.trace("left"), self.trace("right")
        self.modify(b, lambda x: x["identity"].update(weight_version="other"))
        result = diagnose(self.plan, a, b)
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertEqual(result["frontier"], [])
        self.assertTrue(all(r["status"] == "incomparable" for r in result["rows"]))

    def test_signed_zero_and_integer_routes_are_exact(self):
        for dtype, left, right in (("float32", -0., 0.), ("int64", 2**60, 2**60+1)):
            result = diagnose(self.plan, self.trace("left", outputs={"z-root": left}, dtype=dtype),
                              self.trace("right", outputs={"z-root": right}, dtype=dtype))
            self.assertEqual(result["frontier"], ["z-root"])

    def test_dtype_shape_and_missing_port_are_incomparable(self):
        a, b = self.trace("left"), self.trace("right", dtype="int64")
        self.assertEqual(diagnose(self.plan, a, b)["rows"][0]["status"], "incomparable")
        b = self.trace("right")
        self.modify(b, lambda x: x["records"][0]["outputs"]["x"].update(shape=[]))
        self.assertEqual(diagnose(self.plan, a, b)["rows"][0]["status"], "incomparable")
        self.modify(b, lambda x: x["records"][0]["outputs"].update(y=x["records"][0]["outputs"].pop("x")))
        self.assertEqual(diagnose(self.plan, a, b)["rows"][0]["status"], "incomparable")

    def test_nonfinite_and_tampered_blob_rejected(self):
        a = self.trace("left")
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ContractError):
                diagnose(self.plan, a, self.trace("right", outputs={"z-root": value}))
        b = self.trace("right")
        (b.parent / "0-outputs.bin").write_bytes(b"bad!")
        with self.assertRaises(ContractError): diagnose(self.plan, a, b)

    def test_path_escape_and_symlink_escape_rejected(self):
        a, b = self.trace("left"), self.trace("right")
        outside = self.root / "outside.bin"
        outside.write_bytes(struct.pack("<f", 1))
        self.modify(b, lambda x: x["records"][0]["outputs"]["x"].update(path="../outside.bin"))
        with self.assertRaises(ContractError): diagnose(self.plan, a, b)
        (b.parent / "link.bin").symlink_to(outside)
        self.modify(b, lambda x: x["records"][0]["outputs"]["x"].update(path="link.bin"))
        with self.assertRaises(ContractError): diagnose(self.plan, a, b)

    def test_plan_cycles_unknown_parents_duplicates_and_scope(self):
        for mutate in (lambda p: p["nodes"][0].update(parents=["layer.2"]),
                       lambda p: p["nodes"][0].update(parents=["unknown"]),
                       lambda p: p["nodes"].append(copy.deepcopy(p["nodes"][0])),
                       lambda p: p["nodes"][1].update(scope=copy.deepcopy(p["nodes"][0]["scope"])),
                       lambda p: p["nodes"][0]["scope"].update(positions=[1, 1])):
            candidate = copy.deepcopy(self.plan)
            mutate(candidate)
            with self.assertRaises(ContractError): validate_plan(candidate)

    def test_malformed_record_and_false_coverage_rejected(self):
        a = self.trace("left")
        mutations = [lambda x: x.update(records=[None]), lambda x: x.update(missing=["z-root"]),
                     lambda x: x["records"].append(copy.deepcopy(x["records"][0])),
                     lambda x: x.update(plan_sha256="0"*64)]
        for mutate in mutations:
            b = self.trace("right")
            self.modify(b, mutate)
            with self.assertRaises(ContractError): diagnose(self.plan, a, b)

    def test_opaque_region_and_missing_repeats_are_visible(self):
        self.plan["nodes"][1]["kind"] = "opaque"
        result = diagnose(self.plan, self.trace("left"), self.trace("right", outputs={"layer.10": 2}))
        self.assertEqual(result["coverage"]["opaque"], ["layer.10"])
        self.assertFalse(result["repeats_stable_observed"])

    def test_matrix_multiple_cases_html_escape_cli_and_immutable_output(self):
        paths = {k: self.trace(k.removesuffix("_repeat")) for k in ("left", "right", "left_repeat", "right_repeat")}
        write_json(self.root / "plan.json", self.plan)
        case = {"id": "<script>alert(1)</script>", **{k: str(p.relative_to(self.root)) for k, p in paths.items()}}
        mismatch = {"id": "second", "left": str(paths["left"].relative_to(self.root)),
                    "right": str(self.trace("right", outputs={"layer.10": 2}).relative_to(self.root))}
        manifest = self.root / "matrix-input.json"
        write_json(manifest, {"schema": "diagnostic-matrix.v1", "plan": "plan.json", "cases": [case, mismatch]})
        output = self.root / "report"
        result = run_matrix(manifest, output)
        self.assertEqual(result["verdict"], "mismatch")
        content = (output / "matrix.html").read_text()
        self.assertIn("&lt;script&gt;", content)
        self.assertNotIn("<script>", content)
        self.assertEqual(len(read_json(output / "matrix.json")["cases"]), 2)
        with self.assertRaises(FileExistsError): run_matrix(manifest, output)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["diagnose", "--matrix", str(manifest), "--output", str(self.root / "cli")]), 1)
            self.assertEqual(main(["diagnose", "--matrix", str(manifest), "--output", str(output)]), 2)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch capture/replay")
class HookAndReplayTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def writer(self, name, p=None, max_bytes=1024):
        from rl_consistency.diagnostics import DiagnosticWriter
        return DiagnosticWriter(self.root / name, p or plan(), side=name,
                                provenance={"provider": "test-only", "manifest": {"synthetic": True}},
                                execution={"phase": "forward", "mode": "eager", "observer_effects": []},
                                max_bytes=max_bytes)

    def test_hooks_preserve_backward_and_capture_kwargs(self):
        from rl_consistency.diagnostics import ModuleCapture
        torch = self.torch
        model = torch.nn.Linear(2, 2)
        x = torch.tensor([[1., 2.]], requires_grad=True)
        expected = model(x)
        expected.sum().backward()
        grad = x.grad.clone()
        x.grad = None
        writer = self.writer("grad", plan({"linear": []}))
        with ModuleCapture(model, writer, {"": ["linear"]},
                           inputs=lambda m, a, kw: {"x": kw["input"]},
                           outputs=lambda m, a, kw, y: {"y": y}):
            y = model(input=x)
            self.assertTrue(y.requires_grad)
            y.sum().backward()
        self.assertTrue(torch.equal(x.grad, grad))
        self.assertTrue(torch.equal(y, expected))
        self.assertFalse(model._forward_hooks)
        captured = read_json(writer.finish())
        self.assertFalse(captured["records"][0]["state"]["complete"])

    def test_shared_module_invocations_and_cleanup_on_exception(self):
        from rl_consistency.diagnostics import ModuleCapture, discover_modules
        torch = self.torch
        shared = torch.nn.Identity()
        model = torch.nn.Sequential(shared, shared)
        inventory = discover_modules(model)
        self.assertEqual(inventory["modules"][-1]["shared_with"], "0")
        writer = self.writer("calls", plan({"call0": [], "call1": ["call0"]}))
        selectors = {"inputs": lambda m, a, kw: {"x": a[0]}, "outputs": lambda m, a, kw, y: {"y": y}}
        with ModuleCapture(model, writer, {"0": ["call0", "call1"]}, **selectors):
            model(torch.ones(1))
        self.assertEqual(len(read_json(writer.finish())["records"]), 2)
        writer = self.writer("extra", plan({"call0": []}))
        with self.assertRaises(ContractError):
            with ModuleCapture(model, writer, {"0": ["call0"]}, **selectors):
                model(torch.ones(1))
        self.assertFalse(shared._forward_hooks)
        with self.assertRaises(ContractError): writer.finish()
        writer = self.writer("aliases", plan({"call0": [], "call1": []}))
        with self.assertRaises(ContractError):
            with ModuleCapture(model, writer, {"0": ["call0"], "1": ["call1"]}, **selectors): pass

    def test_inplace_snapshot_empty_output_budget_and_unfinished_capture(self):
        from rl_consistency.diagnostics.schema import read_tensor
        torch = self.torch
        writer = self.writer("inplace", plan({"op": []}))
        x = torch.ones(1)
        writer.begin("op", {"x": x}, state={})
        x.add_(1)
        writer.end("op", {"x": x})
        row = read_json(writer.finish())["records"][0]
        self.assertEqual(read_tensor(writer.root, row["inputs"]["x"]), struct.pack("<f", 1))
        with self.assertRaises(ContractError): writer.begin("op", {"x": x}, state={})
        writer = self.writer("budget", max_bytes=2)
        with self.assertRaises(ContractError): writer.begin("z-root", {"x": x}, state={})
        writer = self.writer("empty", plan({"op": []}))
        writer.begin("op", {"x": x}, state={})
        with self.assertRaises(ContractError): writer.end("op", {"x": torch.empty(0)})
        with self.assertRaises(ContractError): writer.finish()

    def test_all_dtypes_noncontiguous_and_nonfinite(self):
        from rl_consistency.diagnostics.schema import raw_tensor, validate_bytes
        torch = self.torch
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int64,
                      torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
            x = torch.ones(2, 4, dtype=dtype)[:, ::2]
            spec, raw = raw_tensor(x)
            validate_bytes(spec, raw)
            self.assertEqual(spec["shape"], [2, 2])
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.assertRaises(ContractError): raw_tensor(torch.tensor(float("nan"), dtype=dtype))

    def test_demo_executes_real_callbacks_and_detects_hidden_drift(self):
        from rl_consistency.diagnostics.demo import run_demo
        result = run_demo(self.root / "demo")
        self.assertEqual(result["replay_verdict"], "local_replay_mismatch")
        matrix = read_json(result["json"])["cases"][0]["result"]
        self.assertEqual(matrix["frontier"], ["block.10.native"])
        self.assertEqual(matrix["rows"][-1]["status"], "equal")
        self.assertTrue(matrix["repeats_stable_observed"])

    def test_replay_reset_and_unstable_or_wrong_input_cannot_confirm(self):
        from rl_consistency.diagnostics import ReplayEndpoint, replay_boundary
        torch = self.torch
        p = plan({"op": []})
        x = torch.ones(1)
        paths = []
        for side in ("left", "right"):
            writer = self.writer(side, p)
            writer.begin("op", {"x": x}, state={"counter": 0}, state_complete=True)
            writer.end("op", {"y": x})
            paths.append(writer.finish())
        provenance = read_json(paths[0])["provenance"]
        execution = {"phase": "replay", "mode": "eager", "observer_effects": []}
        resets = []
        a = ReplayEndpoint(call=lambda v: {"y": v["x"]}, reset=lambda: resets.append(1),
                           state=lambda: {}, provenance=provenance, execution=execution, state_complete=True)
        calls = []
        def changing(values):
            calls.append(1)
            return {"y": values["x"] + len(calls)}
        b = ReplayEndpoint(call=changing, reset=lambda: resets.append(1), state=lambda: {},
                           provenance=provenance, execution=execution, state_complete=True)
        result = replay_boundary(p, *paths, "op", inputs={"x": x}, left_endpoint=a, right_endpoint=b,
                                 output=self.root / "unstable")
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertEqual(len(resets), 4)
        with self.assertRaises(ContractError):
            replay_boundary(p, *paths, "op", inputs={"x": x+1}, left_endpoint=a, right_endpoint=b,
                            output=self.root / "wrong-input")


if __name__ == "__main__":
    unittest.main()
