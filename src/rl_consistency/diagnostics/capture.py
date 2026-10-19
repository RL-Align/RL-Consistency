"""Explicit native-boundary capture and optional local PyTorch hooks."""
from __future__ import annotations

import copy
import uuid
from pathlib import Path

from ..common import digest, publish_json, require
from .schema import TensorStore, name, snapshot_state, tensor_map, validate_plan


class DiagnosticWriter:
    def __init__(self, root, plan, *, side, provenance, execution, max_bytes=64 * 1024 * 1024):
        self.plan = copy.deepcopy(plan)
        self.nodes, _ = validate_plan(self.plan)
        require(name(side), "side label required")
        require(isinstance(provenance, dict) and name(provenance.get("provider")) and
                isinstance(provenance.get("manifest"), dict) and provenance["manifest"],
                "actual provider and worker manifest required")
        require(isinstance(execution, dict) and name(execution.get("phase")) and name(execution.get("mode")),
                "actual execution phase/mode required")
        require(isinstance(execution.get("observer_effects"), list) and
                all(name(x) for x in execution["observer_effects"]), "observer effects must be explicit")
        self.metadata = copy.deepcopy({"schema": "diagnostic-trace.v1", "capture_id": str(uuid.uuid4()),
                                      "plan_sha256": digest(self.plan), "identity": self.plan["identity"],
                                      "side": side, "provenance": provenance, "execution": execution})
        self.metadata["execution"]["observer_effects"].append("synchronous_tensor_snapshot")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.store = TensorStore(self.root, max_bytes)
        self.records, self.pending = {}, {}
        self.finished, self.failed = False, False
        publish_json(self.root / "plan.json", self.plan)

    def begin(self, node, inputs, *, state, state_complete=False):
        require(not self.finished and not self.failed, "capture finalized or aborted")
        require(node in self.nodes and node not in self.records and node not in self.pending,
                "unknown or duplicate semantic invocation")
        inputs = tensor_map(inputs)
        self.pending[node] = {"node": node, "inputs": {k: self.store.put(v) for k, v in inputs.items()},
                              "state": snapshot_state(state, complete=state_complete)}

    def end(self, node, outputs):
        require(not self.finished and not self.failed and node in self.pending, "boundary has no matching begin or capture aborted")
        outputs = tensor_map(outputs)
        require(any(v.numel() > 0 for v in outputs.values()), "empty boundary output is not evidence")
        captured = {k: self.store.put(v) for k, v in outputs.items()}
        record = self.pending.pop(node)
        record["outputs"] = captured
        self.records[node] = record

    def finish(self):
        require(not self.finished and not self.pending and not self.failed, "unfinished, aborted or already finalized capture")
        require(self.records, "empty capture is not evidence")
        value = {**self.metadata, "records": list(self.records.values()),
                 "missing": [n for n in self.nodes if n not in self.records]}
        path = self.root / "trace.json"
        publish_json(path, value)
        self.finished = True
        return path


def discover_modules(model):
    """Inventory only: paths and shapes do not establish semantic equivalence."""
    seen, result = {}, []
    for path, module in model.named_modules(remove_duplicate=False):
        shared = seen.setdefault(id(module), path)
        result.append({"path": path, "type": f"{type(module).__module__}.{type(module).__qualname__}",
                       "shared_with": shared if shared != path else None,
                       "parameters": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                                      for k, v in module.named_parameters(recurse=False)}})
    return {"schema": "diagnostic-inventory.v1", "semantic_mapping_required": True, "modules": result}


def module_state(module):
    """Partial default: arbitrary caches, global RNG and engine state need a selector."""
    import torch
    return {**{f"parameter:{k}": v for k, v in module.named_parameters()},
            **{f"buffer:{k}": v for k, v in module.named_buffers()},
            "training": module.training, "grad_enabled": torch.is_grad_enabled()}


class ModuleCapture:
    """Context-managed hooks, explicit port selectors and invocation-to-node mapping.

    A selector must return the canonical logical slice, preserving dtype. Hooks
    return None and never replace values. Python hooks do not expose fused kernel
    interiors and do not establish equivalence to an uninstrumented graph run.
    """
    def __init__(self, model, writer, bindings, *, inputs, outputs, state=module_state,
                 state_complete=False):
        self.model, self.writer, self.bindings = model, writer, bindings
        self.inputs, self.outputs, self.state = inputs, outputs, state
        self.state_complete = state_complete
        self.handles, self.counts, self.stacks = [], {}, {}

    def __enter__(self):
        require(not self.handles, "hooks already installed")
        require(isinstance(self.bindings, dict) and self.bindings, "explicit semantic bindings required")
        modules = dict(self.model.named_modules(remove_duplicate=False))
        seen, nodes = set(), set()
        for path, calls in self.bindings.items():
            require(path in modules, f"unknown module: {path}")
            require(id(modules[path]) not in seen, "bind a shared module once, with separate call IDs")
            require(isinstance(calls, list) and calls and all(name(n) and n in self.writer.nodes for n in calls),
                    "bindings must list planned semantic invocations")
            require(len(set(calls)) == len(calls) and not nodes.intersection(calls), "duplicate semantic binding")
            seen.add(id(modules[path])); nodes.update(calls)
        try:
            for path in self.bindings:
                self.counts[path], self.stacks[path] = 0, []
                self.handles.append(modules[path].register_forward_pre_hook(self._pre(path), with_kwargs=True))
                self.handles.append(modules[path].register_forward_hook(self._post(path), with_kwargs=True))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def _pre(self, path):
        def capture(module, args, kwargs):
            index = self.counts[path]
            require(index < len(self.bindings[path]), f"unplanned invocation: {path}")
            node = self.bindings[path][index]
            self.writer.begin(node, self.inputs(module, args, kwargs), state=self.state(module),
                              state_complete=self.state_complete)
            self.counts[path] += 1
            self.stacks[path].append(node)
        return capture

    def _post(self, path):
        def capture(module, args, kwargs, output):
            node = self.stacks[path][-1]
            self.writer.end(node, self.outputs(module, args, kwargs, output))
            self.stacks[path].pop()
        return capture

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if exc_type is not None:
            self.writer.failed = True
        # A failed forward must propagate; the writer refuses unfinished traces.
        return False
