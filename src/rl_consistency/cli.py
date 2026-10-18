"""CPU-safe CLI. Accelerator jobs require an explicit `run` command."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from . import __version__
from .common import ContractError, digest, file_hash, read_json
from .contracts import unbound_fields, validate_recipe
from . import adapters, workflow, rocm, model_setup
from .compare import compare_traces
from .report import render
from .rlk_import import import_sidecars


def parser():
    p = argparse.ArgumentParser(prog="rl-consistency")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list bundled candidate recipes (none are certified)")
    for name in ("inspect-model", "prepare-model"):
        s = sub.add_parser(name, help="inspect/prepare registered Hybrid or MoE models; no GPU launch")
        s.add_argument("--model", required=True)
        s.add_argument("--config", type=Path, required=True)
        if name == "prepare-model":
            s.add_argument("--backend", choices=("cuda", "rocm"), required=True)
            s.add_argument("--output", type=Path, required=True)
            s.add_argument("--proxy-layers", type=int, help="model default: Hybrid 8, MoE 4")
    export = sub.add_parser("export-recipe", help="copy a candidate recipe for local binding")
    export.add_argument("id")
    export.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("check", help="validate a recipe; --bound requires full identity")
    check.add_argument("--recipe", type=Path, required=True)
    check.add_argument("--bound", action="store_true")
    h = sub.add_parser("hash", help="hash raw files or canonical JSON")
    h.add_argument("path", type=Path)
    h.add_argument("--json", action="store_true")
    init = sub.add_parser("init", help="create an immutable run from a bound recipe")
    init.add_argument("--recipe", type=Path, required=True)
    init.add_argument("--run-dir", type=Path, required=True)
    gate = sub.add_parser("gate", help="validate and snapshot gate evidence")
    gate.add_argument("gate", choices=list(workflow.DEPENDENCIES))
    gate.add_argument("--run-dir", type=Path, required=True)
    gate.add_argument("--evidence", type=Path, required=True)
    for name in ("status", "report", "review"):
        s = sub.add_parser(name)
        s.add_argument("--run-dir", type=Path, required=True)
        if name == "report":
            s.add_argument("--output", type=Path, required=True)
        if name == "review":
            s.add_argument("kind", choices=("baseline", "precision", "recipe", "release"))
            s.add_argument("--reviewer", required=True)
            s.add_argument("--note", required=True)
    compare = sub.add_parser("compare", help="compare independently captured trace files")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--accuracy", action="store_true")
    compare.add_argument("--atol", type=float, default=0.0)
    compare.add_argument("--rtol", type=float, default=0.0)
    ingest = sub.add_parser("import-rlk", help="compare pinned Vime sidecars as diagnostics; does not pass gates")
    ingest.add_argument("--source", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--tp", type=int, required=True)
    ingest.add_argument("--cp", type=int, required=True)
    ingest.add_argument("--world-size", type=int, required=True)
    ingest.add_argument("--layout", choices=("vime-zigzag",), required=True)
    for name in ("rocm-plan", "rocm-check"):
        command = sub.add_parser(name, help="plan/run one fixed-policy ROCm diagnostic; not C0-C10 certification")
        command.add_argument("--profile", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--mode", choices=("native", "consistency"), default="consistency")
    collect = sub.add_parser("rocm-report", help="collect an existing rocm-check run without launching a GPU job")
    collect.add_argument("--run-dir", type=Path, required=True)
    for name in ("plan", "preflight", "run"):
        s = sub.add_parser(name, help=f"{name} the pinned RL-Kernel Qwen3 bridge")
        s.add_argument("--recipe", type=Path, required=True)
        s.add_argument("--rl-kernel-root", type=Path, required=True)
        s.add_argument("--profile", type=Path, required=True)
        s.add_argument("--python", type=Path, required=True)
        s.add_argument("--mode", choices=("native", "consistency"), default="consistency")
        s.add_argument("--steps", type=int, default=2)
        if name == "run":
            s.add_argument("--output", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    code = 0
    try:
        if args.command == "list":
            result = []
            for p in sorted(files("rl_consistency").joinpath("recipes").iterdir(), key=lambda x: x.name):
                if p.name.endswith(".json"):
                    recipe = json.loads(p.read_text())
                    result.append({"id": recipe["id"], "model": recipe["model"]["id"],
                                   "status": "candidate", "certified": False,
                                   "integration": recipe.get("integration", {"level": "qwen3-launch-bridge"})})
        elif args.command == "inspect-model":
            result = model_setup.inspect(args.model, args.config)
        elif args.command == "prepare-model":
            result = model_setup.prepare(args.model, args.config, args.backend, args.output,
                                         proxy_layers=args.proxy_layers)
        elif args.command == "export-recipe":
            choices = {p.name.removesuffix(".json"): p for p in files("rl_consistency").joinpath("recipes").iterdir()
                       if p.name.endswith(".json")}
            if args.id not in choices:
                raise ContractError("unknown bundled recipe")
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(choices[args.id].read_text())
            result = {"output": str(args.output), "status": "candidate"}
        elif args.command == "check":
            recipe = validate_recipe(read_json(args.recipe), bound=args.bound)
            result = {"valid_schema": True, "recipe_sha256": digest(recipe),
                      "unbound_fields": unbound_fields(recipe), "certified": False}
        elif args.command == "hash":
            result = {"sha256": digest(read_json(args.path)) if args.json else file_hash(args.path)}
        elif args.command == "init":
            result = {"run_dir": str(workflow.initialize(args.run_dir, read_json(args.recipe)))}
        elif args.command == "gate":
            result = workflow.record_gate(args.run_dir, args.gate, args.evidence)
            code = 0 if result["passed"] else 1
        elif args.command == "status":
            result = workflow.status(args.run_dir)
        elif args.command == "review":
            result = workflow.review(args.run_dir, args.kind, args.reviewer, args.note)
        elif args.command == "report":
            render(args.run_dir, args.output)
            result = {"output": str(args.output)}
        elif args.command == "compare":
            result = compare_traces(read_json(args.left), read_json(args.right), accuracy=args.accuracy,
                                    atol=args.atol, rtol=args.rtol)
            code = 0 if result["passed"] else 1
        elif args.command == "import-rlk":
            result = import_sidecars(args.source, args.output, tp=args.tp, cp=args.cp,
                                     world_size=args.world_size, layout=args.layout)
            code = 0 if result["comparison"]["passed"] else 1
        elif args.command == "rocm-plan":
            result = rocm.plan(args.profile, args.output, mode=args.mode)
        elif args.command in {"rocm-check", "rocm-report"}:
            result = rocm.run(args.profile, args.output, mode=args.mode) if args.command == "rocm-check" else rocm.collect(args.run_dir)
            code = {"pass":0, "fail":1, "inconclusive":2}[result["verdict"]]
        else:
            recipe = read_json(args.recipe)
            params = (recipe, args.rl_kernel_root, args.profile, args.python)
            if args.command == "preflight":
                result = adapters.preflight(*params)
            elif args.command == "plan":
                result = adapters.plan(*params, mode=args.mode, steps=args.steps)
            else:
                result = adapters.launch(*params, mode=args.mode, steps=args.steps, output=args.output)
                code = 0 if result["exit_code"] == 0 else 1
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return code
    except (ContractError, ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as exc:
        print(f"rl-consistency: {exc}", file=sys.stderr)
        return 2
