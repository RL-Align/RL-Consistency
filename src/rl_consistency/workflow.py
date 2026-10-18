"""Immutable run inputs, evidence snapshots, dependency invalidation and review records."""
from __future__ import annotations

import contextlib
import fcntl
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .common import ContractError, contained, digest, file_hash, read_json, require, write_json
from .contracts import validate_recipe
from .gates import DEPENDENCIES, evaluate


def now():
    return datetime.now(timezone.utc).isoformat()


@contextlib.contextmanager
def lock(root):
    with (root / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def initialize(root, recipe):
    validate_recipe(recipe, bound=True)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "attempts").mkdir()
    (root / "reviews").mkdir()
    write_json(root / "recipe.json", recipe)
    write_json(root / "manifest.json", {"schema": "rl-consistency.run.v2", "id": uuid.uuid4().hex,
                                       "created": now(), "recipe_sha256": digest(recipe)})
    return root


def _recipe(root):
    recipe = read_json(root / "recipe.json")
    require(digest(recipe) == read_json(root / "manifest.json")["recipe_sha256"],
            "run recipe changed; initialize a new run")
    return validate_recipe(recipe, bound=True)


def _records(root):
    records = []
    for path in (root / "attempts").glob("*/result.json"):
        result = read_json(path)
        intact = all(contained(path.parent, name).is_file() and file_hash(contained(path.parent, name)) == sha
                     for name, sha in result["files"].items())
        result["intact"] = intact
        records.append(result)
    return sorted(records, key=lambda x: x["sequence"])


def status(root):
    root = Path(root)
    recipe = _recipe(root)
    latest = {}
    for result in _records(root):
        latest[result["gate"]] = result
    states = {}
    for gate, dependencies in DEPENDENCIES.items():
        result = latest.get(gate)
        if result is None:
            states[gate] = {"status": "pending"}
            continue
        valid = result["intact"] and result["recipe_sha256"] == digest(recipe)
        for dependency in dependencies:
            valid = valid and states[dependency]["status"] == "passed" and (
                result["dependencies"].get(dependency) == latest.get(dependency, {}).get("id"))
        states[gate] = {"status": ("passed" if result["passed"] else "failed") if valid else "invalidated",
                        "id": result["id"], "result": result}
    reviews = []
    for path in sorted((root / "reviews").glob("*.json")):
        review = read_json(path)
        valid = review["recipe_sha256"] == digest(recipe) and all(
            states[g]["status"] == "passed" and states[g]["id"] == identity
            for g, identity in review["gate_ids"].items())
        reviews.append({**review, "valid": valid})
    complete = all(v["status"] == "passed" for v in states.values())
    reviewed = {r["kind"] for r in reviews if r["valid"]}
    return {"schema": "rl-consistency.status.v2", "recipe": recipe["id"],
            "run_id": read_json(root / "manifest.json")["id"], "gates": states, "reviews": reviews,
            "all_gates_passed": complete,
            "release_eligible": complete and {"baseline", "precision", "recipe", "release"} <= reviewed,
            "trust_note": "Local review records are declarations, not authenticated signatures; inspect worker artifacts."}


def _has_review(snapshot, kind):
    return any(r["kind"] == kind and r["valid"] for r in snapshot["reviews"])


def record_gate(root, gate, evidence_path):
    root, evidence_path = Path(root), Path(evidence_path).resolve()
    require(gate in DEPENDENCIES, "unknown gate")
    with lock(root):
        recipe = _recipe(root)
        current = status(root)
        for dep in DEPENDENCIES[gate]:
            require(current["gates"][dep]["status"] == "passed", f"{gate} requires {dep}")
        if gate == "C7":
            require(_has_review(current, "precision"), "record human precision review before C7")
        if gate == "C9":
            require(_has_review(current, "recipe"), "record human full-model recipe review before C9")
        if gate == "C10":
            require(_has_review(current, "baseline"), "record human baseline review before C10")
        identity = uuid.uuid4().hex
        attempt = root / "attempts" / identity
        attempt.mkdir()
        (attempt / "artifacts").mkdir()
        files = {}
        shutil.copyfile(evidence_path, attempt / "evidence.json")
        files["evidence.json"] = file_hash(attempt / "evidence.json")
        evidence = read_json(attempt / "evidence.json")
        cache = {}

        def artifact(relative):
            if relative not in cache:
                source = contained(evidence_path.parent, relative)
                name = "artifacts/" + digest(relative) + ".json"
                shutil.copyfile(source, attempt / name)
                files[name] = file_hash(attempt / name)
                cache[relative] = read_json(attempt / name)
            return cache[relative]

        try:
            baseline = current["gates"]["C1"].get("result", {}).get("detail", {}).get("benchmark")
            result = evaluate(gate, evidence, recipe, artifact, baseline)
        except (ContractError, ValueError, KeyError, TypeError, OSError) as exc:
            result = {"gate": gate, "passed": False, "error": str(exc), "detail": {}}
        result.update({"id": identity, "created": now(), "sequence": len(_records(root)) + 1,
                       "recipe_sha256": digest(recipe), "files": files,
                       "dependencies": {dep: current["gates"][dep]["id"] for dep in DEPENDENCIES[gate]}})
        write_json(attempt / "result.json", result)
        return result


def review(root, kind, reviewer, note):
    """Record a real human's decision. This command does not authenticate identity."""
    root = Path(root)
    require(kind in {"baseline", "precision", "recipe", "release"}, "unknown review kind")
    require(reviewer.strip() and note.strip(), "reviewer and decision rationale required")
    with lock(root):
        current = status(root)
        gates = {"baseline": ("C1",), "precision": ("C0",), "recipe": ("C0", "C5", "C7", "C8"),
                 "release": tuple(DEPENDENCIES)}[kind]
        require(all(current["gates"][g]["status"] == "passed" for g in gates), "review prerequisites incomplete")
        if kind == "release":
            require(all(_has_review(current, k) for k in ("baseline", "precision", "recipe")), "earlier human reviews missing")
        value = {"id": uuid.uuid4().hex, "kind": kind, "reviewer": reviewer, "note": note,
                 "created": now(), "recipe_sha256": digest(_recipe(root)),
                 "gate_ids": {g: current["gates"][g]["id"] for g in gates}}
        write_json(root / "reviews" / (value["id"] + ".json"), value)
        return value
