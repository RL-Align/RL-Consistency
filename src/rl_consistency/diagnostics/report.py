"""Offline JSON/HTML matrix: boundaries by comparison case, with raw evidence IDs."""
from __future__ import annotations

import html
import json
from pathlib import Path

from ..common import contained, digest, file_hash, publish_json, read_json, require
from .locate import diagnose
from .schema import name, validate_plan


def render_matrix(result):
    esc = lambda value: html.escape(str(value), quote=True)
    details = lambda value: "<pre>" + esc(json.dumps(value, indent=2, ensure_ascii=False)) + "</pre>"
    cases = result["cases"]
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>First-divergence matrix</title><style>',
             'body{font:15px system-ui;margin:2rem;color:#172330;background:#f5f7fa}',
             'table{border-collapse:collapse;background:white;width:100%}th,td{border:1px solid #ccd5dd;padding:12px;text-align:left;vertical-align:top}',
             'th{background:#e5ebf1}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}',
             '.equal{color:#176040}.mismatch{color:#a52828}.missing,.incomparable{color:#865400}',
             '</style><h1>First-divergence matrix</h1>',
             '<p>Native boundary diagnostics · exact original bytes · no model certification.</p>',
             '<p>Rows follow the declared dependency graph. Multiple branches can have first observations. ',
             'Opaque regions and missing ancestors bound what can be located. Expand cells for input, state and repeat checks.</p>',
             '<table><thead><tr><th>Semantic boundary</th>']
    for case in cases:
        parts.append('<th><details><summary>' + esc(case["id"]) + '</summary>' +
                     details({k: case["result"][k] for k in ("verdict", "frontier", "coverage", "sources")}) + '</details></th>')
    parts.append('</tr></thead><tbody>')
    maps = [{r["node"]: r for r in c["result"]["rows"]} for c in cases]
    for node in result["order"]:
        parts.append('<tr><th>' + esc(node) + '</th>')
        for rows in maps:
            row = rows[node]
            label = row["status"] + (' · ' + row["frontier"] if "frontier" in row else '')
            parts.append('<td class="' + esc(row["status"]) + '"><details><summary>' + esc(label) +
                         '</summary>' + details(row) + '</details></td>')
        parts.append('</tr>')
    parts.append('</tbody></table><h2>Interpretation limits</h2>' + details(cases[0]["result"]["limits"]) + '</html>')
    return ''.join(parts)


def run_matrix(manifest_path, output):
    manifest_path = Path(manifest_path).resolve()
    root, manifest = manifest_path.parent, read_json(manifest_path)
    require(isinstance(manifest, dict) and manifest.get("schema") == "diagnostic-matrix.v1", "invalid matrix manifest")
    require(name(manifest.get("plan")), "matrix plan path required")
    plan_path = contained(root, manifest["plan"])
    plan = read_json(plan_path)
    _, order = validate_plan(plan)
    cases = manifest.get("cases")
    require(isinstance(cases, list) and cases, "matrix requires comparison cases")
    results, ids = [], set()
    for case in cases:
        require(isinstance(case, dict) and name(case.get("id")) and case["id"] not in ids, "invalid/duplicate case ID")
        ids.add(case["id"])
        paths = {}
        for field in ("left", "right", "left_repeat", "right_repeat"):
            if field in case:
                require(name(case[field]), f"invalid {field} trace path")
                paths[field] = contained(root, case[field])
        require("left" in paths and "right" in paths, "each case requires left/right traces")
        results.append({"id": case["id"], "result": diagnose(plan, **paths)})
    verdict = ("mismatch" if any(c["result"]["verdict"] == "mismatch" for c in results) else
               "equal_observed" if all(c["result"]["verdict"] == "equal_observed" for c in results) else "inconclusive")
    result = {"schema": "diagnostic-matrix-result.v1", "certified": False, "verdict": verdict,
              "manifest_sha256": file_hash(manifest_path), "plan_sha256": digest(plan),
              "plan_source": {"path": str(plan_path), "sha256": file_hash(plan_path)},
              "order": order, "cases": results}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    publish_json(output / "matrix.json", result)
    with (output / "matrix.html").open("x", encoding="utf-8") as stream:
        stream.write(render_matrix(result))
    return {"verdict": verdict, "certified": False, "json": str(output / "matrix.json"),
            "html": str(output / "matrix.html")}
