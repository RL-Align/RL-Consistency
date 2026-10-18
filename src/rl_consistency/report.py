"""Self-contained HTML; escape all content, embed no executable scripts."""
import html
import json
from pathlib import Path
from .workflow import status


def render(root, output):
    snapshot = status(root)
    rows = "".join(f"<tr><td>{html.escape(g)}</td><td>{html.escape(v['status'])}</td></tr>"
                   for g, v in snapshot["gates"].items())
    payload = html.escape(json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False))
    verdict = "Eligible for release review record inspection" if snapshot["release_eligible"] else "Not certified"
    Path(output).write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RL-Consistency report</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:48px auto;padding:0 24px;color:#172b42}}
h1{{letter-spacing:-.04em}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f9;padding:20px}}.badge{{padding:12px;background:#eef3fc}}</style>
<h1>RL-Consistency</h1><p>{html.escape(snapshot['recipe'])}</p><p class="badge">{verdict}</p>
<p>Scope: the pinned recipe only. A passing launcher is not model certification.</p>
<table><tr><th>Gate</th><th>Status</th></tr>{rows}</table><h2>Evidence and review records</h2><pre>{payload}</pre></html>''', encoding="utf-8")
    return snapshot
