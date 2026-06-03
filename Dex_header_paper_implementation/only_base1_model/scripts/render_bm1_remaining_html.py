#!/usr/bin/env python3
"""Render BM1_remaining.md as an HTML report with embedded figures and diagrams."""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _img_data_uri(path: Path) -> str | None:
    if not path.is_file():
        return None
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _md_table_to_html(block: str) -> str:
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return f"<pre>{escape(block)}</pre>"
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines]
    header = rows[0]
    sep = rows[1] if len(rows) > 1 else []
    is_sep = len(sep) > 0 and all(re.match(r"^:?-+:?$", c.strip()) for c in sep)
    body = rows[2:] if is_sep else rows[1:]
    out = ["<table><thead><tr>"]
    for cell in header:
        out.append(f"<th>{escape(cell)}</th>")
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>")
        for cell in row:
            cell_html = escape(cell)
            if cell == "✅":
                cell_html = '<span class="badge done">✅</span>'
            out.append(f"<td>{cell_html}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _md_to_body_html(md: str) -> str:
    parts: list[str] = []
    blocks = re.split(r"\n\n+", md)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if block.startswith("|"):
            parts.append(_md_table_to_html(block))
            continue
        if block.startswith("```"):
            lang, _, code = block.partition("\n")
            lang = lang.strip("`").strip() or "text"
            parts.append(
                f'<pre class="code"><code class="lang-{escape(lang)}">{escape(code.rstrip("`"))}</code></pre>'
            )
            continue
        if block.startswith("# "):
            parts.append(f"<h1>{escape(block[2:].strip())}</h1>")
            continue
        if block.startswith("## "):
            title = block[3:].strip()
            anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            parts.append(f'<h2 id="{anchor}">{escape(title)}</h2>')
            continue
        if block.startswith("### "):
            parts.append(f"<h3>{escape(block[4:].strip())}</h3>")
            continue
        if block.startswith("---"):
            parts.append("<hr>")
            continue
        para = escape(block)
        para = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", para)
        para = re.sub(r"`([^`]+)`", r"<code>\1</code>", para)
        parts.append(f"<p>{para}</p>")
    return "\n".join(parts)


def _phase_progress_from_md(md: str) -> str:
    phases: list[tuple[str, str, bool]] = []
    for m in re.finditer(r"^## Phase (\d+) — (.+?)(?:\s*✅)?\s*$", md, re.MULTILINE):
        num, title = m.group(1), m.group(2).strip()
        done = "✅" in md[m.start() : md.find("\n", m.start()) + 20]
        phases.append((f"Phase {num}", title, done))
    if not phases:
        phases = [
            ("Phase 1", "Archive + checksums", True),
            ("Phase 2", "Figures & analysis", True),
            ("Phase 3", "ONNX export", False),
            ("Phase 4", "Parity", False),
            ("Phase 5", "Thesis pack", False),
            ("Phase 6", "Optional", False),
        ]
    rows = []
    for name, desc, done in phases:
        pct = 100 if done else 0
        cls = "done" if done else "pending"
        rows.append(
            f'<div class="phase-row {cls}">'
            f'<div class="phase-label"><strong>{escape(name)}</strong> — {escape(desc)}</div>'
            f'<div class="bar"><div class="fill" style="width:{pct}%"></div></div>'
            f'<span class="status">{"Complete" if done else "Pending"}</span>'
            f"</div>"
        )
    return '<section class="viz"><h2>Phase progress</h2>' + "".join(rows) + "</section>"


def _phase_progress_html(md: str) -> str:
    return _phase_progress_from_md(md)


def _figures_gallery(archive_dir: Path, figure_index: Path | None) -> str:
    fig_dir = archive_dir / "figures"
    if not fig_dir.is_dir():
        return "<p><em>No figures directory found.</em></p>"

    captions: dict[str, str] = {}
    if figure_index.is_file():
        data = json.loads(figure_index.read_text(encoding="utf-8"))
        for item in data.get("figures", []):
            captions[item["file"]] = item.get("caption", "")

    cards = ['<section class="viz"><h2>Phase 2 figures</h2><div class="gallery">']
    for png in sorted(fig_dir.glob("*.png")):
        uri = _img_data_uri(png)
        if uri is None:
            continue
        cap = captions.get(png.name, png.stem.replace("_", " "))
        cards.append(
            f'<figure class="card">'
            f'<img src="{uri}" alt="{escape(png.name)}" loading="lazy">'
            f"<figcaption><strong>{escape(png.name)}</strong><br>{escape(cap)}</figcaption>"
            f"</figure>"
        )
    cards.append("</div></section>")
    return "".join(cards)


def _metrics_summary(archive_dir: Path) -> str:
    val_path = archive_dir / "metrics" / "metrics_val.json"
    if not val_path.is_file():
        return ""
    m = json.loads(val_path.read_text(encoding="utf-8"))
    metrics = m.get("metrics", {})
    cm = m.get("confusion_matrix", [[0, 0], [0, 0]])
    return f"""
<section class="viz metrics-cards">
  <h2>Final validation metrics</h2>
  <div class="stat-grid">
    <div class="stat"><span class="num">{metrics.get('accuracy', 0):.4f}</span><span class="lbl">Accuracy</span></div>
    <div class="stat"><span class="num">{metrics.get('f1', 0):.4f}</span><span class="lbl">F1</span></div>
    <div class="stat"><span class="num">{metrics.get('roc_auc', 0):.4f}</span><span class="lbl">ROC-AUC</span></div>
    <div class="stat"><span class="num">{m.get('n_samples', 0)}</span><span class="lbl">Val samples</span></div>
  </div>
  <table class="cm">
    <caption>Confusion matrix (threshold {m.get('threshold', 0.5)})</caption>
    <thead><tr><th></th><th>Pred benign</th><th>Pred malware</th></tr></thead>
    <tbody>
      <tr><th>True benign</th><td>{cm[0][0]}</td><td>{cm[0][1]}</td></tr>
      <tr><th>True malware</th><td>{cm[1][0]}</td><td>{cm[1][1]}</td></tr>
    </tbody>
  </table>
</section>
"""


def _thesis_snippet_section(archive_dir: Path) -> str:
    snippet_path = archive_dir / "THESIS_SNIPPET.md"
    if not snippet_path.is_file():
        return ""
    body = _md_to_body_html(_read_text(snippet_path))
    manifest_rel = f"output_archives/{archive_dir.name}/RUN_MANIFEST.json"
    return f"""
<section class="viz thesis-pack">
  <h2>Thesis pack (Phase 5)</h2>
  <p class="meta">From <code>THESIS_SNIPPET.md</code> · manifest: <code>{escape(manifest_rel)}</code></p>
  <div class="thesis-body">{body}</div>
</section>
"""


def render_html(
    md_path: Path,
    out_path: Path,
    *,
    archive_dir: Path,
) -> Path:
    md = _read_text(md_path)
    body = _md_to_body_html(md)
    figure_index = archive_dir / "figures" / "figure_index.json"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BM1 — Remaining Tasks</title>
  <style>
    :root {{
      --bg: #0f1419;
      --surface: #1a2332;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --done: #3fb950;
      --pending: #d29922;
      --border: #30363d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      margin: 0;
      padding: 2rem max(1rem, 5vw);
      max-width: 1100px;
      margin-inline: auto;
    }}
    h1 {{ color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
    h2 {{ margin-top: 2rem; color: #79c0ff; }}
    h3 {{ color: var(--muted); }}
    a {{ color: var(--accent); }}
    code, pre.code {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 0.9em;
    }}
    code {{ padding: 0.15em 0.4em; }}
    pre.code {{ padding: 1rem; overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1rem 0;
      font-size: 0.95rem;
    }}
    th, td {{ border: 1px solid var(--border); padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: var(--surface); }}
    .badge.done {{ color: var(--done); }}
    .viz {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.25rem;
      margin: 1.5rem 0;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 1rem;
    }}
    .card img {{
      width: 100%;
      height: auto;
      border-radius: 6px;
      background: #fff;
    }}
    .card figcaption {{ font-size: 0.85rem; color: var(--muted); margin-top: 0.5rem; }}
    .phase-row {{
      display: grid;
      grid-template-columns: 1fr 2fr auto;
      gap: 0.75rem;
      align-items: center;
      margin-bottom: 0.6rem;
    }}
    .bar {{
      height: 8px;
      background: var(--border);
      border-radius: 4px;
      overflow: hidden;
    }}
    .fill {{ height: 100%; background: var(--done); transition: width 0.3s; }}
    .phase-row.pending .fill {{ background: var(--pending); width: 0 !important; }}
    .status {{ font-size: 0.8rem; color: var(--muted); min-width: 5rem; text-align: right; }}
    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 1rem;
      margin-bottom: 1rem;
    }}
    .stat {{
      text-align: center;
      padding: 1rem;
      background: var(--bg);
      border-radius: 8px;
    }}
    .stat .num {{ display: block; font-size: 1.5rem; font-weight: 700; color: var(--done); }}
    .stat .lbl {{ font-size: 0.8rem; color: var(--muted); }}
    table.cm {{ max-width: 360px; }}
    .mermaid-wrap {{
      background: #fff;
      border-radius: 8px;
      padding: 1rem;
      margin: 1rem 0;
    }}
    .meta {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 2rem; }}
    .thesis-pack .thesis-body h1 {{ font-size: 1.25rem; margin-top: 0; }}
    .thesis-pack .thesis-body h2 {{ font-size: 1.05rem; margin-top: 1.25rem; }}
    .thesis-pack .thesis-body h3 {{ font-size: 0.95rem; }}
  </style>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});
  </script>
</head>
<body>
  <p class="meta">Generated from <code>{escape(md_path.name)}</code> · archive: <code>{escape(archive_dir.name)}</code></p>
  {_phase_progress_html(md)}
  {_metrics_summary(archive_dir)}
  <section class="viz">
    <h2>Execution order</h2>
    <div class="mermaid-wrap">
      <pre class="mermaid">
flowchart LR
  P1[Phase 1 Archive] --> P2[Phase 2 Figures]
  P2 --> P3[Phase 3 ONNX]
  P3 --> P4[Phase 4 Parity]
  P4 --> P5[Phase 5 Thesis]
  P5 --> P6[Phase 6 Optional]
      </pre>
    </div>
  </section>
  {_figures_gallery(archive_dir, figure_index)}
  {_thesis_snippet_section(archive_dir)}
  <main class="doc">
  {body}
  </main>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render BM1_remaining.md to HTML.")
    parser.add_argument("--md", type=Path, default=ROOT / "BM1_remaining.md")
    parser.add_argument("--out", type=Path, default=ROOT / "BM1_remaining.html")
    parser.add_argument("--archive-dir", type=Path, default=None)
    args = parser.parse_args()

    archive_dir = args.archive_dir
    if archive_dir is None:
        latest = ROOT / "output_archives" / "LATEST_RUN.txt"
        if latest.is_file():
            archive_dir = ROOT / "output_archives" / latest.read_text(encoding="utf-8").strip()
        else:
            archive_dir = ROOT / "output_archives" / "run_20260524_fresh_logged"

    out = render_html(args.md, args.out, archive_dir=archive_dir)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
