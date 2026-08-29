"""Builds experiments/results/report.html, a single-page dashboard, from
experiments/results/*.jsonl.

Only reads the top-level files in each family (see MANIFEST.md).
checkpoints/ files are listed in a table at the bottom, not charted.

Usage (from the repo root):
    python3 experiments/results/visualize.py
    open experiments/results/report.html
"""

from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent
OUT = RESULTS_DIR / "report.html"

# Palette (validated default, see the dataviz skill's references/palette.md)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]
GOOD, GOOD_DARK = "#0ca30c", "#0ca30c"
CRITICAL, CRITICAL_DARK = "#d03b3b", "#e66767"


def load(name: str) -> list[dict]:
    path = RESULTS_DIR / name
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def esc(s) -> str:
    return html.escape(str(s))


# SVG bars

def hbar_chart(groups, *, series_labels, unit="%", width=640, bar_h=16, gap=4,
                group_gap=16, max_val=100.0, value_fmt=None, label_w=170):
    """groups: [(group_label, [value_or_None, ...])], one value per series_labels slot."""
    value_fmt = value_fmt or (lambda v: f"{v:.0f}{unit}")
    n_series = len(series_labels)
    chart_w = width - label_w - 60
    y = 12
    rows = []
    ticks_drawn = False
    for glabel, values in groups:
        rows.append(f'<text x="0" y="{y + (n_series * (bar_h + gap) - gap) / 2 + 4}" '
                    f'class="grp-label">{esc(glabel)}</text>')
        for i, v in enumerate(values):
            by = y + i * (bar_h + gap)
            cls = f"s{i}"
            if v is None:
                rows.append(f'<text x="{label_w}" y="{by + bar_h - 4}" class="muted-note">no data</text>')
            else:
                w = max(2, (v / max_val) * chart_w) if max_val else 0
                rows.append(
                    f'<rect x="{label_w}" y="{by}" width="{w:.1f}" height="{bar_h}" rx="4" '
                    f'class="bar {cls}"><title>{esc(series_labels[i])}: {esc(value_fmt(v))}</title></rect>'
                )
                rows.append(f'<text x="{label_w + w + 6:.1f}" y="{by + bar_h - 4}" class="val-label">'
                            f'{esc(value_fmt(v))}</text>')
        y += n_series * (bar_h + gap) + group_gap
    height = y
    axis_x = label_w
    grid = f'<line x1="{axis_x}" y1="8" x2="{axis_x}" y2="{height - group_gap + 4}" class="axis"/>'
    legend = ""
    if n_series > 1:
        lx = label_w
        legend_items = []
        for i, lab in enumerate(series_labels):
            legend_items.append(
                f'<span class="legend-item"><span class="swatch s{i}"></span>{esc(lab)}</span>'
            )
        legend = f'<div class="legend">{"".join(legend_items)}</div>'
    svg = (f'<svg viewBox="0 0 {width} {height}" width="100%" '
           f'style="max-width:{width}px" role="img">{grid}{"".join(rows)}</svg>')
    return legend + svg


def stat_tiles(tiles):
    """tiles: [(label, value_str, sub, tone)]"""
    parts = []
    for label, value, sub, tone in tiles:
        tone_cls = f" tone-{tone}" if tone else ""
        sub_html = f'<div class="tile-sub">{esc(sub)}</div>' if sub else ""
        parts.append(
            f'<div class="tile{tone_cls}"><div class="tile-value">{esc(value)}</div>'
            f'<div class="tile-label">{esc(label)}</div>{sub_html}</div>'
        )
    return f'<div class="tiles">{"".join(parts)}</div>'


def table(headers, rows):
    thead = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trows = "".join(
        "<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows
    )
    return f'<table class="data-table"><thead><tr>{thead}</tr></thead><tbody>{trows}</tbody></table>'


def section(title, subtitle, body, note=None):
    note_html = f'<p class="note">{note}</p>' if note else ""
    return (f'<section class="card"><h2>{esc(title)}</h2>'
            f'<p class="subtitle">{subtitle}</p>{body}{note_html}</section>')


# Sections

def sec_feasibility():
    rows = load("feasibility_results.jsonl")
    loop_rows = load("feasibility_results_loop.jsonl")
    if not rows:
        return ""
    by_model = defaultdict(lambda: Counter())
    for r in rows:
        m = r["model"]
        by_model[m]["n"] += 1
        for k in ("parsed", "ran", "correct"):
            if r[k]:
                by_model[m][k] += 1
    order = ["llama3.2:3b", "qwen2.5:3b", "qwen2.5:32b"]
    models = [m for m in order if m in by_model] + [m for m in by_model if m not in order]
    groups = []
    for m in models:
        c = by_model[m]
        n = c["n"]
        groups.append((f"{m} (n={n})", [100 * c[k] / n for k in ("parsed", "ran", "correct")]))
    chart = hbar_chart(groups, series_labels=["parsed", "ran", "correct"], max_val=100)

    loop_html = ""
    if loop_rows:
        by_model2 = defaultdict(lambda: Counter())
        for r in loop_rows:
            m = r["model"]
            by_model2[m]["n"] += 1
            if r["correct"]:
                by_model2[m]["correct"] += 1
        lgroups = [(f"{m} (n={c['n']})", [100 * c["correct"] / c["n"]]) for m, c in by_model2.items()]
        loop_chart = hbar_chart(lgroups, series_labels=["correct"], max_val=100)
        loop_html = (f'<h3 class="subhead">loop_count (added later, not in the main 5-task grid)</h3>'
                     f'{loop_chart}')

    return section(
        "Feasibility: can a model write valid prompt-lang at all",
        "5 tasks x 5 reps per model (experiments/feasibility_live.py). "
        "parsed = valid Python syntax; ran = passed the interpreter's whitelist and executed; "
        "correct = the right whitelisted functions were called in the right branch.",
        chart + loop_html,
        note="Bigger model wins cleanly here. This is a syntax-following question, "
             "answered before turn-by-turn or AgentDojo questions become meaningful at all.",
    )


def sec_turn_by_turn():
    v2 = load("turn_by_turn_results_32b_v2.jsonl")
    r72 = load("turn_by_turn_results_72b.jsonl")
    groups = []
    for label, rows in (("qwen2.5:32b", v2), ("qwen2.5:72b", r72)):
        if not rows:
            continue
        n = len(rows)
        correct = sum(1 for r in rows if r["correct"])
        groups.append((f"{label} (n={n})", [100 * correct / n]))
    if not groups:
        return ""
    chart = hbar_chart(groups, series_labels=["correct"], max_val=100)
    return section(
        "Turn-by-turn: does showing real intermediate results change what a model writes",
        "experiments/turn_by_turn_live.py, one statement per turn, real result shown before the next. "
        "llama3.2:3b and qwen2.5:3b never produced a working turn-by-turn statement in any batch "
        "(see Feasibility above); for them this question is still open, not answered negatively.",
        chart,
        note="turn_by_turn_results_32b_v2.jsonl is a re-verification after an in/not-in operator fix "
             "(96.7%, up from 93.3% pre-fix); the pre-fix run and an even earlier harness-bug run "
             "(assigned values showed as None) are in checkpoints/, not charted here.",
    )


def sec_agentdojo_ground_truth():
    rows = load("agentdojo_ground_truth_results.jsonl")
    if not rows:
        return ""
    by_suite = defaultdict(lambda: [0, 0])
    for r in rows:
        s = r["suite"]
        by_suite[s][1] += 1
        if r["utility"] is True:
            by_suite[s][0] += 1
    groups = [(f"{s} (n={n})", [100 * ok / n]) for s, (ok, n) in sorted(by_suite.items())]
    chart = hbar_chart(groups, series_labels=["utility pass"], max_val=100)
    return section(
        "AgentDojo ground truth: utility pass rate by suite",
        "experiments/agentdojo_live.py, real AgentDojo tasks scored with AgentDojo's own utility(), "
        "97 tasks across 4 suites.",
        chart,
    )


def sec_agentdojo_boundary():
    rows = load("agentdojo_boundary_results.jsonl")
    if not rows:
        return ""
    n = len(rows)
    blocked = sum(1 for r in rows if r["blocked"])
    by_suite = defaultdict(lambda: [0, 0])
    for r in rows:
        by_suite[r["suite"]][1] += 1
        if r["blocked"]:
            by_suite[r["suite"]][0] += 1
    tiles = stat_tiles([
        ("privileged tools correctly blocked", f"{blocked}/{n}", "with an untrusted argument", "good" if blocked == n else None),
    ])
    rows_t = [(s, f"{ok}/{n}") for s, (ok, n) in sorted(by_suite.items())]
    return section(
        "AgentDojo capability boundary: does prompt-lang's CapabilityError actually fire",
        "Every privileged tool in each suite, called once with a deliberately untrusted argument.",
        tiles + table(["suite", "blocked"], rows_t),
    )


def sec_agentdojo_live():
    rows = load("agentdojo_live_results.jsonl")
    if not rows:
        return ""
    rows_t = [(r["model"], r["injected"], r["utility"], r["security_breached"]) for r in rows]
    return section(
        "AgentDojo live: baseline vs. injected, real model run",
        f"n={len(rows)}: too small to chart as a rate. Shown as raw rows on purpose.",
        table(["model", "injected", "utility", "security_breached"], rows_t),
        note="Small-n exploratory run, not a statistically meaningful rate. See Overhead "
             "measurement below for the scaled-up (n=16) version of this same banking scenario.",
    )


def sec_overhead():
    full = load("overhead_measurement_results_72b_full_banking.jsonl")
    task3 = load("overhead_measurement_results_72b_task3.jsonl")
    if not full:
        return ""
    by_path = defaultdict(lambda: {"n": 0, "util": 0, "secs": 0.0})
    for r in full:
        p = by_path[r["path"]]
        p["n"] += 1
        p["secs"] += r["seconds"]
        if r["utility"]:
            p["util"] += 1
    path_order = ["native", "prompt-lang"]
    paths = [p for p in path_order if p in by_path] + [p for p in by_path if p not in path_order]

    util_groups = [(p, [100 * by_path[p]["util"] / by_path[p]["n"]]) for p in paths]
    util_chart = hbar_chart(util_groups, series_labels=["utility pass"], max_val=100)

    max_secs = max(by_path[p]["secs"] for p in paths) * 1.1
    time_groups = [(p, [by_path[p]["secs"]]) for p in paths]
    time_chart = hbar_chart(
        time_groups, series_labels=["total wall-clock"], max_val=max_secs, unit="s",
        value_fmt=lambda v: f"{v:.0f}s",
    )

    task3_note = ""
    if task3:
        t3_by_path = {r["path"]: r for r in task3}
        rows_t = [(p, r["utility"], f'{r["seconds"]:.1f}s') for p, r in t3_by_path.items()]
        task3_note = (
            '<h3 class="subhead">user_task_3 isolated re-check (n=1 per path)</h3>'
            + table(["path", "utility", "seconds"], rows_t)
            + '<p class="note">Earlier isolated run: both paths passed. In the full 16-task run above, '
              "native failed this same task again. Direct evidence a single-task \"confirmed fixed\" "
              "result did not reproduce. Kept visible with this caveat rather than charted as fact.</p>"
        )

    n = by_path[paths[0]]["n"]
    return section(
        "Overhead measurement: prompt-lang vs. AgentDojo's native tool-calling",
        f"qwen2.5:72b, full 16-task banking suite (n={n} per path), "
        "experiments/overhead_measurement.py. Same model, same tasks, same AgentDojo utility() scoring.",
        f'<h3 class="subhead">utility pass rate</h3>{util_chart}'
        f'<h3 class="subhead">total wall-clock time, whole suite</h3>{time_chart}'
        + task3_note,
        note="This n=16 result reversed an earlier n=5 read (prompt-lang looked ahead there); "
             "the n=5 bugfix-verification snapshots that produced that earlier read are in "
             "checkpoints/ below, not charted here as if current.",
    )


def sec_defense_comparison():
    rows = load("defense_comparison_results.jsonl")
    if not rows:
        return ""
    by_def = defaultdict(lambda: [0, 0])
    for r in rows:
        key = "spotlighting" if r["defended"] else "undefended"
        by_def[key][1] += 1
        if r["security_breached"]:
            by_def[key][0] += 1
    groups = [(k, [100 * b / n]) for k, (b, n) in by_def.items()]
    chart = hbar_chart(groups, series_labels=["security breached"], max_val=100)
    return section(
        "Defense comparison: does AgentDojo's own spotlighting defense catch what "
        "prompt-lang's capability system missed",
        "Same banking injection scenario (task #57's literal-retyping gap), run through AgentDojo's "
        "own native agent_pipeline, undefended vs. spotlighting_with_delimiting.",
        chart,
        note="Spotlighting did not help in this sample: security_breached 3/3 either way. The model "
             "wrote the attacker's IBAN as a fresh string literal, leaving no data-flow trace for a "
             "prompt-side defense to catch.",
    )


def sec_describe_handle():
    rows = load("describe_handle_isolation_results.jsonl")
    if not rows:
        return ""
    by_spot = defaultdict(lambda: Counter())
    models = set()
    for r in rows:
        key = "spotlit" if r["spotlit"] else "undefended"
        by_spot[key]["n"] += 1
        by_spot[key][r["outcome"]] += 1
        models.add(r["model"])
    outcomes = sorted({r["outcome"] for r in rows})
    groups = []
    for key in ("undefended", "spotlit"):
        if key not in by_spot:
            continue
        c = by_spot[key]
        groups.append((f'{key} (n={c["n"]})', [100 * c[o] / c["n"] for o in outcomes]))
    chart = hbar_chart(groups, series_labels=outcomes, max_val=100)
    return section(
        "describe_handle isolation: does the one channel that still shows the model real text leak "
        "the attacker's IBAN",
        f"Model: {', '.join(sorted(models))}. Real poisoned bill text from AgentDojo's own environment, "
        "handed directly to describe_handle(), no agent loop.",
        chart,
        note='"attacker" = answered with the injected IBAN; "other" = refused or said it could not find '
             "one (not necessarily the correct IBAN either; describe_handle's own docstring covers "
             "this nuance in more depth).",
    )


def sec_retyping_guard():
    guard = load("retyping_guard_live_results.jsonl")
    allow = load("retyping_guard_live_allow_results.jsonl")
    if not guard or not allow:
        return ""

    def injected_breach_rate(rows):
        inj = [r for r in rows if r["injected"]]
        if not inj:
            return None
        return 100 * sum(1 for r in inj if r["security_breached"]) / len(inj)

    def utility_rate(rows):
        return 100 * sum(1 for r in rows if r["utility"]) / len(rows)

    groups_breach = [
        (f"guard active (n={len([r for r in guard if r['injected']])})", [injected_breach_rate(guard)]),
        (f"guard bypassed (n={len([r for r in allow if r['injected']])})", [injected_breach_rate(allow)]),
    ]
    chart_breach = hbar_chart(groups_breach, series_labels=["security breached, injected runs"], max_val=100)

    tiles = stat_tiles([
        ("overall utility, guard active", f"{utility_rate(guard):.0f}%", f"n={len(guard)}", None),
        ("overall utility, guard bypassed", f"{utility_rate(allow):.0f}%", f"n={len(allow)}", None),
    ])

    return section(
        "RetypingGuard + opaque handles: the literal-retyping gap's live fix",
        "qwen2.5:32b, same banking scenario. Opaque handles keep the model from ever holding the "
        "bill's raw text; RetypingGuard watches describe_handle()'s answers for a later literal match.",
        chart_breach + tiles,
        note="This run used --approve-mode deny (any flagged action is auto-denied), which is also why "
             "guard-active utility is 0% here: that specific setting trades all utility for the "
             "security result shown, not a free win. n=6 total, 3 injected per file, a small sample.",
    )


def sec_checkpoints():
    ckpt_dir = RESULTS_DIR / "checkpoints"
    files = sorted(ckpt_dir.glob("*.jsonl")) + sorted(ckpt_dir.glob("*.jsonl.bak"))
    rows = []
    for f in files:
        try:
            data = load(f"checkpoints/{f.name}")
        except Exception:
            data = []
        rows.append((f.name, len(data)))
    return (
        '<section class="card checkpoints">'
        "<h2>Superseded runs (checkpoints/)</h2>"
        '<p class="subtitle">Bugfix-verification snapshots, each superseded by a later run in the '
        "same family, see MANIFEST.md for exactly which top-level result replaces which of these "
        "and why. Listed for completeness, not charted as current findings.</p>"
        + table(["file", "rows"], rows)
        + "</section>"
    )


def build():
    sections = [
        sec_feasibility(),
        sec_turn_by_turn(),
        sec_agentdojo_ground_truth(),
        sec_agentdojo_boundary(),
        sec_agentdojo_live(),
        sec_overhead(),
        sec_defense_comparison(),
        sec_describe_handle(),
        sec_retyping_guard(),
    ]
    body = "\n".join(s for s in sections if s) + sec_checkpoints()

    html_out = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>prompt-lang experiment results</title>
<style>
:root {{
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --text: #0b0b0b; --text2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s0: {SERIES[0]}; --s1: {SERIES[1]}; --s2: {SERIES[2]}; --s3: {SERIES[3]};
  --good: {GOOD}; --critical: {CRITICAL};
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --text: #ffffff; --text2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s0: {SERIES_DARK[0]}; --s1: {SERIES_DARK[1]}; --s2: {SERIES_DARK[2]}; --s3: {SERIES_DARK[3]};
    --good: {GOOD_DARK}; --critical: {CRITICAL_DARK};
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 32px 20px 80px; background: var(--page); color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}}
.wrap {{ max-width: 760px; margin: 0 auto; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
.lede {{ color: var(--text2); margin: 0 0 32px; max-width: 62ch; }}
.card {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 20px 24px; margin-bottom: 20px;
}}
h2 {{ font-size: 16px; margin: 0 0 4px; }}
h3.subhead {{ font-size: 13px; color: var(--text2); margin: 18px 0 8px; font-weight: 600; }}
.subtitle {{ color: var(--text2); font-size: 13px; margin: 0 0 14px; max-width: 68ch; }}
.note {{ color: var(--muted); font-size: 12.5px; margin: 14px 0 0; max-width: 68ch; }}
svg {{ display: block; margin-top: 4px; }}
.grp-label {{ font-size: 12px; fill: var(--text2); }}
.val-label {{ font-size: 11.5px; fill: var(--text2); font-variant-numeric: tabular-nums; }}
.muted-note {{ font-size: 11.5px; fill: var(--muted); font-style: italic; }}
.axis {{ stroke: var(--axis); stroke-width: 1; }}
.bar {{ opacity: 0.95; }}
.bar.s0, .swatch.s0 {{ fill: var(--s0); background: var(--s0); }}
.bar.s1, .swatch.s1 {{ fill: var(--s1); background: var(--s1); }}
.bar.s2, .swatch.s2 {{ fill: var(--s2); background: var(--s2); }}
.bar.s3, .swatch.s3 {{ fill: var(--s3); background: var(--s3); }}
.legend {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 0 0 8px 170px; }}
.legend-item {{ font-size: 12px; color: var(--text2); display: inline-flex; align-items: center; gap: 6px; }}
.swatch {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; }}
.tiles {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 4px 0 12px; }}
.tile {{ border: 1px solid var(--border); border-radius: 10px; padding: 10px 16px; min-width: 140px; }}
.tile-value {{ font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }}
.tile.tone-good .tile-value {{ color: var(--good); }}
.tile.tone-critical .tile-value {{ color: var(--critical); }}
.tile-label {{ font-size: 12px; color: var(--text2); margin-top: 2px; }}
.tile-sub {{ font-size: 11px; color: var(--muted); }}
table.data-table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 4px; }}
table.data-table th, table.data-table td {{
  text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid);
  font-variant-numeric: tabular-nums;
}}
table.data-table th {{ color: var(--text2); font-weight: 600; }}
section.checkpoints {{ opacity: 0.85; }}
footer {{ color: var(--muted); font-size: 12px; margin-top: 24px; max-width: 68ch; }}
footer code {{ background: var(--grid); padding: 1px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<div class="wrap">
<h1>prompt-lang experiment results</h1>
<p class="lede">Generated from <code>experiments/results/*.jsonl</code>. Each section charts the
current top-level result for its family only; superseded bugfix-verification snapshots are listed,
not charted, at the bottom, see <code>MANIFEST.md</code> for the reasoning behind each split.</p>
{body}
<footer>Regenerate with <code>python3 experiments/results/visualize.py</code> after a new run lands.
Source data and full transcripts: <code>experiments/results/*.jsonl</code>.</footer>
</div>
</body>
</html>
"""
    OUT.write_text(html_out)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
