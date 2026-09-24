"""Self-contained interactive HTML audit reports.

One file, no external assets, no network access at render time. The data is
embedded as JSON and drawn with inline SVG, so a report can be archived
alongside a dataset and still open years later.

The report answers one question: for this signal, which regions of the spectrum
disagreed, and what does the raw data look like there. Selecting a band anywhere
in the report highlights it everywhere else, including the raw trace.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

__all__ = ["write_report", "report_payload"]


def _series(values) -> list:
    return [None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 6)
            for v in np.asarray(values).ravel()]


def _downsample(axis: np.ndarray, values: np.ndarray, max_points: int = 900):
    """Thin a trace for display without distorting its shape."""
    n = axis.size
    if n <= max_points:
        return axis, values
    step = int(np.ceil(n / max_points))
    return axis[::step], values[::step]


def report_payload(audit, sample: int | None = None) -> dict:
    """Assemble everything the HTML report needs as a plain dictionary."""
    r = audit.result
    bands = audit.bands
    conflict = audit.conflict_table()
    rank_basis = conflict.attrs.get("conflict_rank_basis", "informed_conflict")

    payload: dict = {
        "regime": r.regime,
        "n_samples": int(r.n_samples),
        "n_bands": int(bands.n_bands),
        "axis_name": bands.axis_name,
        "topology": {"name": r.topology.name, "edges": r.topology.edges.tolist()},
        "rank_basis": rank_basis,
        "accuracy": audit.accuracy,
        "bands": [
            {
                "index": int(b.index),
                "label": b.label,
                "lo": float(b.lo),
                "hi": float(b.hi),
                "n_features": int(b.n_features),
            }
            for b in bands
        ],
        "conflict": {
            c: _series(conflict[c]) if conflict[c].dtype.kind in "fiu" else conflict[c].astype(str).tolist()
            for c in conflict.columns
        },
    }

    if sample is not None:
        i = int(sample)
        payload["sample"] = {
            "index": i,
            "y_pred": str(audit.y_pred_labels[i]) if audit.y_pred_labels is not None else str(r.y_pred[i]),
            "y_true": None if audit.y_true is None else str(audit.y_true[i]),
            "correct": None if audit.correct is None else bool(audit.correct[i]),
            "confidence": float(r.proba[i].max()),
            "proba": _series(r.proba[i]),
            "dg": float(r.dg[i]),
            "dg_early": float(r.dg_early[i]),
            "redg": float(r.redg[i]),
            "stress": r.stress_history[i].T.tolist() if r.stress_history is not None else None,
            "muted": r.mute_history[i].T.astype(int).tolist() if r.mute_history is not None else None,
            "weights": r.weight_history[i].T.tolist() if r.weight_history is not None else None,
            "band_dg": _series(r.band_dg[i]),
            "band_stress": _series(r.mean_stress[i]),
            "energy": _series(r.energy_history[i]) if r.energy_history is not None else None,
        }
        if audit.X is not None:
            ax, vals = _downsample(bands.axis, audit.X[i])
            payload["signal"] = {"axis": _series(ax), "values": _series(vals)}
    else:
        summary = audit.summary()
        payload["cohort"] = {
            "dg": _series(summary["DG"]),
            "eredg": _series(summary["eREDG"]),
            "correct": summary["correct"].tolist() if "correct" in summary else None,
            "mean_stress": _series(r.mean_stress.mean(axis=0)),
        }
        if audit.X is not None:
            ax, vals = _downsample(bands.axis, audit.X.mean(axis=0))
            payload["signal"] = {"axis": _series(ax), "values": _series(vals)}

    if audit.X is not None:
        traces = {}
        for b in bands:
            block = audit.X[[sample], :][:, b.features] if sample is not None else audit.X[:, b.features]
            ax, vals = _downsample(bands.axis[b.features], block.mean(axis=0), 300)
            traces[b.label] = {"axis": _series(ax), "mean": _series(vals)}
            if sample is None and audit.y_true is not None:
                by_class = {}
                for c in np.unique(audit.y_true):
                    m = audit.y_true == c
                    _, cv = _downsample(bands.axis[b.features], audit.X[m][:, b.features].mean(axis=0), 300)
                    by_class[str(c)] = _series(cv)
                traces[b.label]["by_class"] = by_class
        payload["band_traces"] = traces
    return payload


_CSS = """
:root{--bg:#f6f7f9;--panel:#fff;--ink:#17212b;--soft:#4a5a68;--mut:#6b7d8d;
--rule:#dde4ea;--accent:#1f5f8b;--accent-w:#e6eef4;--warn:#a6392e;--warn-w:#f7e7e4;
--ok:#2e6b57;--grid:#eef2f6;}
@media(prefers-color-scheme:dark){:root{--bg:#0f1721;--panel:#16202b;--ink:#e4ebf1;
--soft:#bcc9d5;--mut:#8ea0b1;--rule:#27353f;--accent:#74b4dc;--accent-w:#17303f;
--warn:#e08878;--warn-w:#37201c;--ok:#77bfa3;--grid:#1d2833;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:28px 20px 72px}
h1{font-size:1.5rem;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:.72rem;letter-spacing:.11em;text-transform:uppercase;color:var(--mut);
margin:28px 0 10px;font-weight:600}
.sub{color:var(--soft);margin:0 0 18px}
.kpis{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:8px}
.kpi{background:var(--panel);border:1px solid var(--rule);border-radius:5px;
padding:9px 14px;min-width:110px}
.kpi .v{font-size:1.24rem;font-variant-numeric:tabular-nums;line-height:1.15}
.kpi .k{font-size:.72rem;color:var(--mut);margin-top:1px}
.kpi.bad .v{color:var(--warn)} .kpi.good .v{color:var(--ok)}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:5px;padding:14px 16px}
svg{display:block;width:100%;height:auto;overflow:visible}
table{border-collapse:collapse;width:100%;font-size:.84rem}
th{text-align:right;font-weight:600;color:var(--mut);font-size:.7rem;letter-spacing:.06em;
text-transform:uppercase;padding:7px 9px;border-bottom:1px solid var(--rule);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td{padding:7px 9px;border-bottom:1px solid var(--grid);text-align:right;
font-variant-numeric:tabular-nums}
tbody tr{cursor:pointer}
tbody tr:hover{background:var(--accent-w)}
tbody tr.sel{background:var(--accent-w);box-shadow:inset 3px 0 0 var(--accent)}
.bandrect{cursor:pointer}
.tick{font-size:10px;fill:var(--mut)}
.lbl{font-size:10px;fill:var(--mut)}
.detail{display:grid;grid-template-columns:1fr 280px;gap:16px;align-items:start}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:.72rem;
background:var(--accent-w);color:var(--accent)}
.pill.warn{background:var(--warn-w);color:var(--warn)}
dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:5px 12px;font-size:.84rem}
dt{color:var(--mut)} dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
footer{margin-top:34px;padding-top:16px;border-top:1px solid var(--rule);
color:var(--mut);font-size:.8rem}
@media(max-width:720px){.detail{grid-template-columns:1fr}}
"""

_JS = r"""
const D = window.__PYPASI__;
const $ = s => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
let selected = 0;

const el = (t, a = {}, kids = []) => {
  const n = document.createElementNS(NS, t);
  for (const k in a) n.setAttribute(k, a[k]);
  for (const c of [].concat(kids)) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  return n;
};
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fin = a => a.filter(v => v !== null && isFinite(v));
const fmt = (v, d = 4) => (v === null || !isFinite(v)) ? "—" : (+v).toFixed(d);

function scale(dom, rng) {
  const [a, b] = dom, [c, d] = rng, s = (b - a) || 1;
  return v => c + (d - c) * (v - a) / s;
}

/* ---------- signal with clickable band regions ---------- */
function drawSignal() {
  const box = $("#signal"); box.innerHTML = "";
  const sig = D.signal; if (!sig) return;
  const W = box.clientWidth || 900, H = 230, M = { t: 10, r: 8, b: 26, l: 40 };
  const xs = sig.axis, ys = sig.values;
  const x = scale([Math.min(...xs), Math.max(...xs)], [M.l, W - M.r]);
  const yv = fin(ys);
  const y = scale([Math.min(...yv), Math.max(...yv)], [H - M.b, M.t]);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
                          "aria-label": "signal with band regions" });

  D.bands.forEach((b, i) => {
    const x0 = x(b.lo), x1 = x(b.hi);
    const conf = D.conflict.conflict_rank ? D.conflict.conflict_rank[i] : null;
    const hot = conf !== null && conf <= 2;
    const r = el("rect", {
      class: "bandrect", x: x0, y: M.t, width: Math.max(1, x1 - x0), height: H - M.b - M.t,
      fill: hot ? css("--warn") : css("--accent"),
      "fill-opacity": i === selected ? 0.20 : 0.06,
      stroke: css("--rule"), "stroke-width": 1,
    });
    r.addEventListener("click", () => select(i));
    svg.appendChild(r);
    svg.appendChild(el("text", { class: "lbl", x: (x0 + x1) / 2, y: M.t + 12,
                                 "text-anchor": "middle" }, b.label));
  });

  let d = "";
  xs.forEach((v, i) => { if (ys[i] !== null) d += (d ? "L" : "M") + x(v) + " " + y(ys[i]); });
  svg.appendChild(el("path", { d, fill: "none", stroke: css("--ink"), "stroke-width": 1.4 }));

  const lo = Math.min(...xs), hi = Math.max(...xs);
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4;
    svg.appendChild(el("text", { class: "tick", x: x(v), y: H - 8, "text-anchor": "middle" },
                       v.toFixed(0)));
  }
  if (D.axis_name !== "axis")
    svg.appendChild(el("text", { class: "lbl", x: W / 2, y: H + 4, "text-anchor": "middle" }, D.axis_name));
  box.appendChild(svg);
}

/* ---------- stress heat map: bands x iterations ---------- */
function drawStress() {
  const box = $("#stress"); if (!box) return;
  box.innerHTML = "";
  const S = D.sample && D.sample.stress; if (!S) return;
  const K = S.length, T = S[0].length;
  const W = box.clientWidth || 900, cell = Math.max(5, (W - 52) / T);
  const H = K * 18 + 26;
  const flat = fin(S.flat()); const mx = Math.max(...flat, 1e-9);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "stress per band per iteration" });
  const M = D.sample.muted;
  for (let k = 0; k < K; k++) {
    svg.appendChild(el("text", { class: "lbl", x: 34, y: k * 18 + 25, "text-anchor": "end" },
                       D.bands[k].label));
    for (let t = 0; t < T; t++) {
      // Stress falls steeply once the system starts settling, so a linear ramp
      // would leave every step after the first invisible. The square root keeps
      // the late, small disagreements legible without overstating them.
      const v = Math.sqrt(Math.max(0, S[k][t]) / mx);
      const muted = M && M[k][t] === 1;
      const r = el("rect", { x: 40 + t * cell, y: k * 18 + 14, width: Math.max(1, cell - 1), height: 15,
        fill: muted ? css("--warn") : css("--accent"),
        "fill-opacity": muted ? 0.9 : Math.max(0.04, Math.min(1, v)) });
      r.appendChild(el("title", {}, `${D.bands[k].label} step ${t + 1}: stress ${fmt(S[k][t])}${muted ? " (silenced)" : ""}`));
      svg.appendChild(r);
    }
  }
  svg.appendChild(el("text", { class: "lbl", x: 40, y: H - 2 }, "negotiation step →"));
  box.appendChild(svg);
}

/* ---------- raw trace for the selected band ---------- */
function drawBandTrace() {
  const box = $("#bandtrace"); box.innerHTML = "";
  const tr = D.band_traces && D.band_traces[D.bands[selected].label];
  if (!tr) { box.textContent = "No raw signal was attached to this report."; return; }
  const W = box.clientWidth || 600, H = 170, M = { t: 10, r: 8, b: 24, l: 40 };
  const xs = tr.axis;
  const series = tr.by_class ? Object.entries(tr.by_class) : [["mean", tr.mean]];
  const all = fin(series.flatMap(([, v]) => v));
  const x = scale([Math.min(...xs), Math.max(...xs)], [M.l, W - M.r]);
  const y = scale([Math.min(...all), Math.max(...all)], [H - M.b, M.t]);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "raw signal in selected band" });
  const palette = [css("--accent"), css("--warn"), css("--ok"), css("--soft"), css("--mut")];
  series.forEach(([name, vals], i) => {
    let d = "";
    xs.forEach((v, j) => { if (vals[j] !== null) d += (d ? "L" : "M") + x(v) + " " + y(vals[j]); });
    svg.appendChild(el("path", { d, fill: "none", stroke: palette[i % palette.length], "stroke-width": 1.6 }));
    if (tr.by_class) {
      svg.appendChild(el("text", { class: "lbl", x: W - M.r, y: M.t + 12 * (i + 1),
        "text-anchor": "end", fill: palette[i % palette.length] }, String(name)));
    }
  });
  [Math.min(...xs), Math.max(...xs)].forEach((v, i) =>
    svg.appendChild(el("text", { class: "tick", x: x(v), y: H - 6,
      "text-anchor": i ? "end" : "start" }, v.toFixed(0))));
  box.appendChild(svg);
}

/* ---------- selection ---------- */
function select(i) {
  selected = i;
  document.querySelectorAll("#bandtable tbody tr").forEach((tr, k) =>
    tr.classList.toggle("sel", k === i));
  const b = D.bands[i], C = D.conflict;
  $("#bandname").textContent = `${b.label}  ·  ${b.lo.toFixed(0)}–${b.hi.toFixed(0)} ${D.axis_name}`;
  const rows = [
    ["features", b.n_features],
    ["band confidence", fmt(C.band_confidence ? C.band_confidence[i] : null, 3)],
    ["mean stress", fmt(C.mean_stress[i])],
    ["informed conflict", fmt(C.informed_conflict ? C.informed_conflict[i] : null)],
    ["silenced fraction", fmt(C.mute_frequency[i], 3)],
    ["band DG", fmt(C.band_DG[i], 3)],
    ["conflict rank", C.conflict_rank ? C.conflict_rank[i] : "—"],
  ];
  if (C.excess_stress) rows.splice(5, 0, ["excess stress (wrong − right)", fmt(C.excess_stress[i], 3)]);
  $("#banddl").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  drawBandTrace();
  drawSignal();
}

function buildTable() {
  const C = D.conflict;
  const cols = ["label", "band_confidence", "mean_stress", "informed_conflict",
                "excess_stress", "mute_frequency", "band_DG", "conflict_rank"]
                .filter(c => C[c]);
  const head = cols.map(c => `<th>${c.replace(/_/g, " ")}</th>`).join("");
  const body = D.bands.map((b, i) => "<tr>" + cols.map(c => {
    const v = C[c][i];
    return `<td>${typeof v === "number" ? fmt(v, c === "conflict_rank" ? 0 : 4) : v}</td>`;
  }).join("") + "</tr>").join("");
  $("#bandtable").innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;
  document.querySelectorAll("#bandtable tbody tr").forEach((tr, i) =>
    tr.addEventListener("click", () => select(i)));
}

function boot() {
  buildTable();
  drawStress();
  const ranked = D.conflict.conflict_rank;
  select(ranked ? ranked.indexOf(Math.min(...fin(ranked))) : 0);
}
window.addEventListener("resize", () => { drawSignal(); drawStress(); drawBandTrace(); });
boot();
"""


def _kpi(value: str, key: str, tone: str = "") -> str:
    return f'<div class="kpi {tone}"><div class="v">{value}</div><div class="k">{key}</div></div>'


def write_report(audit, path, *, sample: int | None = None, title: str | None = None) -> str:
    """Render an audit as a single self-contained HTML file.

    Parameters
    ----------
    audit
        An :class:`~pypasi.audit.AuditResult`.
    path
        Destination file.
    sample
        Render the per-signal view for this sample. ``None`` renders the cohort
        view, in which band traces are averaged over all signals.
    title
        Heading. Defaults to something descriptive.

    Returns
    -------
    str
        The path written.
    """
    payload = report_payload(audit, sample)
    r = audit.result

    if sample is None:
        heading = title or f"Band conflict audit · {r.n_samples} signals"
        sub = (
            f"Regime {r.regime} · {audit.bands.n_bands} bands · "
            f"{r.topology.name} topology"
        )
        kpis = [_kpi(f"{r.n_samples}", "signals"), _kpi(r.regime, "regime")]
        if audit.accuracy is not None:
            kpis.append(_kpi(f"{audit.accuracy:.3f}", "accuracy"))
        kpis += [
            _kpi(f"{r.dg.mean():.1f}", "mean DG"),
            _kpi(f"{r.mute_frequency.mean():.3f}", "silenced fraction"),
            _kpi(audit.top_conflict_bands(1)[0], "top conflict band", "bad"),
        ]
    else:
        s = payload["sample"]
        heading = title or f"Band conflict audit · signal #{sample}"
        sub = (
            f"Regime {r.regime} · {audit.bands.n_bands} bands · "
            f"{r.topology.name} topology"
        )
        tone = "" if s["correct"] is None else ("good" if s["correct"] else "bad")
        kpis = [_kpi(html.escape(s["y_pred"]), "predicted", tone)]
        if s["y_true"] is not None:
            kpis.append(_kpi(html.escape(s["y_true"]), "actual"))
        kpis += [
            _kpi(f"{s['confidence']:.3f}", "confidence"),
            _kpi(f"{s['dg']:.1f}", "DG"),
            _kpi(f"{s['redg']:.3f}", "REDG"),
        ]

    stress_section = (
        '<h2>Disagreement over the negotiation</h2>'
        '<div class="panel"><div id="stress"></div>'
        '<p class="sub" style="margin:8px 0 0;font-size:.8rem">'
        "Each row is a band, each column one negotiation step. Darker means more "
        "disagreement with its neighbours; solid blocks mark steps where the band "
        "was silenced by the conflict gate.</p></div>"
        if sample is not None
        else ""
    )

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(heading)}</title>
<style>{_CSS}</style></head><body><div class="wrap">
<h1>{html.escape(heading)}</h1>
<p class="sub">{html.escape(sub)}</p>
<div class="kpis">{''.join(kpis)}</div>

<h2>Signal and band partition</h2>
<div class="panel"><div id="signal"></div>
<p class="sub" style="margin:8px 0 0;font-size:.8rem">
Shaded regions are the bands. Red marks the two most implicated in conflict.
Click any band to inspect it.</p></div>

{stress_section}

<h2>Band conflict table</h2>
<div class="panel"><table id="bandtable"></table>
<p class="sub" style="margin:10px 0 0;font-size:.8rem">
High stress means a band disagreed with its neighbours, but that happens both
when a band is <em>wrong</em> and when it is merely <em>uninformative</em>.
Band confidence separates the two; informed conflict combines them. Ranked by
<strong>{html.escape(payload['rank_basis'].replace('_', ' '))}</strong>.</p></div>

<h2>Selected band, raw signal</h2>
<div class="panel detail">
  <div><div id="bandname" style="font-weight:600;margin-bottom:6px"></div>
       <div id="bandtrace"></div></div>
  <dl id="banddl"></dl>
</div>

<footer>Generated by pypasi · process-aware signal inference. Self-contained:
no external assets, no network access.</footer>
</div>
<script>window.__PYPASI__ = {json.dumps(payload)};</script>
<script>{_JS}</script>
</body></html>"""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
    return str(p)
