"""Matplotlib figures.

Six plots, each answering one question:

:func:`plot_bands`
    What does the signal look like, and where do the bands fall?
:func:`plot_band_conflict`
    Which bands carry conflict, and is it associated with being wrong?
:func:`plot_importance_vs_conflict`
    Where does the classifier find information, versus where does inference run
    into trouble? These are different questions and the pair is the point.
:func:`plot_stress_map`
    For one signal, which band disagreed at which iteration?
:func:`plot_regime_comparison`
    How do Plain, H1 and H2 differ in accuracy and in trajectory?
:func:`plot_dg_curve`
    How does reconciliation effort grow as evidence is corrupted?
:func:`plot_control_chart`
    Has a new batch left the limits, and in which wavenumber interval?
:func:`plot_control_trend`
    How has the drift score moved across a run of batches?

All take ``dark=True`` for a dark-surface variant and return the Matplotlib
figure, so callers can adjust or save as they like. Requires the ``viz`` extra.

The palette is colour-vision-deficiency validated: the three categorical hues
clear the all-pairs CVD and normal-vision separation floors in both modes, and
every multi-series figure also carries direct labels so identity never rests on
colour alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "plot_bands",
    "plot_band_conflict",
    "plot_importance_vs_conflict",
    "plot_stress_map",
    "plot_regime_comparison",
    "plot_dg_curve",
    "plot_triage_map",
    "plot_risk_coverage",
    "plot_control_chart",
    "plot_control_trend",
    "Palette",
]


@dataclass(frozen=True)
class Palette:
    """Colour roles for one surface mode."""

    surface: str
    ink: str
    secondary: str
    muted: str
    grid: str
    series: tuple[str, ...]
    seq: tuple[str, ...]
    div_low: str
    div_mid: str
    div_high: str

    @staticmethod
    def light() -> "Palette":
        return Palette(
            surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e", muted="#7c7b76",
            grid="#e7e6e2",
            series=("#2a78d6", "#eb6834", "#1baf7a"),
            seq=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
            div_low="#2a78d6", div_mid="#f0efec", div_high="#e34948",
        )

    @staticmethod
    def dark() -> "Palette":
        return Palette(
            surface="#1a1a19", ink="#ffffff", secondary="#c3c2b7", muted="#8f8e86",
            grid="#33332f",
            series=("#3987e5", "#d95926", "#199e70"),
            seq=("#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
            div_low="#3987e5", div_mid="#383835", div_high="#e66767",
        )

    @staticmethod
    def of(dark: bool) -> "Palette":
        return Palette.dark() if dark else Palette.light()


def _figure(dark: bool, figsize, nrows=1, ncols=1, **kw):
    import matplotlib.pyplot as plt

    p = Palette.of(dark)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=p.surface, **kw)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(p.surface)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(p.grid)
            ax.spines[side].set_linewidth(1.0)
        ax.tick_params(colors=p.secondary, labelsize=9, length=3, width=0.8)
        ax.grid(True, color=p.grid, linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)
    return fig, axes, p


def _title(ax, text, p, sub=None):
    """Title above, subtitle beneath it, both clear of the axes."""
    if text:
        ax.set_title(text, color=p.ink, fontsize=11.5, loc="left",
                     pad=22 if sub else 7)
    if sub:
        ax.text(0.0, 1.012, sub, transform=ax.transAxes, color=p.muted,
                fontsize=8.8, va="bottom", ha="left")


def _seq_cmap(p: Palette):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("pypasi_seq", list(p.seq))


def _div_cmap(p: Palette):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "pypasi_div", [p.div_low, p.div_mid, p.div_high]
    )


# ------------------------------------------------------------------ figures


def plot_bands(bands, X, y=None, *, dark: bool = False, ax=None, max_classes: int = 3,
               title: str = "Signal and band partition"):
    """Mean signal with the band partition shaded behind it.

    With ``y`` supplied, one mean trace per class, direct-labelled at the right
    edge so identity never depends on colour alone.
    """
    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=float)
    p = Palette.of(dark)
    if ax is None:
        fig, ax, p = _figure(dark, (9.5, 3.6))
    else:
        fig = ax.figure

    axis = bands.axis
    if y is None:
        traces = [("mean", X.mean(axis=0))]
    else:
        y = np.asarray(y)
        classes = np.unique(y)[:max_classes]
        traces = [(str(c), X[y == c].mean(axis=0)) for c in classes]

    lo, hi = bands.lo, bands.hi
    ymin = min(t.min() for _, t in traces)
    ymax = max(t.max() for _, t in traces)
    pad = 0.10 * (ymax - ymin or 1.0)
    # Band labels sit along the bottom, leaving the top clear for the legend.
    for i, b in enumerate(bands):
        ax.axvspan(b.lo, b.hi, color=p.grid, alpha=0.55 if i % 2 == 0 else 0.22, lw=0)
        ax.text(0.5 * (b.lo + b.hi), ymin - pad * 0.72, b.label, ha="center",
                va="center", color=p.muted, fontsize=8.5)

    # Room at the right edge so direct labels are never clipped.
    gutter = 0.085 * (hi - lo)
    for i, (name, tr) in enumerate(traces):
        ax.plot(axis, tr, color=p.series[i % len(p.series)], linewidth=2.0,
                solid_capstyle="round", zorder=3, label=name)

    if len(traces) > 1:
        # Spectra converge at the edges, so end labels collide. Nudge them
        # apart, keeping their vertical order, before drawing.
        ends = np.array([tr[-1] for _, tr in traces], dtype=float)
        gap = 0.055 * (ymax - ymin or 1.0)
        order = np.argsort(ends)
        placed = ends.astype(float).copy()
        for j in range(1, len(order)):
            a, b = order[j - 1], order[j]
            if placed[b] - placed[a] < gap:
                placed[b] = placed[a] + gap
        for i, (name, _) in enumerate(traces):
            ax.annotate(name, (hi, placed[i]), xytext=(7, 0),
                        textcoords="offset points", color=p.secondary,
                        fontsize=9, va="center", annotation_clip=False)
    if len(traces) > 1:
        ax.legend(frameon=False, fontsize=9, loc="upper right",
                  labelcolor=p.secondary, ncol=len(traces), borderaxespad=0.2)

    ax.set_xlim(lo, hi + gutter)
    ax.set_ylim(ymin - pad * 1.25, ymax + pad * 1.5)
    ax.set_xlabel(bands.axis_name, color=p.secondary, fontsize=9.5)
    ax.set_ylabel("intensity", color=p.secondary, fontsize=9.5)
    ax.grid(axis="x", visible=False)
    _title(ax, title, p, f"{bands.n_bands} bands over {lo:g}–{hi:g} {bands.axis_name}")
    fig.tight_layout()
    return fig


def plot_band_conflict(audit, *, dark: bool = False, metric: str | None = None,
                       ax=None, title: str = "Where inference runs into conflict"):
    """Per-band conflict as a bar chart.

    With ground truth available the metric is ``excess_stress`` - conflict on
    wrong predictions minus conflict on right ones - which is genuinely signed,
    so it gets a diverging scale with a neutral zero. Without labels it falls
    back to ``informed_conflict``, a magnitude, drawn in a single hue.
    """
    tbl = audit.conflict_table()
    p = Palette.of(dark)
    if metric is None:
        metric = "excess_stress" if ("excess_stress" in tbl
                                     and tbl["excess_stress"].notna().any()) else "informed_conflict"
    vals = tbl[metric].to_numpy(dtype=float)
    labels = tbl["label"].tolist()
    diverging = metric == "excess_stress"

    if ax is None:
        fig, ax, p = _figure(dark, (8.2, 3.6))
    else:
        fig = ax.figure

    x = np.arange(len(labels))
    if diverging:
        span = np.nanmax(np.abs(vals)) or 1.0
        cmap = _div_cmap(p)
        colors = [cmap(0.5 + 0.5 * (v / span)) for v in vals]
    else:
        span = np.nanmax(vals) or 1.0
        cmap = _seq_cmap(p)
        colors = [cmap(0.25 + 0.7 * (v / span)) for v in vals]

    # a 2px surface gap between neighbouring bars keeps them reading as separate
    ax.bar(x, vals, width=0.78, color=colors, edgecolor=p.surface, linewidth=2.0,
           zorder=3)
    ax.axhline(0, color=p.secondary, linewidth=1.0, zorder=4)

    off = 0.03 * (np.nanmax(vals) - min(0.0, np.nanmin(vals)) or 1.0)
    for xi, v in zip(x, vals):
        if not np.isfinite(v):
            continue
        ax.text(xi, v + (off if v >= 0 else -off), f"{v:.3g}", ha="center",
                va="bottom" if v >= 0 else "top", color=p.secondary, fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{l}\n{lo:g}–{hi:g}" for l, lo, hi in
         zip(labels, tbl.iloc[:, 2], tbl.iloc[:, 3])],
        color=p.secondary, fontsize=8.5,
    )
    ax.set_ylabel(metric.replace("_", " "), color=p.secondary, fontsize=9.5)
    ax.grid(axis="x", visible=False)
    sub = ("cumulative stress on wrong predictions minus right ones"
           if diverging else "stress weighted by how confident the band was")
    _title(ax, title, p, sub)
    fig.tight_layout()
    return fig


def plot_importance_vs_conflict(audit, importance, per_feature=None, *,
                                dark: bool = False,
                                title: str = "Discriminative importance versus conflict"):
    """The two questions side by side, on a shared signal axis.

    Top: where a model fitted on the whole signal finds class information.
    Bottom: where the negotiation runs into conflict. A band high in both is
    informative but unstable - the first place to look when a prediction fails.

    Parameters
    ----------
    audit
        An :class:`~pypasi.audit.AuditResult`.
    importance
        The per-band table from :func:`~pypasi.audit.band_importance`.
    per_feature
        Optional per-feature importance from the same call, drawn as the
        continuous profile behind the band summary.
    """
    fig, axes, p = _figure(dark, (9.5, 5.6), nrows=2, sharex=True,
                           gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.18})
    top, bot = axes
    bands = audit.bands
    tbl = audit.conflict_table()

    # --- top: discriminative importance
    if per_feature is not None:
        top.plot(bands.axis, np.asarray(per_feature, dtype=float),
                 color=p.series[0], linewidth=1.4, alpha=0.55, zorder=2)
    for b, v in zip(bands, importance["importance_mean"]):
        top.add_patch(__import__("matplotlib").patches.Rectangle(
            (b.lo, 0), b.width, v, facecolor=p.series[0], alpha=0.30,
            edgecolor=p.surface, linewidth=2.0, zorder=3))
    top.set_ylabel("importance", color=p.secondary, fontsize=9.5)
    top.grid(axis="x", visible=False)
    _title(top, title, p, "where the classifier finds information (band mean, profile behind)")

    # --- bottom: conflict
    metric = ("excess_stress" if ("excess_stress" in tbl
              and tbl["excess_stress"].notna().any()) else "informed_conflict")
    vals = tbl[metric].to_numpy(dtype=float)
    span = np.nanmax(np.abs(vals)) or 1.0
    cmap = _div_cmap(p) if metric == "excess_stress" else _seq_cmap(p)
    for b, v in zip(bands, vals):
        c = cmap(0.5 + 0.5 * (v / span)) if metric == "excess_stress" else cmap(0.25 + 0.7 * (v / span))
        bot.add_patch(__import__("matplotlib").patches.Rectangle(
            (b.lo, min(0.0, v)), b.width, abs(v), facecolor=c,
            edgecolor=p.surface, linewidth=2.0, zorder=3))
    bot.axhline(0, color=p.secondary, linewidth=1.0, zorder=4)
    bot.set_ylim(min(0.0, np.nanmin(vals)) * 1.25 - 1e-9, np.nanmax(vals) * 1.25 + 1e-9)
    bot.set_ylabel(metric.replace("_", " "), color=p.secondary, fontsize=9.5)
    bot.set_xlabel(bands.axis_name, color=p.secondary, fontsize=9.5)
    bot.grid(axis="x", visible=False)
    _title(bot, "", p, "where inference runs into conflict")

    for ax in (top, bot):
        ax.set_xlim(bands.lo, bands.hi)
    top.set_ylim(0, top.get_ylim()[1] * 1.14)
    for b in bands:
        top.text(0.5 * (b.lo + b.hi), 0.955, b.label, transform=top.get_xaxis_transform(),
                 ha="center", va="top", color=p.muted, fontsize=8.5)
    # Rectangle patches make tight_layout unreliable here, so place the axes.
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.10, hspace=0.30)
    return fig


def plot_stress_map(result, bands, sample: int = 0, *, dark: bool = False,
                    title: str | None = None):
    """Band-by-iteration disagreement for one signal.

    Silenced steps are outlined rather than recoloured, so the map keeps a
    single sequential scale and muting reads as a separate channel.
    """
    import matplotlib.pyplot as plt

    if result.stress_history is None:
        raise RuntimeError("stress history was not recorded")
    S = result.stress_history[sample].T          # (bands, steps)
    M = None if result.mute_history is None else result.mute_history[sample].T

    fig, ax, p = _figure(dark, (9.0, 0.42 * S.shape[0] + 1.9))
    # sqrt keeps late, small disagreements visible without overstating them
    im = ax.imshow(np.sqrt(np.clip(S, 0, None)), aspect="auto", cmap=_seq_cmap(p),
                   interpolation="nearest", origin="upper")
    if M is not None:
        for k, t in zip(*np.nonzero(M)):
            ax.add_patch(plt.Rectangle((t - 0.5, k - 0.5), 1, 1, fill=False,
                                       edgecolor=p.div_high, linewidth=1.6, zorder=4))
    ax.set_yticks(range(S.shape[0]))
    ax.set_yticklabels(bands.labels, color=p.secondary, fontsize=9)
    ax.set_xlabel("negotiation step", color=p.secondary, fontsize=9.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03)
    cb.set_label("stress (sqrt scale)", color=p.secondary, fontsize=9)
    cb.ax.tick_params(colors=p.secondary, labelsize=8)
    cb.outline.set_visible(False)
    sub = "outlined cells were silenced by the conflict gate" if M is not None else None
    _title(ax, title or f"Disagreement over the negotiation · signal #{sample}", p, sub)
    fig.tight_layout()
    return fig


def plot_regime_comparison(table, *, dark: bool = False,
                           metrics=("accuracy", "DG_mean"),
                           title: str = "Regime comparison"):
    """One panel per metric - never two scales on one axis.

    Accuracy and DG have nothing in common dimensionally, so putting them on a
    shared axis would imply a relationship that is not there.
    """
    metrics = [m for m in metrics if m in table.columns]
    if not metrics:
        raise ValueError(f"none of the requested metrics are in the table: {list(table.columns)}")
    fig, axes, p = _figure(dark, (4.4 * len(metrics), 3.5), ncols=len(metrics))
    axes = np.atleast_1d(axes)

    regimes = table["regime"].astype(str).tolist()
    x = np.arange(len(regimes))
    for ax, metric in zip(axes, metrics):
        vals = table[metric].to_numpy(dtype=float)
        colors = [p.series[i % len(p.series)] for i in range(len(regimes))]
        ax.bar(x, vals, width=0.62, color=colors, edgecolor=p.surface,
               linewidth=2.0, zorder=3)
        top = np.nanmax(vals) or 1.0
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.03 * top, f"{v:.3g}", ha="center", va="bottom",
                    color=p.secondary, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(regimes, color=p.secondary, fontsize=9.5)
        ax.set_ylim(0, top * 1.18)
        ax.grid(axis="x", visible=False)
        _title(ax, metric.replace("_", " "), p)
    fig.suptitle(title, color=p.ink, fontsize=12, x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_dg_curve(frame, *, dark: bool = False, x="damage_fraction", value="DG_mean",
                  group="regime", title: str = "Reconciliation effort under corruption"):
    """One line per regime, direct-labelled at the right edge.

    Expects a long-format table with a corruption level, a value and a group.
    """
    fig, ax, p = _figure(dark, (7.6, 4.0))
    groups = list(dict.fromkeys(frame[group].astype(str)))
    for i, g in enumerate(groups):
        sub = frame[frame[group].astype(str) == g].sort_values(x)
        c = p.series[i % len(p.series)]
        ax.plot(sub[x], sub[value], color=c, linewidth=2.0, marker="o",
                markersize=6.5, markeredgecolor=p.surface, markeredgewidth=1.6,
                zorder=3, label=g)
        ax.annotate(g, (sub[x].iloc[-1], sub[value].iloc[-1]), xytext=(7, 0),
                    textcoords="offset points", color=p.secondary, fontsize=9.5,
                    va="center")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=p.secondary)
    ax.set_xlabel(x.replace("_", " "), color=p.secondary, fontsize=9.5)
    ax.set_ylabel(value.replace("_", " "), color=p.secondary, fontsize=9.5)
    ax.margins(x=0.12)
    _title(ax, title, p, "higher means the decision took more internal negotiation")
    fig.tight_layout()
    return fig


def plot_triage_map(report, *, dark: bool = False, log_novelty: bool = True,
                    title: str = "Triage map"):
    """Novelty against conflict, with the four actions as quadrants.

    The picture makes the central claim visible: a strange spectrum and a
    self-contradicting one are different failures, and they land in different
    quadrants. Where ground truth is available, errors are drawn as crosses so
    identity does not rest on colour.

    Parameters
    ----------
    report
        A :class:`~pypasi.triage.TriageReport` from
        :meth:`~pypasi.triage.Triage.assess`.
    """
    t = report.table
    thr_c = report.thresholds["conflict"]
    fig, ax, p = _figure(dark, (7.6, 5.4))
    good, bad = "#0ca30c", "#d03b3b"

    x = t["novelty"].to_numpy(dtype=float)
    yv = t["conflict"].to_numpy(dtype=float)
    if log_novelty:
        x = np.clip(x, 1e-3, None)
        ax.set_xscale("log")

    if "correct" in t:
        ok = t["correct"].to_numpy() == 1
        ax.scatter(x[ok], yv[ok], s=26, facecolor=good, edgecolor=p.surface,
                   linewidth=0.8, alpha=0.75, zorder=3, label="correct")
        ax.scatter(x[~ok], yv[~ok], s=52, marker="X", facecolor=bad,
                   edgecolor=p.surface, linewidth=1.0, zorder=4, label="incorrect")
        # Above the axes: the four corners are taken by the quadrant labels.
        ax.legend(frameon=False, fontsize=9, ncol=2, labelcolor=p.secondary,
                  loc="lower right", bbox_to_anchor=(1.0, 1.005), borderaxespad=0.0)
    else:
        ax.scatter(x, yv, s=26, facecolor=p.series[0], edgecolor=p.surface,
                   linewidth=0.8, alpha=0.75, zorder=3)

    ax.axvline(1.0, color=p.secondary, linewidth=1.2, linestyle="--", zorder=2)
    ax.axhline(thr_c, color=p.secondary, linewidth=1.2, linestyle="--", zorder=2)

    ax.margins(y=0.12)
    counts = report.action_counts
    # Quadrant labels in axes coordinates, inset from each corner.
    corners = {
        "accept":    (0.015, 0.02, "left", "bottom"),
        "review":    (0.015, 0.98, "left", "top"),
        "remeasure": (0.985, 0.02, "right", "bottom"),
        "reject":    (0.985, 0.98, "right", "top"),
    }
    for action, (fx, fy, ha, va) in corners.items():
        ax.text(fx, fy, f"{action}\nn = {int(counts.get(action, 0))}",
                transform=ax.transAxes, ha=ha, va=va, color=p.muted,
                fontsize=9, linespacing=1.3, zorder=5)

    ax.set_xlabel("novelty  (> 1 means unlike the training data)",
                  color=p.secondary, fontsize=9.5)
    ax.set_ylabel(f"conflict  ({report.thresholds['conflict_score']})",
                  color=p.secondary, fontsize=9.5)
    _title(ax, title, p,
           f"review rate {report.thresholds['review_rate']:.0%} "
           f"· {len(t)} signals")
    fig.tight_layout()
    return fig


def plot_risk_coverage(frame, *, dark: bool = False,
                       title: str = "Accuracy retained as cases are handed off"):
    """Risk-coverage curves for several flags.

    Ranks signals by each score and hands off the least trustworthy first. A
    useful flag lifts accuracy as coverage falls; one that carries no
    information leaves the curve flat. The model's own confidence is the
    baseline worth beating.
    """
    fig, ax, p = _figure(dark, (7.8, 4.4))
    names = list(dict.fromkeys(frame["score"].astype(str)))
    ends = []
    for i, name in enumerate(names):
        sub = frame[frame["score"].astype(str) == name].sort_values("coverage")
        c = p.series[i % len(p.series)]
        ax.plot(sub["coverage"], sub["accuracy"], color=c, linewidth=2.0,
                marker="o", markersize=6.5, markeredgecolor=p.surface,
                markeredgewidth=1.6, zorder=3, label=name)
        ends.append((name, float(sub.iloc[0]["coverage"]), float(sub.iloc[0]["accuracy"])))
    # Stagger the left-edge labels so close curves stay readable.
    lo = min(e[2] for e in ends)
    hi = max(e[2] for e in ends)
    gap = 0.055 * ((hi - lo) or 1.0)
    placed = {}
    for name, cx, cy in sorted(ends, key=lambda e: e[2]):
        prev = max(placed.values(), default=-np.inf)
        placed[name] = max(cy, prev + gap) if placed else cy
    for name, cx, _ in ends:
        ax.annotate(name, (cx, placed[name]), xytext=(-9, 0),
                    textcoords="offset points", ha="right", va="center",
                    color=p.secondary, fontsize=9, annotation_clip=False)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=p.secondary)
    ax.set_xlabel("coverage (share of signals kept)", color=p.secondary, fontsize=9.5)
    ax.set_ylabel("accuracy on the kept signals", color=p.secondary, fontsize=9.5)
    ax.margins(x=0.18)
    _title(ax, title, p, "steeper to the left means the flag is finding real errors")
    fig.tight_layout()
    return fig


def plot_control_chart(report, *, dark: bool = False, statistic: str | None = None,
                       title: str = "Per-band control chart"):
    """One batch against its limits, band by band.

    The vertical axis is standardised, so every statistic shares one scale and
    the shaded strip is the in-control region whatever is being charted. Bands
    are laid out in wavenumber order with their intervals under the labels,
    because the useful output of this chart is an interval to take to the
    instrument, not a score.
    """
    import numpy as _np

    table = report.table
    stats = [statistic] if statistic else list(dict.fromkeys(table["statistic"]))
    table = table[table["statistic"].isin(stats)]
    labels = list(dict.fromkeys(table["label"]))
    axis_name = report.axis_name
    lo = {r["label"]: r[f"{axis_name}_lo"] for _, r in table.iterrows()}
    hi = {r["label"]: r[f"{axis_name}_hi"] for _, r in table.iterrows()}

    fig, ax, p = _figure(dark, (max(7.2, 1.15 * len(labels) + 3.4), 4.2))
    x = _np.arange(len(labels))
    ax.axhspan(-report.k, report.k, color=p.grid, alpha=0.55, zorder=0)
    ax.axhline(0.0, color=p.muted, lw=1.0, zorder=1)
    for s in (report.k, -report.k):
        ax.axhline(s, color=p.div_high, lw=1.2, ls="--", zorder=1)

    offsets = _np.linspace(-0.22, 0.22, len(stats)) if len(stats) > 1 else [0.0]
    for si, (stat, off) in enumerate(zip(stats, offsets)):
        sub = table[table["statistic"] == stat].set_index("label").reindex(labels)
        z = sub["z"].to_numpy(dtype=float)
        out = sub["out_of_control"].to_numpy(dtype=bool)
        colour = p.series[si % len(p.series)]
        ax.vlines(x + off, 0, z, color=colour, lw=1.4, alpha=0.55, zorder=2)
        ax.scatter(x + off, z, s=[110 if o else 64 for o in out],
                   marker="X" if si else "o",
                   color=[p.div_high if o else colour for o in out],
                   edgecolor=p.surface, linewidth=1.6, zorder=3,
                   label=stat.replace("_", " "))
        for xi, (zi, oi) in enumerate(zip(z, out)):
            if oi and _np.isfinite(zi):
                ax.annotate(f"{zi:+.0f}", (xi + off, zi), textcoords="offset points",
                            xytext=(0, 9 if zi > 0 else -15), ha="center",
                            fontsize=8.5, color=p.secondary)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n{lo[lab]:.0f}-{hi[lab]:.0f}" for lab in labels],
                       color=p.secondary, fontsize=8.5)
    ax.set_xlabel(f"band  ({axis_name})", color=p.secondary)
    ax.set_ylabel(f"standardised deviation from the reference", color=p.secondary)
    # Bound by the data, not symmetrically: drift is usually one-signed and a
    # mirrored axis would waste half the chart. Both limit lines stay visible.
    z_all = table["z"].to_numpy(dtype=float)
    z_all = z_all[_np.isfinite(z_all)]
    zlo = min(-report.k * 1.35, float(z_all.min()) * 1.2 if z_all.size else 0.0)
    zhi = max(report.k * 1.35, float(z_all.max()) * 1.2 if z_all.size else 0.0)
    pad = 0.08 * (zhi - zlo)
    ax.set_ylim(zlo - pad, zhi + pad)
    ax.grid(axis="x", visible=False)
    state = "in control" if report.is_in_control else (
        f"{report.n_out_of_control} outside the limits: "
        + ", ".join(report.bands_out_of_control()))
    _title(ax, f"{title} - {report.name}", p,
           sub=f"{report.n_samples} spectra, no labels used  |  {state}")
    if len(stats) > 1:
        ax.legend(frameon=False, fontsize=8.8, labelcolor=p.secondary, ncol=len(stats),
                  loc="upper left")
    fig.tight_layout()
    return fig


def plot_control_trend(frame, *, dark: bool = False, k: float | None = None,
                       title: str = "Drift across batches"):
    """The drift score of each batch in order, with the control threshold.

    Takes the frame returned by :meth:`pypasi.ControlProfile.check_many`. This
    is the view that turns a one-off check into monitoring: a single batch
    outside the limits is a signal, several in a row is a change.
    """
    import numpy as _np

    k = float(k if k is not None else frame.attrs.get("k", 3.0))
    fig, ax, p = _figure(dark, (max(7.0, 0.7 * len(frame) + 3.0), 3.9))
    x = _np.arange(len(frame))
    y = frame["drift_score"].to_numpy(dtype=float)
    out = ~frame["in_control"].to_numpy(dtype=bool)

    ax.axhspan(0, k, color=p.grid, alpha=0.55, zorder=0)
    ax.axhline(k, color=p.div_high, lw=1.2, ls="--", zorder=1)
    ax.plot(x, y, color=p.series[0], lw=1.8, zorder=2)
    ax.scatter(x, y, s=[105 if o else 60 for o in out],
               color=[p.div_high if o else p.series[0] for o in out],
               edgecolor=p.surface, linewidth=1.6, zorder=3)
    ax.text(len(frame) - 0.5, k, f" {k:g} sigma", color=p.div_high, fontsize=8.5,
            va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["batch"].astype(str), color=p.secondary, fontsize=8.8,
                       rotation=25, ha="right")
    ax.set_ylabel("drift score  (largest |z| in the batch)", color=p.secondary)
    ax.set_ylim(0, max(k * 1.35, float(_np.nanmax(y)) * 1.18 if len(y) else k))
    ax.grid(axis="x", visible=False)
    n_out = int(out.sum())
    _title(ax, title, p,
           sub=f"{len(frame)} batches, {n_out} outside the limits  |  "
               f"statistic: {frame.attrs.get('statistic', 'mean_stress').replace('_', ' ')}")
    fig.tight_layout()
    return fig
