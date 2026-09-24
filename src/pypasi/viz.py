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
