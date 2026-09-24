"""How much does negotiating the whole cohort at once actually buy?

The samples in a negotiation are independent, so the obvious implementation is a
Python loop: one sample, thirty iterations, repeat. ``pypasi`` instead advances
every sample through the same iteration simultaneously, which turns the inner
loop into a handful of array operations whose cost barely moves as the cohort
grows.

This script measures the difference against a reference implementation that is
deliberately written the obvious way, and checks that the two agree to floating
point before reporting any timing - a speedup that changes the answer is not a
speedup.

    python benchmarks/bench_vectorisation.py --out bench.csv --figure bench.png
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from pypasi.divergence import js_divergence, softmax
from pypasi.energy import negotiation_energy
from pypasi.gates import AbsoluteGate
from pypasi.negotiate import negotiate
from pypasi.regimes import resolve_regime
from pypasi.topology import Topology

EPS = 1e-12


def negotiate_looped(band_logits, *, topology, regime, gate, lam=1.0,
                     temperature=0.01, max_iter=30, random_state=None):
    """Reference implementation: one sample at a time, the obvious way.

    Kept deliberately naive. It exists to be correct and to be slow, so the
    batched engine has something honest to be compared against.
    """
    L0 = np.asarray(band_logits, dtype=float)
    n, k, c = L0.shape
    A = topology.adjacency
    reg = resolve_regime(regime)
    rng = np.random.default_rng(random_state)
    out = np.empty_like(L0)
    weights = np.empty((n, k))

    for i in range(n):
        Li = L0[i : i + 1].copy()
        wi = np.clip(softmax(Li).max(axis=-1), 0.0, 1.0)
        e = negotiation_energy(Li, L0[i : i + 1], wi, topology, lam=lam)
        frozen = np.zeros((1, k), dtype=bool)
        for _ in range(max_iter):
            num = np.einsum("kj,nj,njc->nkc", A, wi, Li)
            den = np.einsum("kj,nj->nk", A, wi)
            cons = num / (den[..., None] + EPS)
            # A band whose neighbours have all been silenced has no consensus to
            # move toward, and holds its own position. Omitting this is what made
            # the first draft of this reference disagree with the engine.
            dead = den <= EPS
            if np.any(dead):
                cons = np.where(dead[..., None], Li, cons)
            stress = js_divergence(softmax(Li), softmax(cons))
            muted = gate.mask(stress) if gate is not None else np.zeros((1, k), bool)
            Lp, wp = reg.propose(Li, cons, wi, muted, frozen)
            ep = negotiation_energy(Lp, L0[i : i + 1], wp, topology, lam=lam)
            accept = bool(ep[0] <= e[0])
            if not accept and temperature > 0:
                accept = rng.random() < float(np.exp(-(ep[0] - e[0]) / temperature))
            if accept:
                Li, wi, e = Lp, wp, ep
        out[i] = Li[0]
        weights[i] = wi[0]
    return out, weights


def run(sizes, n_bands=7, n_classes=5, max_iter=30, repeats=3, seed=0):
    rng = np.random.default_rng(seed)
    topo = Topology.chain(n_bands)
    gate = AbsoluteGate(0.02)
    rows = []
    for n in sizes:
        L = rng.normal(size=(n, n_bands, n_classes)) * 2.0

        # correctness first: identical accept/reject stream, identical result
        batched = negotiate(L, topology=topo, regime="H1", gate=gate,
                            max_iter=max_iter, temperature=0.0,
                            record_trajectory=False, random_state=seed)
        looped, w_loop = negotiate_looped(L, topology=topo, regime="H1", gate=gate,
                                          max_iter=max_iter, temperature=0.0,
                                          random_state=seed)
        agree = np.allclose(batched.logits_final, looped, atol=1e-9)
        if not agree:
            raise SystemExit(
                f"the two implementations disagree at n={n}; refusing to report timings"
            )

        def _time(fn):
            best = np.inf
            for _ in range(repeats):
                t0 = time.perf_counter()
                fn()
                best = min(best, time.perf_counter() - t0)
            return best

        t_batched = _time(lambda: negotiate(
            L, topology=topo, regime="H1", gate=gate, max_iter=max_iter,
            temperature=0.0, record_trajectory=False, random_state=seed))
        t_looped = _time(lambda: negotiate_looped(
            L, topology=topo, regime="H1", gate=gate, max_iter=max_iter,
            temperature=0.0, random_state=seed))

        rows.append({
            "n_samples": n,
            "looped_s": t_looped,
            "batched_s": t_batched,
            "speedup": t_looped / t_batched,
            "batched_us_per_signal": 1e6 * t_batched / n,
            "looped_us_per_signal": 1e6 * t_looped / n,
            "identical": agree,
        })
        print(f"  n={n:6d}  looped {t_looped:7.3f}s  batched {t_batched:7.3f}s  "
              f"speedup {t_looped / t_batched:6.1f}x")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[100, 250, 500, 1000, 2500])
    ap.add_argument("--n-bands", type=int, default=7)
    ap.add_argument("--n-classes", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None, help="write results as CSV")
    ap.add_argument("--figure", default=None, help="write a figure")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    print(f"negotiation: {args.n_bands} bands, {args.n_classes} classes, "
          f"{args.max_iter} iterations, best of {args.repeats}")
    rows = run(args.sizes, args.n_bands, args.n_classes, args.max_iter,
               args.repeats, args.seed)

    import pandas as pd

    df = pd.DataFrame(rows)
    print("\n" + df.round(4).to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        from pypasi.viz import Palette, _figure, _title

        fig, axes, p = _figure(False, (9.0, 3.6), ncols=2)
        left, right = axes
        left.plot(df.n_samples, df.looped_s, marker="o", markersize=6.5,
                  linewidth=2.0, color=p.series[1], markeredgecolor=p.surface,
                  markeredgewidth=1.6, label="per-sample loop")
        left.plot(df.n_samples, df.batched_s, marker="o", markersize=6.5,
                  linewidth=2.0, color=p.series[0], markeredgecolor=p.surface,
                  markeredgewidth=1.6, label="batched")
        for name, col, colr in (("per-sample loop", "looped_s", p.series[1]),
                                ("batched", "batched_s", p.series[0])):
            left.annotate(name, (df.n_samples.iloc[-1], df[col].iloc[-1]),
                          xytext=(7, 0), textcoords="offset points",
                          color=p.secondary, fontsize=9, va="center",
                          annotation_clip=False)
        left.set_xscale("log"); left.set_yscale("log")
        left.set_xlabel("signals in the cohort", color=p.secondary, fontsize=9.5)
        left.set_ylabel("seconds", color=p.secondary, fontsize=9.5)
        left.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=p.secondary)
        left.margins(x=0.22)
        _title(left, "Wall clock", p, "log-log; lower is better")

        right.plot(df.n_samples, df.speedup, marker="o", markersize=6.5,
                   linewidth=2.0, color=p.series[0], markeredgecolor=p.surface,
                   markeredgewidth=1.6)
        for x, v in zip(df.n_samples, df.speedup):
            right.annotate(f"{v:.0f}x", (x, v), xytext=(0, 8),
                           textcoords="offset points", ha="center",
                           color=p.secondary, fontsize=9)
        right.set_xscale("log")
        right.set_xlabel("signals in the cohort", color=p.secondary, fontsize=9.5)
        right.set_ylabel("speedup", color=p.secondary, fontsize=9.5)
        right.margins(x=0.15, y=0.20)
        _title(right, "Speedup", p, "batched versus per-sample loop")
        fig.tight_layout()
        fig.savefig(args.figure, dpi=150, bbox_inches="tight")
        print(f"wrote {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
