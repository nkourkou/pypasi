"""Command-line interface.

Three entry points::

    pypasi demo   --out results/demo
    pypasi run    --x X.npy --y y.npy --axis axis.npy --out results/mine
    pypasi bacteria --data-dir path/to/bacteria-ID --out results/bacteria

Each writes the band-conflict table, a per-sample summary, a regime comparison
and self-contained HTML reports into ``--out``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__


def _load(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: {p} does not exist")
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=False)
    if p.suffix in (".csv", ".txt", ".tsv"):
        delim = "\t" if p.suffix == ".tsv" else ","
        return np.loadtxt(p, delimiter=delim)
    raise SystemExit(f"error: cannot read {p.suffix} files; use .npy, .csv or .tsv")


def _run_workflow(args, X_train, y_train, X_eval, y_eval, axis, label: str) -> int:
    from .audit import compare_regimes
    from .estimator import BandNegotiationClassifier

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clf = BandNegotiationClassifier(
        axis=axis,
        n_bands=args.n_bands,
        band_method=args.band_method,
        band_range=tuple(args.band_range) if args.band_range else None,
        axis_name=args.axis_name,
        topology=args.topology,
        regime=args.regime,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    print(f"fitting {label}: {X_train.shape[0]} signals, {X_train.shape[1]} points, "
          f"{np.unique(y_train).size} classes")
    clf.fit(X_train, y_train)
    print(f"  bands      {', '.join(clf.bands_.labels)}")
    print(f"  topology   {clf.topology_}")
    print(f"  gate       {clf.gate_}")

    audits = clf.compare_regimes(X_eval, y_eval)
    table = compare_regimes(audits)
    table.to_csv(out / "regime_comparison.csv", index=False)
    print("\n" + table.round(4).to_string(index=False))

    primary = audits.get(str(args.regime), next(iter(audits.values())))
    paths = primary.to_csv(str(out / label))
    written = [out / "regime_comparison.csv", *(Path(p) for p in paths.values())]

    written.append(Path(primary.report(str(out / f"{label}_cohort.html"))))
    n = min(args.n_reports, primary.n_samples)
    if n:
        worst = np.argsort(primary.result.dg)[::-1][:n]
        for i in worst:
            written.append(
                Path(primary.report(str(out / f"{label}_signal{i}.html"), sample=int(i)))
            )

    print(f"\ntop conflict bands ({args.regime}): {', '.join(primary.top_conflict_bands(3))}")
    print(f"\nwrote {len(written)} files to {out}/")
    for p in written:
        print(f"  {p.name}")
    return 0


def _cmd_demo(args) -> int:
    from .datasets import make_conflict_signals
    from sklearn.model_selection import train_test_split

    X, y, axis, spec = make_conflict_signals(
        n_samples=args.n_samples, noise=args.noise, signal_strength=0.45,
        random_state=args.seed
    )
    print(f"synthetic signals with conflict planted at "
          f"{', '.join(f'{c:g}' for c in spec.conflicting)} {args.axis_name} "
          f"({spec.conflicted.mean():.0%} of signals)")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=args.seed
    )
    return _run_workflow(args, Xtr, ytr, Xte, yte, axis, "demo")


def _cmd_run(args) -> int:
    from sklearn.model_selection import train_test_split

    X, y = _load(args.x), _load(args.y).ravel()
    axis = _load(args.axis) if args.axis else np.arange(X.shape[1], dtype=float)
    if args.x_eval and args.y_eval:
        Xtr, ytr = X, y
        Xte, yte = _load(args.x_eval), _load(args.y_eval).ravel()
    else:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=args.seed
        )
    return _run_workflow(args, Xtr, ytr, Xte, yte, axis, args.name)


def _cmd_bacteria(args) -> int:
    from .datasets import load_bacteria

    data = load_bacteria(
        args.data_dir, subsample=args.subsample, random_state=args.seed
    )
    print(data)
    args.axis_name = "cm-1"
    if args.band_range is None:
        args.band_range = [400.0, min(1800.0, float(data.axis.max()))]
    return _run_workflow(
        args, data.X_dev, data.y_dev, data.X_test, data.y_test, data.axis, "bacteria"
    )


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", default="pypasi_results", help="output directory")
    p.add_argument("--n-bands", type=int, default=7, help="number of macro-bands")
    p.add_argument("--band-method", default="equal_width",
                   choices=["equal_width", "peak_informed"])
    p.add_argument("--band-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"), help="axis limits for band construction")
    p.add_argument("--axis-name", default="axis", help="axis units, e.g. cm-1")
    p.add_argument("--topology", default="chain",
                   choices=["chain", "complete", "knn"])
    p.add_argument("--regime", default="H1", choices=["plain", "H1", "H2"])
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--n-reports", type=int, default=3,
                   help="per-signal HTML reports to write, highest DG first")
    p.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pypasi",
        description="Process-aware band-resolved inference and conflict auditing.",
    )
    parser.add_argument("--version", action="version", version=f"pypasi {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run on synthetic signals with planted conflict")
    d.add_argument("--n-samples", type=int, default=600)
    d.add_argument("--noise", type=float, default=0.85,
                   help="signal noise; higher makes the task harder and produces\nthe errors eREDG needs in order to be scored")
    _add_common(d)
    d.set_defaults(func=_cmd_demo)

    r = sub.add_parser("run", help="run on your own signals")
    r.add_argument("--x", required=True, help="signals, .npy or .csv")
    r.add_argument("--y", required=True, help="labels, .npy or .csv")
    r.add_argument("--axis", default=None, help="signal axis, .npy or .csv")
    r.add_argument("--x-eval", default=None, help="optional held-out signals")
    r.add_argument("--y-eval", default=None, help="optional held-out labels")
    r.add_argument("--test-size", type=float, default=0.3)
    r.add_argument("--name", default="run", help="prefix for output files")
    _add_common(r)
    r.set_defaults(func=_cmd_run)

    b = sub.add_parser("bacteria", help="run the Ho et al. bacteria/yeast benchmark")
    b.add_argument("--data-dir", required=True, help="directory holding the .npy files")
    b.add_argument("--subsample", type=int, default=None,
                   help="cap signals per cohort, for quick iteration")
    _add_common(b)
    b.set_defaults(func=_cmd_bacteria)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, NotADirectoryError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
