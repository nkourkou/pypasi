"""Chemometric classifiers, sklearn conformance, band importance and figures."""

import warnings

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils.estimator_checks import check_estimator

from pypasi import BandNegotiationClassifier
from pypasi.audit import band_importance
from pypasi.chemometrics import PCALDA, PLSDA, vip_scores
from pypasi.datasets import make_conflict_signals
from pypasi.divergence import softmax
from pypasi.logits import to_logits

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


@pytest.fixture(scope="module")
def data():
    X, y, axis, spec = make_conflict_signals(
        n_samples=260, noise=0.85, signal_strength=0.45, random_state=0
    )
    return X, y, axis, spec


# ----------------------------------------------------------------- PLS-DA


def test_plsda_fits_and_separates(data):
    X, y, _, _ = data
    m = PLSDA(n_components=8).fit(X, y)
    assert m.score(X, y) > 0.7
    assert m.classes_.tolist() == sorted(np.unique(y).tolist())


def test_plsda_components_are_clamped_to_what_data_supports():
    X = np.random.default_rng(0).normal(size=(6, 4))
    y = np.array([0, 0, 1, 1, 2, 2])
    m = PLSDA(n_components=50).fit(X, y)
    assert m.n_components_ <= min(X.shape[0] - 1, X.shape[1])


def test_vip_scores_shape_and_scale(data):
    X, y, _, _ = data
    m = PLSDA(n_components=8).fit(X, y)
    assert m.vip_scores_.shape == (X.shape[1],)
    assert np.all(m.vip_scores_ >= 0)
    # VIP is normalised so the mean of squares is 1; some features exceed 1
    assert m.vip_scores_.max() > 1.0
    np.testing.assert_allclose(np.mean(m.vip_scores_**2), 1.0, rtol=0.05)


def test_vip_scores_identify_the_informative_region(data):
    """The generator plants peaks at known positions; VIP should light up there."""
    X, y, axis, spec = data
    m = PLSDA(n_components=8).fit(X, y)
    vip = m.vip_scores_
    informative = np.zeros(axis.size, dtype=bool)
    for c in spec.informative + spec.conflicting:
        informative |= np.abs(axis - c) < 80
    assert vip[informative].mean() > vip[~informative].mean()


def test_vip_helper_matches_the_estimator_attribute(data):
    X, y, _, _ = data
    m = PLSDA(n_components=6).fit(X, y)
    Y = (y[:, None] == m.classes_[None, :]).astype(float)
    raw = PLSRegression(n_components=6, scale=False).fit(m.scaler_.transform(X), Y)
    np.testing.assert_allclose(vip_scores(raw), m.vip_scores_, rtol=1e-8)


def test_plsda_binary_decision_function_is_one_dimensional():
    """scikit-learn's convention, and what to_logits expects."""
    X, y, _, _ = make_conflict_signals(n_samples=80, n_classes=2, random_state=0)
    m = PLSDA(n_components=5).fit(X, y)
    d = m.decision_function(X)
    assert d.ndim == 1
    L = to_logits(m, X)
    assert L.shape == (X.shape[0], 2)
    np.testing.assert_allclose(softmax(L)[:, 1] > 0.5, d > 0)


def test_plsda_predict_proba_is_a_distribution(data):
    X, y, _, _ = data
    p = PLSDA(n_components=6).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)


def test_plsda_transform_gives_latent_scores(data):
    X, y, _, _ = data
    m = PLSDA(n_components=4).fit(X, y)
    assert m.transform(X).shape == (X.shape[0], 4)


# ----------------------------------------------------------------- PCA-LDA


def test_pcalda_fits_wide_short_data():
    """More features than samples - the case LDA cannot handle unaided."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 30)
    X = rng.normal(size=(30, 500)) + y[:, None] * 0.6
    m = PCALDA(n_components=10).fit(X, y)
    assert m.score(X, y) > 0.8
    assert m.feature_importances_.shape == (500,)


def test_pcalda_shrinkage_option(data):
    X, y, _, _ = data
    m = PCALDA(n_components=10, shrinkage="auto").fit(X, y)
    assert m.lda_.solver == "lsqr"
    assert m.predict(X).shape == y.shape


# -------------------------------------------------- use as band estimators


@pytest.mark.parametrize("base", [PLSDA(n_components=4), PCALDA(n_components=4)])
def test_chemometric_models_drive_the_bands(data, base):
    X, y, axis, _ = data
    clf = BandNegotiationClassifier(
        axis=axis, n_bands=5, base_estimator=base, max_iter=8, random_state=0
    ).fit(X, y)
    assert clf.predict(X).shape == y.shape
    assert clf.audit(X, y).conflict_table().shape[0] == 5


# ------------------------------------------------------ sklearn conformance


def test_deterministic_estimator_is_fully_sklearn_conformant():
    """At zero temperature nothing is stochastic, so every check must pass."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = check_estimator(
            BandNegotiationClassifier(n_bands=3, max_iter=4, temperature=0.0,
                                      random_state=0),
            on_fail=None,
        )
    failed = [r["check_name"] for r in res if r["status"] not in ("passed", "skipped")]
    assert not failed, f"failing checks: {failed}"


def test_stochastic_estimator_only_deviates_on_invariance_checks():
    """Metropolis acceptance draws per sample, so subsetting can change a call.

    This is inherent to the method rather than a defect, and it is the entire
    difference between the stochastic and deterministic configurations.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = check_estimator(
            BandNegotiationClassifier(n_bands=3, max_iter=4, temperature=0.05,
                                      random_state=0),
            on_fail=None,
        )
    failed = {r["check_name"] for r in res if r["status"] not in ("passed", "skipped")}
    assert failed <= {"check_methods_subset_invariance",
                      "check_methods_sample_order_invariance"}, failed


@pytest.mark.parametrize("est", [PLSDA(n_components=2), PCALDA(n_components=2)])
def test_chemometric_estimators_are_sklearn_conformant(est):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = check_estimator(est, on_fail=None)
    failed = [r["check_name"] for r in res if r["status"] not in ("passed", "skipped")]
    assert not failed, f"failing checks: {failed}"


@pytest.mark.parametrize("est", [PLSDA(n_components=3), PCALDA(n_components=3)])
def test_chemometric_estimators_cross_validate(data, est):
    X, y, _, _ = data
    assert cross_val_score(est, X, y, cv=3).mean() > 0.4


# ------------------------------------------------------------ band importance


def test_band_importance_table(data):
    X, y, axis, _ = data
    clf = BandNegotiationClassifier(axis=axis, n_bands=7, max_iter=6,
                                    axis_name="cm-1", random_state=0).fit(X, y)
    tbl, per_feature = clf.band_importance(X, y)
    assert len(tbl) == 7
    assert per_feature.shape == (X.shape[1],)
    assert {"importance_mean", "importance_max", "importance_sum"} <= set(tbl.columns)
    assert per_feature.max() == pytest.approx(1.0)


def test_band_importance_finds_the_informative_bands(data):
    X, y, axis, spec = data
    clf = BandNegotiationClassifier(axis=axis, n_bands=7, max_iter=6,
                                    random_state=0).fit(X, y)
    tbl, _ = clf.band_importance(X, y)
    informative = {b.label for b in clf.bands_
                   if any(b.lo <= c < b.hi for c in spec.informative)}
    top = set(tbl.nlargest(4, "importance_mean")["label"])
    assert informative & top, f"informative bands {informative} missed by {top}"


def test_band_importance_accepts_any_importance_bearing_model(data):
    from sklearn.ensemble import RandomForestClassifier

    X, y, axis, _ = data
    clf = BandNegotiationClassifier(axis=axis, n_bands=5, max_iter=4,
                                    random_state=0).fit(X, y)
    tbl, _ = clf.band_importance(
        X, y, estimator=RandomForestClassifier(n_estimators=12, random_state=0)
    )
    assert len(tbl) == 5


def test_band_importance_rejects_models_without_importance(data):
    from sklearn.neighbors import KNeighborsClassifier

    X, y, axis, _ = data
    clf = BandNegotiationClassifier(axis=axis, n_bands=4, max_iter=4,
                                    random_state=0).fit(X, y)
    with pytest.raises(TypeError, match="neither feature_importances_"):
        clf.band_importance(X, y, estimator=KNeighborsClassifier())


# ----------------------------------------------------------------- figures


@pytest.fixture(scope="module")
def fitted_audit(data):
    X, y, axis, _ = data
    clf = BandNegotiationClassifier(axis=axis, n_bands=7, axis_name="cm-1",
                                    max_iter=10, random_state=0).fit(X, y)
    return clf, clf.audit(X, y)


@pytest.mark.parametrize("dark", [False, True])
def test_every_figure_renders_in_both_modes(data, fitted_audit, dark, tmp_path):
    import pandas as pd

    from pypasi import viz

    X, y, _, _ = data
    clf, audit = fitted_audit
    imp, per_feature = clf.band_importance(X, y)
    comparison = pd.DataFrame(
        {"regime": ["plain", "H1", "H2"], "accuracy": [0.9, 0.91, 0.89],
         "DG_mean": [20.0, 26.0, 15.0]}
    )
    curve = pd.DataFrame(
        {"damage_fraction": [0.0, 0.2, 0.4] * 2, "DG_mean": [20, 26, 31, 15, 19, 23],
         "regime": ["plain"] * 3 + ["H1"] * 3}
    )
    figures = [
        viz.plot_bands(clf.bands_, X, y, dark=dark),
        viz.plot_band_conflict(audit, dark=dark),
        viz.plot_importance_vs_conflict(audit, imp, per_feature, dark=dark),
        viz.plot_stress_map(audit.result, clf.bands_, 0, dark=dark),
        viz.plot_regime_comparison(comparison, dark=dark),
        viz.plot_dg_curve(curve, dark=dark),
    ]
    for i, fig in enumerate(figures):
        out = tmp_path / f"f{i}_{int(dark)}.png"
        fig.savefig(out, dpi=60)
        assert out.stat().st_size > 1000
        matplotlib.pyplot.close(fig)


def test_palette_roles_are_defined_for_both_modes():
    from pypasi.viz import Palette

    for p in (Palette.light(), Palette.dark()):
        assert len(p.series) >= 3
        assert len(p.seq) >= 5
        for slot in (p.surface, p.ink, p.div_low, p.div_mid, p.div_high):
            assert slot.startswith("#") and len(slot) == 7


def test_regime_comparison_rejects_unknown_metrics():
    import pandas as pd

    from pypasi import viz

    with pytest.raises(ValueError, match="none of the requested metrics"):
        viz.plot_regime_comparison(pd.DataFrame({"regime": ["a"]}), metrics=("nope",))


# --------------------------------------------------------------- benchmark


def test_reference_implementation_matches_the_batched_engine():
    """The benchmark's slow reference must agree, or its timings mean nothing."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from bench_vectorisation import negotiate_looped

    from pypasi.gates import AbsoluteGate
    from pypasi.negotiate import negotiate
    from pypasi.topology import Topology

    L = np.random.default_rng(0).normal(size=(24, 7, 4)) * 2.0
    topo, gate = Topology.chain(7), AbsoluteGate(0.02)
    kw = dict(topology=topo, regime="H1", gate=gate, max_iter=12, temperature=0.0)
    batched = negotiate(L, record_trajectory=False, random_state=0, **kw)
    looped, weights = negotiate_looped(L, random_state=0, **kw)
    np.testing.assert_allclose(batched.logits_final, looped, atol=1e-9)
    np.testing.assert_allclose(batched.weights_final, weights, atol=1e-9)


def test_minority_evidence_is_internally_consistent():
    """A row flagged as dissenting must not show the same class on both sides."""
    import numpy as np
    from pypasi import BandNegotiationClassifier
    from pypasi.datasets import make_conflict_signals

    X, y, axis, *_ = make_conflict_signals(n_samples=400, random_state=0)
    clf = BandNegotiationClassifier(axis=axis, n_bands=7, regime="H1",
                                    random_state=0).fit(X, y)
    tbl = clf.audit(X, y).minority_evidence()
    dis = tbl[tbl["dissented"]]
    assert len(dis) > 0
    assert (dis["supported_class"] != dis["consensus_class"]).all()
    assert (tbl["n_samples_dissenting"] <= tbl["n_samples_muted"]).all()


def test_minority_evidence_needs_a_gate():
    from pypasi import BandNegotiationClassifier
    from pypasi.datasets import make_conflict_signals
    import pytest as _pytest

    X, y, axis, *_ = make_conflict_signals(n_samples=200, random_state=0)
    clf = BandNegotiationClassifier(axis=axis, n_bands=5, regime="plain",
                                    random_state=0).fit(X, y)
    audit = clf.audit(X, y)
    tbl = audit.minority_evidence()
    assert tbl.empty or not tbl["dissented"].any()
