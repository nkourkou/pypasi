"""Estimator, audit, traceback, reporting and datasets."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pypasi import (
    AbsoluteGate,
    BandNegotiationClassifier,
    BandSet,
    QuantileGate,
    Topology,
    compare_regimes,
)
from pypasi.datasets import make_conflict_signals


@pytest.fixture(scope="module")
def data():
    X, y, axis, spec = make_conflict_signals(
        n_samples=300, noise=0.5, signal_strength=0.45, random_state=0
    )
    return X, y, axis, spec


@pytest.fixture(scope="module")
def fitted(data):
    X, y, axis, _ = data
    return BandNegotiationClassifier(
        axis=axis, n_bands=7, axis_name="cm-1", random_state=0
    ).fit(X, y)


# --------------------------------------------------------------- estimator


def test_fit_predict_roundtrip(data, fitted):
    X, y, _, _ = data
    pred = fitted.predict(X)
    assert pred.shape == y.shape
    assert set(np.unique(pred)).issubset(set(fitted.classes_))
    assert (pred == y).mean() > 0.6


def test_predict_proba_is_a_distribution(data, fitted):
    X, _, _, _ = data
    p = fitted.predict_proba(X)
    assert p.shape == (X.shape[0], fitted.classes_.size)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(p >= 0)


def test_get_params_and_clone_roundtrip(data):
    _, _, axis, _ = data
    clf = BandNegotiationClassifier(axis=axis, n_bands=5, regime="H2", alpha=0.4)
    twin = clone(clf)
    assert twin.get_params()["n_bands"] == 5
    assert twin.get_params()["regime"] == "H2"
    assert twin.get_params()["alpha"] == 0.4


def test_works_inside_a_sklearn_pipeline(data):
    X, y, axis, _ = data
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pasi", BandNegotiationClassifier(axis=axis, n_bands=5, random_state=0)),
    ])
    pipe.fit(X, y)
    assert pipe.predict(X).shape == y.shape


def test_works_with_cross_val_score(data):
    X, y, axis, _ = data
    clf = BandNegotiationClassifier(axis=axis, n_bands=5, max_iter=8, random_state=0)
    scores = cross_val_score(clf, X, y, cv=3)
    assert scores.shape == (3,)
    assert np.all(scores > 0.4)


@pytest.mark.parametrize("base", [
    RandomForestClassifier(n_estimators=12, random_state=0),   # predict_proba only
    GaussianNB(),                                             # predict_proba only
    SVC(kernel="linear", random_state=0),                     # decision_function only
])
def test_arbitrary_sklearn_base_estimators(data, base):
    X, y, axis, _ = data
    clf = BandNegotiationClassifier(
        axis=axis, n_bands=5, base_estimator=base, max_iter=8, random_state=0
    ).fit(X, y)
    assert clf.predict(X).shape == y.shape


def test_unsupported_base_estimator_is_rejected(data):
    X, y, axis, _ = data

    class Dumb:
        def get_params(self, deep=True):
            return {}

        def fit(self, X, y):
            return self

    with pytest.raises(TypeError, match="neither decision_function"):
        BandNegotiationClassifier(axis=axis, base_estimator=Dumb()).fit(X, y)


def test_quantile_gate_is_calibrated_during_fit(fitted):
    assert isinstance(fitted.gate_, QuantileGate)
    assert fitted.gate_.tau_ is not None
    assert fitted.train_stress_.shape[1] == fitted.bands_.n_bands


def test_explicit_bandset_is_honoured(data):
    X, y, axis, _ = data
    bs = BandSet.from_edges(axis, [400, 900, 1400, 1800], axis_name="cm-1")
    clf = BandNegotiationClassifier(bands=bs, max_iter=6, random_state=0).fit(X, y)
    assert clf.bands_.n_bands == 3
    assert clf.bands_.labels == ["B1", "B2", "B3"]


def test_feature_count_mismatch_is_caught(data, fitted):
    X, _, _, _ = data
    with pytest.raises(ValueError, match="is expecting 400 features"):
        fitted.predict(X[:, :10])


def test_custom_topology_flows_through(data):
    X, y, axis, _ = data
    topo = Topology.from_edges(7, [(0, 6)], include_chain=True, name="chain+longrange")
    clf = BandNegotiationClassifier(
        axis=axis, n_bands=7, topology=topo, max_iter=8, random_state=0
    ).fit(X, y)
    assert clf.topology_.name == "chain+longrange"
    assert clf.audit(X, y).result.topology.n_edges == 7


# ------------------------------------------------------------------- audit


def test_conflict_table_shape_and_columns(data, fitted):
    X, y, _, _ = data
    t = fitted.audit(X, y).conflict_table()
    assert len(t) == 7
    for col in ("label", "cm-1_lo", "cm-1_hi", "band_confidence", "mean_stress",
                "informed_conflict", "mute_frequency", "band_DG", "conflict_rank"):
        assert col in t.columns, col
    assert t["conflict_rank"].notna().all()


def test_conflict_rank_falls_back_without_errors(data, fitted):
    """With no misclassifications there is no excess stress to rank on."""
    X, y, _, _ = data
    perfect = fitted.audit(X, fitted.predict(X))
    t = perfect.conflict_table()
    assert t.attrs["conflict_rank_basis"] == "informed_conflict"
    assert t["conflict_rank"].notna().all()


def test_audit_without_labels_still_ranks(data, fitted):
    X, _, _, _ = data
    t = fitted.audit(X).conflict_table()
    assert "excess_stress" not in t.columns
    assert t["conflict_rank"].notna().all()


def test_summary_has_one_row_per_sample(data, fitted):
    X, y, _, _ = data
    s = fitted.audit(X, y).summary()
    assert len(s) == X.shape[0]
    for col in ("DG", "DG_early", "REDG", "eREDG", "worst_band", "correct"):
        assert col in s.columns
    assert np.isfinite(s.attrs["activity_threshold"])


def test_audit_locates_planted_conflict(data):
    """The generator plants confidently-wrong evidence; the audit must find it."""
    X, y, axis, spec = data
    Xtr, Xte, ytr, yte, _, cte = train_test_split(
        X, y, spec.conflicted, test_size=0.4, stratify=y, random_state=0
    )
    clf = BandNegotiationClassifier(axis=axis, n_bands=7, random_state=0).fit(Xtr, ytr)
    a = clf.audit(Xte, yte)
    planted = [b.index for b in clf.bands_ if b.lo <= spec.conflicting[0] < b.hi][0]
    stress = a.result.mean_stress
    lift_planted = stress[cte, planted].mean() / stress[~cte, planted].mean()
    others = [k for k in range(7) if k != planted]
    lift_others = stress[cte][:, others].mean() / stress[~cte][:, others].mean()
    assert lift_planted > 1.3, "contaminated signals should stress the planted band"
    assert lift_planted > lift_others * 1.2, "the lift should be localised, not global"


# --------------------------------------------------------------- traceback


def test_trace_band_returns_the_right_raw_columns(data, fitted):
    X, y, axis, _ = data
    a = fitted.audit(X, y)
    band = fitted.bands_["B3"]
    tr = a.trace_band("B3")
    assert tr.X.shape == (X.shape[0], band.n_features)
    np.testing.assert_allclose(tr.X, X[:, band.features])
    np.testing.assert_allclose(tr.axis, axis[band.features])
    assert tr.axis.min() >= band.lo and tr.axis.max() <= band.hi


def test_trace_band_carries_class_means_and_conflict_row(data, fitted):
    X, y, _, _ = data
    tr = fitted.audit(X, y).trace_band(0)
    assert tr.class_means.shape == (np.unique(y).size, tr.axis.size)
    assert tr.conflict["mean_stress"] > 0
    frame = tr.to_frame()
    assert {"axis", "mean", "sd"}.issubset(frame.columns)


def test_trace_band_accepts_index_or_label(data, fitted):
    X, y, _, _ = data
    a = fitted.audit(X, y)
    np.testing.assert_allclose(a.trace_band(2).X, a.trace_band("B3").X)


def test_trace_requires_signals(data, fitted):
    from pypasi.audit import AuditResult

    X, y, _, _ = data
    bare = AuditResult(fitted.negotiate(X), fitted.bands_, X=None, y_true=y)
    with pytest.raises(RuntimeError, match="needs the signals"):
        bare.trace_band(0)


def test_trace_top_conflict_is_ordered(data, fitted):
    X, y, _, _ = data
    a = fitted.audit(X, y)
    traces = a.trace_top_conflict(3)
    assert [t.band.label for t in traces] == a.top_conflict_bands(3)


# ------------------------------------------------------- regime comparison


def test_compare_regimes_covers_all_three(data, fitted):
    X, y, _, _ = data
    audits = fitted.compare_regimes(X, y)
    assert set(audits) == {"plain", "H1", "H2"}
    table = compare_regimes(audits)
    assert len(table) == 3
    for col in ("accuracy", "DG_mean", "REDG_mean", "mute_fraction", "top_conflict_band"):
        assert col in table.columns


def test_plain_never_silences_anything(data, fitted):
    X, y, _, _ = data
    audits = fitted.compare_regimes(X, y)
    assert audits["plain"].result.mute_frequency.sum() == 0
    assert audits["H1"].result.mute_frequency.sum() > 0


def test_h1_and_h2_are_not_the_same_regime(data, fitted):
    X, y, _, _ = data
    audits = fitted.compare_regimes(X, y)
    h1, h2 = audits["H1"].result, audits["H2"].result
    assert not np.allclose(h1.dg, h2.dg)
    assert abs(h1.dg.mean() - h2.dg.mean()) / h1.dg.mean() > 0.05


# ----------------------------------------------------------------- exports


def test_csv_export_writes_every_table(data, fitted, tmp_path):
    X, y, _, _ = data
    paths = fitted.audit(X, y).to_csv(str(tmp_path / "run"))
    assert set(paths) == {"summary", "conflict", "bands"}
    for p in paths.values():
        assert Path(p).stat().st_size > 0


@pytest.mark.parametrize("sample", [None, 0])
def test_report_is_self_contained_html(data, fitted, tmp_path, sample):
    X, y, _, _ = data
    out = fitted.audit(X, y).report(str(tmp_path / "r.html"), sample=sample)
    text = Path(out).read_text()
    assert text.startswith("<!doctype html>")
    assert "__PYPASI__" in text
    # the only absolute URL permitted is the SVG XML namespace
    import re
    urls = set(re.findall(r"https?://[^\s\"'<>)]+", text))
    assert urls <= {"http://www.w3.org/2000/svg"}
    assert "<script src" not in text and "<link " not in text


def test_report_payload_is_json_serialisable(data, fitted):
    from pypasi.report import report_payload

    X, y, _, _ = data
    payload = report_payload(fitted.audit(X, y), sample=3)
    json.dumps(payload)
    assert payload["sample"]["index"] == 3
    assert len(payload["bands"]) == 7


# ---------------------------------------------------------------- datasets


def test_synthetic_generator_shapes_and_mask():
    X, y, axis, spec = make_conflict_signals(n_samples=50, n_points=200, random_state=1)
    assert X.shape == (50, 200)
    assert y.shape == (50,)
    assert axis.shape == (200,)
    assert spec.conflicted.shape == (50,)
    assert 0 < spec.conflicted.mean() < 1
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-9)


def test_synthetic_generator_is_reproducible():
    a = make_conflict_signals(n_samples=20, random_state=5)[0]
    b = make_conflict_signals(n_samples=20, random_state=5)[0]
    np.testing.assert_array_equal(a, b)


def test_bacteria_loader_reports_a_missing_directory():
    from pypasi.datasets import load_bacteria

    with pytest.raises(NotADirectoryError):
        load_bacteria("/definitely/not/here")
