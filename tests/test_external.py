"""Tests for auditing a classifier the library did not fit."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from pypasi import ControlProfile, ExternalAudit, audit_external
from pypasi.datasets import make_conflict_signals


@pytest.fixture(scope="module")
def cohort():
    X, y, axis, *_ = make_conflict_signals(n_samples=400, random_state=0)
    return X, y, axis


@pytest.fixture(scope="module")
def host(cohort):
    X, y, _ = cohort
    return LogisticRegression(max_iter=1000).fit(X, y)


@pytest.fixture(scope="module")
def ext(cohort, host):
    X, y, axis = cohort
    return audit_external(host, X, y, axis=axis, n_bands=5, random_state=0)


def test_the_host_prediction_is_never_overridden(ext, host, cohort):
    """The whole contract: the laboratory's answer of record is untouched."""
    X, _, _ = cohort
    np.testing.assert_array_equal(ext.predict(X), host.predict(X))
    np.testing.assert_array_equal(
        ext.explain(X)["host_prediction"].to_numpy(), host.predict(X))


def test_report_separates_the_two_predictions(ext, cohort):
    X, _, _ = cohort
    rep = ext.explain(X[:50])
    assert {"host_prediction", "band_prediction", "agreement"} <= set(rep.columns)
    agree = rep["host_prediction"].to_numpy() == rep["band_prediction"].to_numpy()
    np.testing.assert_array_equal(rep["agreement"].to_numpy(), agree)


def test_report_carries_an_interval_not_just_a_score(ext, cohort):
    X, _, _ = cohort
    rep = ext.explain(X[:20])
    lo = rep["worst_band_axis_lo"].to_numpy()
    hi = rep["worst_band_axis_hi"].to_numpy()
    assert np.all(hi > lo)
    assert set(rep["worst_band"]) <= set(ext.bands_.labels)


def test_report_says_what_the_negotiation_describes(ext, cohort):
    X, _, _ = cohort
    assert "auxiliary band ensemble" in ext.explain(X[:5]).attrs["note"]


def test_novelty_is_optional(cohort, host):
    X, y, axis = cohort
    bare = audit_external(host, X, y, axis=axis, n_bands=5, novelty=False,
                          random_state=0)
    assert "novelty" not in bare.explain(X[:5]).columns
    with pytest.raises(RuntimeError, match="no NoveltyDetector"):
        bare.novelty_ratio(X[:5])


def test_a_host_without_predict_is_rejected(ext):
    class NotAClassifier:
        pass

    with pytest.raises(TypeError, match="has no predict"):
        ExternalAudit(NotAClassifier(), ext.band_model)


def test_host_without_predict_proba_gives_nan_confidence(cohort):
    X, y, axis = cohort
    svm = LinearSVC(C=0.1, max_iter=5000).fit(X, y)
    ext = audit_external(svm, X, y, axis=axis, n_bands=5, random_state=0)
    assert np.all(np.isnan(ext.explain(X[:10])["host_confidence"].to_numpy()))


def test_validate_reports_against_the_host_errors(cohort, host):
    """Error prediction is host-specific and must be measured, not assumed."""
    X, y, axis = cohort
    noisy = np.asarray(y).copy()
    rng = np.random.default_rng(0)
    flip = rng.choice(len(noisy), size=len(noisy) // 4, replace=False)
    noisy[flip] = (noisy[flip] + 1) % len(np.unique(y))

    ext = audit_external(host, X, y, axis=axis, n_bands=5, random_state=0)
    out = ext.validate(X, noisy)
    assert "auroc_vs_host_error" in out.columns
    assert 0.0 <= out["host_accuracy"].iloc[0] <= 1.0
    assert "specific to this host" in out.attrs["note"]


def test_validate_refuses_an_auroc_with_one_class(cohort, host):
    X, y, axis = cohort
    ext = audit_external(host, X, y, axis=axis, n_bands=5, random_state=0)
    out = ext.validate(X, host.predict(X))      # host is right about everything
    assert out["auroc_vs_host_error"].isna().all()
    assert out["host_accuracy"].iloc[0] == pytest.approx(1.0)


def test_control_profile_builds_on_an_external_audit(ext, cohort):
    """The point of delegating bands_ and audit(): batch monitoring unchanged."""
    X, _, _ = cohort
    prof = ControlProfile.fit(ext, batches=[X[i::10] for i in range(10)])
    report = prof.check(X[:100])
    assert report.table["label"].nunique() == ext.bands_.n_bands


def test_the_auxiliary_model_is_a_second_opinion_not_a_decomposition(ext, cohort):
    """Its prediction comes from the band ensemble, not from the host."""
    X, _, _ = cohort
    rep = ext.explain(X[:100])
    band_from_model = ext.band_model.predict(X[:100])
    np.testing.assert_array_equal(rep["band_prediction"].to_numpy(), band_from_model)
