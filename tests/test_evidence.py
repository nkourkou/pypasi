"""Tests for the typed evidence interface and temperature calibration."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from pypasi import (
    BandNegotiationClassifier,
    PCALDA,
    PLSDA,
    TemperatureCalibrator,
    UncalibratedEvidenceWarning,
    to_evidence,
)
from pypasi.datasets import make_conflict_signals
from pypasi.divergence import softmax
from pypasi.evidence import evidence_kind
from pypasi.logits import to_logits


@pytest.fixture(scope="module")
def cohort():
    X, y, axis, *_ = make_conflict_signals(n_samples=300, random_state=0)
    return X, y, axis


# --------------------------------------------------------------- classification


@pytest.mark.parametrize(
    "factory, expected",
    [
        (lambda: LogisticRegression(max_iter=500), "exact_logit"),
        (lambda: LinearDiscriminantAnalysis(), "exact_logit"),
        (lambda: PCALDA(n_components=4), "exact_logit"),
        (lambda: LinearSVC(C=0.1, max_iter=2000), "uncalibrated_score"),
        (lambda: PLSDA(n_components=4), "uncalibrated_score"),
        (lambda: RandomForestClassifier(n_estimators=10, random_state=0), "log_proba"),
    ],
)
def test_evidence_kind_is_identified(cohort, factory, expected):
    X, y, _ = cohort
    est = factory().fit(X, y)
    kind, _ = evidence_kind(est)
    assert kind == expected


def test_sgd_kind_depends_on_the_loss(cohort):
    X, y, _ = cohort
    assert evidence_kind(SGDClassifier(loss="log_loss").fit(X, y))[0] == "exact_logit"
    assert evidence_kind(SGDClassifier(loss="hinge").fit(X, y))[0] == "uncalibrated_score"


def test_pipeline_is_unwrapped_to_its_terminal_step(cohort):
    X, y, _ = cohort
    pipe = make_pipeline(StandardScaler(), LinearSVC(C=0.1, max_iter=2000)).fit(X, y)
    kind, source = evidence_kind(pipe)
    assert (kind, source) == ("uncalibrated_score", "decision_function")


def test_exact_logit_really_reproduces_predict_proba(cohort):
    """The claim the 'exact_logit' label makes, checked rather than asserted."""
    X, y, _ = cohort
    est = LogisticRegression(max_iter=500).fit(X, y)
    L, spec = to_evidence(est, X, len(est.classes_))
    assert spec.kind == "exact_logit"
    np.testing.assert_allclose(softmax(L), est.predict_proba(X), atol=1e-8)


def test_binary_exact_logit_reproduces_predict_proba():
    X, y, _, *_ = make_conflict_signals(n_samples=200, n_classes=2, random_state=1)
    est = LogisticRegression(max_iter=500).fit(X, y)
    L, spec = to_evidence(est, X, 2)
    assert spec.kind == "exact_logit"
    np.testing.assert_allclose(softmax(L), est.predict_proba(X), atol=1e-8)


def test_to_evidence_agrees_with_to_logits_up_to_centring(cohort):
    """0.5.0 must not silently change the numbers 0.4.x produced."""
    X, y, _ = cohort
    est = PLSDA(n_components=4).fit(X, y)
    old = to_logits(est, X, len(est.classes_))
    new, _ = to_evidence(est, X, len(est.classes_))
    old_c = old - old.mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(new, old_c, atol=1e-10)


def test_spec_records_the_scale(cohort):
    X, y, _ = cohort
    pls, _ = to_evidence(PLSDA(n_components=4).fit(X, y), X)
    lda, _ = to_evidence(PCALDA(n_components=4).fit(X, y), X)
    # the confound this module exists for: the same information on scales that
    # differ by an order of magnitude
    assert np.std(lda) > 3 * np.std(pls)


def test_uncalibrated_evidence_warns_when_asked(cohort):
    X, y, _ = cohort
    _, spec = to_evidence(PLSDA(n_components=4).fit(X, y), X)
    with pytest.warns(UncalibratedEvidenceWarning, match="not a log-odds"):
        spec.warn_if_uncalibrated()


def test_probabilistic_evidence_does_not_warn(cohort):
    X, y, _ = cohort
    _, spec = to_evidence(LogisticRegression(max_iter=500).fit(X, y), X)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec.warn_if_uncalibrated()


def test_missing_evidence_is_a_clear_error():
    class Useless:
        def predict(self, X):
            return np.zeros(len(X))

    with pytest.raises(TypeError, match="neither decision_function nor predict_proba"):
        evidence_kind(Useless())


def test_wrong_class_count_is_caught(cohort):
    X, y, _ = cohort
    est = LogisticRegression(max_iter=500).fit(X, y)
    with pytest.raises(ValueError, match="class scores but"):
        to_evidence(est, X, n_classes=len(est.classes_) + 1)


# ---------------------------------------------------------------- calibration


def _band_logits(X, y, factory, n_bands=5):
    clf = BandNegotiationClassifier(n_bands=n_bands, base_estimator=factory(),
                                    calibration=None, random_state=0).fit(X, y)
    return clf.band_logits(X), np.searchsorted(clf.classes_, y)


def test_temperature_reduces_nll_for_a_miscalibrated_scale(cohort):
    X, y, _ = cohort
    L, idx = _band_logits(X, y, lambda: PLSDA(n_components=4))
    cal = TemperatureCalibrator().fit(L, idx)
    assert cal.nll_after_ < cal.nll_before_
    assert cal.improvement > 0


def test_an_exact_logit_is_not_the_same_as_a_calibrated_one(cohort):
    """The distinction 'auto' relies on, and the one it deliberately ignores.

    ``softmax(decision_function) == predict_proba`` holds exactly for logistic
    regression, so the evidence is on a genuine log-odds scale. That says
    nothing about whether the scale is *right*: L2 shrinks the coefficients, so
    the log-odds come out too small and a temperature well above one still
    helps. ``calibration='always'`` exists for exactly this.
    """
    X, y, _ = cohort
    L, idx = _band_logits(X, y, lambda: LogisticRegression(max_iter=500))
    cal = TemperatureCalibrator().fit(L, idx)
    assert cal.improvement > 0.0
    assert cal.temperature_ > 1.0


def test_always_calibrates_even_an_exact_logit(cohort):
    X, y, axis = cohort
    auto = BandNegotiationClassifier(axis=axis, n_bands=5, calibration="auto",
                                     random_state=0).fit(X, y)
    always = BandNegotiationClassifier(axis=axis, n_bands=5, calibration="always",
                                       random_state=0).fit(X, y)
    assert auto.calibrator_ is None
    assert always.calibrator_ is not None


def test_temperature_is_order_preserving(cohort):
    X, y, _ = cohort
    L, idx = _band_logits(X, y, lambda: PLSDA(n_components=4))
    cal = TemperatureCalibrator().fit(L, idx)
    np.testing.assert_array_equal(cal.transform(L).argmax(-1), L.argmax(-1))


def test_per_band_mode_gives_one_temperature_per_band(cohort):
    X, y, _ = cohort
    L, idx = _band_logits(X, y, lambda: PLSDA(n_components=4))
    cal = TemperatureCalibrator(mode="per_band").fit(L, idx)
    assert np.shape(cal.temperature_) == (L.shape[1],)
    assert cal.transform(L).shape == L.shape


def test_per_band_mode_rejects_a_single_band_slab(cohort):
    X, y, _ = cohort
    L, idx = _band_logits(X, y, lambda: PLSDA(n_components=4))
    cal = TemperatureCalibrator(mode="per_band").fit(L, idx)
    with pytest.raises(ValueError, match="per-band calibration needs"):
        cal.transform(L[:, 0, :])


def test_calibrator_rejects_labels_that_are_not_indices(cohort):
    X, y, _ = cohort
    L, _ = _band_logits(X, y, lambda: PLSDA(n_components=4))
    with pytest.raises(ValueError, match="class indices"):
        TemperatureCalibrator().fit(L, np.full(len(X), 99))


def test_unfitted_calibrator_refuses_to_transform():
    with pytest.raises(RuntimeError, match="not fitted"):
        TemperatureCalibrator().transform(np.zeros((2, 3, 2)))


def test_bad_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        TemperatureCalibrator(mode="isotonic")


# ------------------------------------------------------------- estimator wiring


def test_auto_calibration_is_a_no_op_for_exact_logits(cohort):
    X, y, axis = cohort
    clf = BandNegotiationClassifier(axis=axis, n_bands=5, calibration="auto",
                                    random_state=0).fit(X, y)
    assert clf.calibrator_ is None
    assert clf.evidence_report.iloc[0]["kind"] == "exact_logit"


def test_auto_calibration_fires_for_a_score_scale(cohort):
    X, y, axis = cohort
    clf = BandNegotiationClassifier(axis=axis, n_bands=5,
                                    base_estimator=PLSDA(n_components=4),
                                    calibration="auto", random_state=0).fit(X, y)
    assert clf.calibrator_ is not None
    assert clf.evidence_report.iloc[0]["kind"] == "calibrated_log_proba"


def test_calibration_none_reproduces_0_4_behaviour(cohort):
    """The migration guarantee: opting out must give the old numbers back."""
    X, y, axis = cohort
    kw = dict(axis=axis, n_bands=5, base_estimator=PLSDA(n_components=4),
              random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        off = BandNegotiationClassifier(calibration=None, **kw).fit(X, y)
    L = off.band_logits(X)
    raw = np.stack([to_logits(est, X[:, b.features], off.classes_.size)
                    for est, b in zip(off.band_estimators_, off.bands_)], axis=1)
    np.testing.assert_allclose(L, raw - raw.mean(-1, keepdims=True), atol=1e-10)


def test_calibration_none_warns_about_the_scale(cohort):
    X, y, axis = cohort
    with pytest.warns(UncalibratedEvidenceWarning):
        BandNegotiationClassifier(axis=axis, n_bands=5,
                                  base_estimator=PLSDA(n_components=4),
                                  calibration=None, random_state=0).fit(X, y)


def test_explicit_calibration_data_is_used(cohort):
    X, y, axis = cohort
    clf = BandNegotiationClassifier(
        axis=axis, n_bands=5, base_estimator=PLSDA(n_components=4),
        calibration="temperature", calibration_data=(X[:120], y[:120]),
        random_state=0).fit(X[120:], y[120:])
    assert clf.calibrator_ is not None
    # nothing was held back from fitting, because calibration data was supplied
    assert clf.band_estimators_[0].pls_.x_scores_.shape[0] == len(X) - 120


def test_too_few_samples_to_split_warns_and_skips_calibration():
    X, y, axis, *_ = make_conflict_signals(n_samples=8, random_state=0)
    with pytest.warns(UserWarning, match="too few to hold out"):
        clf = BandNegotiationClassifier(axis=axis, n_bands=3,
                                        base_estimator=PLSDA(n_components=2),
                                        calibration="temperature",
                                        random_state=0).fit(X, y)
    assert clf.calibrator_ is None


def test_bad_calibration_option_is_rejected(cohort):
    X, y, axis = cohort
    with pytest.raises(ValueError, match="unknown calibration"):
        BandNegotiationClassifier(axis=axis, n_bands=3,
                                  calibration="platt", random_state=0).fit(X, y)


def test_bad_calibration_split_is_rejected(cohort):
    X, y, axis = cohort
    with pytest.raises(ValueError, match="calibration_split"):
        BandNegotiationClassifier(axis=axis, n_bands=3,
                                  base_estimator=PLSDA(n_components=4),
                                  calibration="temperature",
                                  calibration_split=1.5, random_state=0).fit(X, y)


def test_calibrated_model_still_classifies(cohort):
    X, y, axis = cohort
    clf = BandNegotiationClassifier(axis=axis, n_bands=5,
                                    base_estimator=PLSDA(n_components=4),
                                    random_state=0).fit(X, y)
    assert clf.predict(X).shape == (len(X),)
    assert clf.predict_proba(X).shape == (len(X), clf.classes_.size)


def test_evidence_report_is_one_row(cohort):
    X, y, axis = cohort
    clf = BandNegotiationClassifier(axis=axis, n_bands=5,
                                    base_estimator=PLSDA(n_components=4),
                                    random_state=0).fit(X, y)
    rep = clf.evidence_report
    assert len(rep) == 1
    assert rep.iloc[0]["is_probabilistic"]
    assert rep.iloc[0]["temperature"] != 1.0
