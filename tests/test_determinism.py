"""Fitting the same data twice must give the same model.

The regression guarded here is not hypothetical. ``NoveltyDetector`` and
``PCALDA`` built their ``PCA`` without a solver or a seed, and scikit-learn's
``svd_solver="auto"`` selects the *randomized* solver at spectroscopic shapes,
which draws from the global NumPy stream when nothing seeds it. Two fits of one
cohort therefore returned different Q residual and Hotelling thresholds
depending only on how much other code had run first: measured on a 2000 x 1000
cohort, the T-squared limit moved by 4%.

A control chart whose limits depend on execution order cannot be filed, so the
solver now defaults to the exact decomposition. Each test below drains a
different amount of the global stream between the two fits; before the fix that
alone was enough to change the answer.
"""

import numpy as np
import pytest
from sklearn.base import clone

from pypasi import BandNegotiationClassifier
from pypasi.chemometrics import PCALDA
from pypasi.datasets import make_conflict_signals
from pypasi.triage import NoveltyDetector, Triage


# scikit-learn's "auto" solver only switches to the randomized path once the
# cohort is bigger than 500 in its larger dimension. A smaller fixture would
# take the exact path anyway and every assertion below would pass against the
# unpatched code, which is the failure mode this file exists to prevent.
N_SAMPLES, N_POINTS = 600, 400


@pytest.fixture(scope="module")
def cohort():
    X, y, axis, _ = make_conflict_signals(n_samples=N_SAMPLES, n_classes=3,
                                          n_points=N_POINTS, random_state=0)
    return X, y, axis


def test_the_fixture_actually_exercises_the_randomized_solver(cohort):
    """Without this the rest of the file can pass while guarding nothing.

    Asserted against scikit-learn itself rather than against a remembered rule,
    so a future release that moves the threshold is caught here rather than
    silently disarming the tests below.
    """
    from sklearn.decomposition import PCA

    X, _, _ = cohort
    assert max(X.shape) > 500

    def components(n):
        _disturb(n)
        return PCA(n_components=10, svd_solver="auto").fit(X).components_

    assert not np.array_equal(components(0), components(5000)), (
        "sklearn's auto solver is exact on this cohort, so the determinism "
        "tests below would pass even against the unpatched code - enlarge the "
        "fixture until this assertion holds again")


def _disturb(n):
    """Leave the global stream where an arbitrary earlier cell would have."""
    np.random.seed(4321)
    np.random.random(n)


def test_novelty_detector_is_independent_of_the_global_stream(cohort):
    X, _, _ = cohort
    _disturb(0)
    a = NoveltyDetector(n_components=10).fit(X)
    _disturb(5000)
    b = NoveltyDetector(n_components=10).fit(X)

    assert a.spe_threshold_ == b.spe_threshold_
    assert a.t2_threshold_ == b.t2_threshold_
    assert np.array_equal(a.pca_.components_, b.pca_.components_)
    sa, sb = a.score(X), b.score(X)
    for key in ("spe", "t2"):
        assert np.array_equal(np.asarray(sa[key]), np.asarray(sb[key]))


def test_pcalda_is_independent_of_the_global_stream(cohort):
    X, y, _ = cohort
    _disturb(0)
    a = PCALDA(n_components=8).fit(X, y)
    _disturb(5000)
    b = PCALDA(n_components=8).fit(X, y)

    assert np.array_equal(a.pca_.components_, b.pca_.components_)
    assert np.array_equal(a.decision_function(X), b.decision_function(X))
    assert np.array_equal(a.feature_importances_, b.feature_importances_)


def test_triage_novelty_column_is_reproducible(cohort):
    """The end-to-end path the study uses, not just the component in isolation."""
    X, y, axis = cohort
    clf = BandNegotiationClassifier(axis=axis, n_bands=5, random_state=0).fit(X, y)

    _disturb(0)
    ta = Triage(n_components=8).fit(clf, X, y)
    _disturb(5000)
    tb = Triage(n_components=8).fit(clf, X, y)

    assert ta.novelty_.spe_threshold_ == tb.novelty_.spe_threshold_
    na = ta.assess(X, y).table["novelty"].to_numpy()
    nb = tb.assess(X, y).table["novelty"].to_numpy()
    assert np.array_equal(na, nb)


@pytest.mark.parametrize("cls", [NoveltyDetector, PCALDA])
def test_default_solver_is_exact(cls):
    """Guard the default itself: 'auto' silently reintroduces the randomness."""
    assert cls().svd_solver == "full"


def test_randomized_solver_still_available_and_seedable(cohort):
    X, _, _ = cohort
    a = NoveltyDetector(n_components=10, svd_solver="randomized", random_state=0).fit(X)
    _disturb(5000)
    b = NoveltyDetector(n_components=10, svd_solver="randomized", random_state=0).fit(X)
    assert np.array_equal(a.pca_.components_, b.pca_.components_)


def test_pcalda_new_parameters_survive_the_estimator_contract():
    """get_params / clone must carry the new arguments, or Pipeline breaks them."""
    est = PCALDA(n_components=4, svd_solver="randomized", random_state=7)
    params = est.get_params()
    assert params["svd_solver"] == "randomized"
    assert params["random_state"] == 7
    assert clone(est).get_params() == params
