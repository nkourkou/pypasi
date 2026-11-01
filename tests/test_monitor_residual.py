"""Per-band PCA residual statistics in ControlProfile (0.5.4)."""

import numpy as np
import pytest

from pypasi import (
    RESIDUAL_STATISTICS,
    BandNegotiationClassifier,
    ControlProfile,
    NoveltyDetector,
)
from pypasi.datasets import make_conflict_signals


@pytest.fixture(scope="module")
def setup():
    X, y, axis, _ = make_conflict_signals(
        n_samples=2400, noise=0.85, signal_strength=0.45, random_state=0
    )
    Xfit, yfit, Xref, Xnew = X[:1200], y[:1200], X[1200:2000], X[2000:]
    clf = BandNegotiationClassifier(
        axis=axis, n_bands=7, axis_name="cm-1", random_state=0
    ).fit(Xfit, yfit)
    nd = NoveltyDetector(n_components=5).fit(Xfit)
    refs = np.array_split(Xref, 20)
    return clf, nd, refs, Xnew, axis


def test_residual_statistics_are_monitored():
    from pypasi.monitor import MONITORED_STATISTICS

    assert set(RESIDUAL_STATISTICS) <= set(MONITORED_STATISTICS)


def test_residual_stats_need_a_residual_model(setup):
    clf, _nd, refs, _, _ = setup
    with pytest.raises(ValueError, match="residual_model"):
        ControlProfile.fit(
            clf, batches=refs, statistics=("residual_q",), random_state=0
        )
    with pytest.raises(ValueError, match="not fitted"):
        ControlProfile.fit(
            clf,
            batches=refs,
            statistics=("residual_q",),
            residual_model=NoveltyDetector(n_components=5),
            random_state=0,
        )


def test_residual_only_profile_flags_a_local_change_and_names_its_band(setup):
    clf, nd, refs, Xnew, _ = setup
    prof = ControlProfile.fit(
        clf,
        batches=refs,
        statistics=RESIDUAL_STATISTICS,
        residual_model=nd,
        random_state=0,
    )
    clean = prof.check(Xnew[:40])
    b = clf.bands_[3]
    bad = Xnew[:40].copy()
    bad[:, b.features] += 1.5 * np.sin(
        np.linspace(0, np.pi, b.n_features)
    )  # a real-looking local bump
    rep = prof.check(bad)
    assert rep.drift_score > clean.drift_score
    assert not rep.is_in_control
    assert rep.worst["label"] == b.label


def test_combined_profile_counts_all_tests(setup):
    clf, nd, refs, Xnew, _ = setup
    stats = ("mean_stress", "band_confidence") + RESIDUAL_STATISTICS
    prof = ControlProfile.fit(
        clf, batches=refs, statistics=stats, residual_model=nd, random_state=0
    )
    assert prof.n_tests == 4 * len(clf.bands_)
    rep = prof.check(Xnew[:40])
    assert set(rep.table["statistic"]) == set(stats)


def test_save_load_requires_and_checks_the_residual_model(setup, tmp_path):
    clf, nd, refs, Xnew, _ = setup
    prof = ControlProfile.fit(
        clf,
        batches=refs,
        statistics=RESIDUAL_STATISTICS,
        residual_model=nd,
        random_state=0,
    )
    path = prof.save(tmp_path / "qc.json")
    with pytest.raises(ValueError, match="residual_model"):
        ControlProfile.load(path, estimator=clf)
    other = NoveltyDetector(n_components=3).fit(refs[0].repeat(3, axis=0) + 0.0)
    with pytest.raises(ValueError, match="residual model"):
        ControlProfile.load(path, estimator=clf, residual_model=other)
    back = ControlProfile.load(path, estimator=clf, residual_model=nd)
    a, b = prof.check(Xnew[:40]).table, back.check(Xnew[:40]).table
    np.testing.assert_allclose(a["z"].to_numpy(), b["z"].to_numpy())


def test_unused_residual_model_warns(setup):
    clf, nd, refs, _, _ = setup
    with pytest.warns(UserWarning, match="no residual statistic"):
        ControlProfile.fit(clf, batches=refs, residual_model=nd, random_state=0)
