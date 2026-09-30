"""Quality-control triage: novelty, conflict, and keeping them apart."""

import numpy as np
import pytest

from pypasi import BandNegotiationClassifier
from pypasi.datasets import make_conflict_signals
from pypasi.triage import (
    ACTIONS,
    NoveltyDetector,
    Triage,
    compare_conflict_scores,
    risk_coverage,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


@pytest.fixture(scope="module")
def cohorts():
    """A clean cohort to calibrate on and a contaminated one to assess."""
    Xtr, ytr, axis, _ = make_conflict_signals(
        n_samples=600, noise=0.85, signal_strength=0.45, random_state=0
    )
    Xte, yte, _, spec = make_conflict_signals(
        n_samples=400, noise=0.85, signal_strength=0.45,
        outlier_rate=0.15, random_state=1
    )
    return Xtr, ytr, Xte, yte, axis, spec


@pytest.fixture(scope="module")
def fitted(cohorts):
    Xtr, ytr, _, _, axis, _ = cohorts
    return BandNegotiationClassifier(
        axis=axis, n_bands=7, axis_name="cm-1", random_state=0
    ).fit(Xtr, ytr)


# ------------------------------------------------------------------ novelty


def test_novelty_detector_finds_planted_outliers(cohorts):
    from sklearn.metrics import roc_auc_score

    Xtr, _, Xte, _, _, spec = cohorts
    nd = NoveltyDetector(n_components=10, q=0.99).fit(Xtr)
    s = nd.score(Xte)
    assert {"spe", "t2", "novelty"} <= set(s.columns)
    assert roc_auc_score(spec.outlier, s["novelty"]) > 0.9
    flagged = s["novelty"].to_numpy() > 1.0
    assert flagged[spec.outlier].mean() > 0.8


def test_novelty_threshold_matches_its_quantile(cohorts):
    Xtr = cohorts[0]
    nd = NoveltyDetector(n_components=8, q=0.95).fit(Xtr)
    on_train = nd.score(Xtr)["novelty"].to_numpy() > 1.0
    # roughly 1 - q of the calibration data sits outside, by construction
    assert 0.01 < on_train.mean() < 0.15


def test_novelty_rejects_impossible_quantile(cohorts):
    with pytest.raises(ValueError, match="q must lie"):
        NoveltyDetector(q=1.5).fit(cohorts[0])


# ------------------------------------------------------------------- triage


def test_triage_must_be_calibrated_first(cohorts):
    Xte = cohorts[2]
    with pytest.raises(RuntimeError, match="must be calibrated first"):
        Triage().assess(Xte)


def test_assess_produces_one_row_per_signal_and_valid_actions(cohorts, fitted):
    Xtr, ytr, Xte, yte, _, _ = cohorts
    tri = Triage(review_rate=0.12).fit(fitted, Xtr, ytr)
    rep = tri.assess(Xte, yte)
    assert len(rep.table) == Xte.shape[0]
    assert set(rep.table["action"]) <= set(ACTIONS)
    assert set(rep.action_counts.index) == set(ACTIONS)
    assert rep.action_counts.sum() == Xte.shape[0]
    for col in ("novelty", "conflict", "is_novel", "is_conflicted", "reason", "worst_band"):
        assert col in rep.table.columns


def test_review_rate_is_honoured_on_the_calibration_cohort(cohorts, fitted):
    """The referral budget is the knob a clinic actually has; it must hold."""
    Xtr, ytr, _, _, _, _ = cohorts
    tri = Triage(review_rate=0.2).fit(fitted, Xtr, ytr)
    flagged = (tri.calibration_conflict_ > tri.conflict_threshold_).mean()
    assert 0.15 < flagged < 0.25


def test_actions_follow_the_two_flags(cohorts, fitted):
    Xtr, ytr, Xte, yte, _, _ = cohorts
    t = Triage(review_rate=0.15).fit(fitted, Xtr, ytr).assess(Xte, yte).table
    novel, conf = t["is_novel"].to_numpy(), t["is_conflicted"].to_numpy()
    assert (t["action"][novel & conf] == "reject").all()
    assert (t["action"][novel & ~conf] == "remeasure").all()
    assert (t["action"][~novel & conf] == "review").all()
    assert (t["action"][~novel & ~conf] == "accept").all()


def test_novelty_flag_tracks_planted_outliers_not_planted_conflict(cohorts, fitted):
    """The whole point: the novelty flag must answer the input-space question."""
    Xtr, ytr, Xte, yte, _, spec = cohorts
    t = Triage(review_rate=0.12).fit(fitted, Xtr, ytr).assess(Xte, yte).table
    novel = t["is_novel"].to_numpy()
    assert novel[spec.outlier].mean() > 0.8, "should catch the strange spectra"
    # and it should not be driven by conflict, which lives elsewhere entirely
    clean = ~spec.outlier
    assert abs(novel[clean & spec.conflicted].mean()
               - novel[clean & ~spec.conflicted].mean()) < 0.15


# ------------------------------------------------------------- orientation


def test_conflict_orientation_is_learned_not_assumed(cohorts, fitted):
    Xtr, ytr, _, _, _, _ = cohorts
    with_labels = Triage().fit(fitted, Xtr, ytr)
    assert with_labels.conflict_sign_ in (-1, 1)
    assert "learned" in with_labels.orientation_source_

    without = Triage().fit(fitted, Xtr)
    assert without.conflict_sign_ == 1
    assert "assumed" in without.orientation_source_


def test_inactive_trajectories_stay_calm_under_either_orientation(cohorts, fitted):
    """A trajectory with nothing to reconcile is not conflicted, either way round.

    Mapping inactive samples to a fixed sentinel before the sign is known put
    them at the *most* conflicted end whenever the sign flipped, which silenced
    the review queue entirely.
    """
    from pypasi.triage import _orient

    raw = np.array([0.0, 0.2, 0.5, 0.9, 0.0])
    active = np.array([False, True, True, True, False])
    for sign in (1, -1):
        oriented = _orient(raw, active, sign)
        assert oriented[0] == oriented[4] == oriented[active].min()


def test_review_queue_is_non_empty_when_the_sign_flips(cohorts, fitted):
    Xtr, ytr, Xte, yte, _, _ = cohorts
    for kind in ("eredg", "dg", "stress", "mute"):
        tri = Triage(conflict_score=kind, review_rate=0.15).fit(fitted, Xtr, ytr)
        rep = tri.assess(Xte, yte)
        assert rep.table["is_conflicted"].sum() > 0, f"{kind} flagged nothing"


def test_unknown_conflict_score_is_rejected(cohorts, fitted):
    Xtr, ytr, _, _, _, _ = cohorts
    with pytest.raises(ValueError, match="unknown conflict score"):
        Triage(conflict_score="vibes").fit(fitted, Xtr, ytr)


def test_impossible_review_rate_is_rejected(cohorts, fitted):
    Xtr, ytr, _, _, _, _ = cohorts
    with pytest.raises(ValueError, match="review_rate must lie"):
        Triage(review_rate=1.5).fit(fitted, Xtr, ytr)


# --------------------------------------------------------------- evidence


def test_evaluate_returns_the_full_evidence_set(cohorts, fitted):
    Xtr, ytr, Xte, yte, _, _ = cohorts
    ev = Triage(review_rate=0.12).fit(fitted, Xtr, ytr).evaluate(Xte, yte)
    for key in ("separation", "conditional", "among_confident", "orthogonality",
                "incremental", "risk_coverage", "summary"):
        assert key in ev, key
    assert len(ev["separation"]) == 3
    assert "1 - confidence" in ev["separation"]["score"].tolist()


def test_novelty_and_conflict_are_nearly_orthogonal(cohorts, fitted):
    """The claim that makes this more than outlier detection."""
    Xtr, ytr, Xte, yte, _, _ = cohorts
    ev = Triage(review_rate=0.12).fit(fitted, Xtr, ytr).evaluate(Xte, yte)
    o = ev["orthogonality"]
    assert abs(o["spearman_rho"]) < 0.3
    assert o["jaccard_overlap"] < 0.3
    assert o["n_novel_only"] > 0 and o["n_conflicted_only"] > 0
    assert ev["summary"]["novelty_and_conflict_uncorrelated"]


def test_evaluation_reports_the_confidence_baseline_honestly(cohorts, fitted):
    """A flag that cannot beat the softmax must be reported as such."""
    Xtr, ytr, Xte, yte, _, _ = cohorts
    ev = Triage(review_rate=0.12).fit(fitted, Xtr, ytr).evaluate(Xte, yte)
    sep = ev["separation"].set_index("score")
    assert np.isfinite(sep.loc["1 - confidence", "auroc_error"])
    assert isinstance(ev["summary"]["conflict_beats_confidence"], bool)
    assert isinstance(ev["summary"]["conflict_adds_to_confidence"], bool)
    assert ev["incremental"]["note"] == "weights fitted on calibration data"


def test_compare_conflict_scores_covers_every_descriptor(cohorts, fitted):
    Xtr, ytr, Xte, yte, _, _ = cohorts
    tbl = compare_conflict_scores(fitted, Xtr, ytr, Xte, yte, review_rate=0.12)
    assert set(tbl["conflict_score"]) == {"eredg", "dg", "stress", "mute"}
    assert tbl["sign"].isin([-1, 1]).all()
    assert tbl["worth_using"].dtype == bool


# --------------------------------------------------------- risk / coverage


def test_risk_coverage_shape_and_bookkeeping():
    rng = np.random.default_rng(0)
    wrong = rng.integers(0, 2, 200)
    score = wrong + rng.normal(0, 0.3, 200)      # a good flag
    rc = risk_coverage(score, wrong)
    assert rc["coverage"].is_monotonic_decreasing
    assert (rc["n_kept"] <= 200).all()
    assert np.isfinite(rc.attrs["aurc"])
    assert rc.attrs["total_errors"] == int(wrong.sum())
    assert rc["accuracy"].iloc[0] < rc["accuracy"].iloc[-1], "a good flag lifts accuracy"


def test_risk_coverage_is_flat_for_an_uninformative_flag():
    rng = np.random.default_rng(1)
    wrong = rng.integers(0, 2, 400)
    rc = risk_coverage(rng.normal(size=400), wrong)
    assert abs(rc["accuracy"].iloc[0] - rc["accuracy"].iloc[-1]) < 0.12


def test_risk_coverage_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        risk_coverage(np.zeros(5), np.zeros(4))


# ----------------------------------------------------------------- figures


@pytest.mark.parametrize("dark", [False, True])
def test_triage_figures_render(cohorts, fitted, dark, tmp_path):
    from pypasi import viz

    Xtr, ytr, Xte, yte, _, _ = cohorts
    tri = Triage(review_rate=0.12).fit(fitted, Xtr, ytr)
    rep = tri.assess(Xte, yte)
    ev = tri.evaluate(Xte, yte)
    for i, fig in enumerate([viz.plot_triage_map(rep, dark=dark),
                             viz.plot_risk_coverage(ev["risk_coverage"], dark=dark)]):
        out = tmp_path / f"t{i}_{int(dark)}.png"
        fig.savefig(out, dpi=60)
        assert out.stat().st_size > 1000
        matplotlib.pyplot.close(fig)


def test_triage_map_works_without_ground_truth(cohorts, fitted, tmp_path):
    from pypasi import viz

    Xtr, ytr, Xte, _, _, _ = cohorts
    rep = Triage().fit(fitted, Xtr, ytr).assess(Xte)
    assert "correct" not in rep.table.columns
    fig = viz.plot_triage_map(rep)
    fig.savefig(tmp_path / "nolabels.png", dpi=60)
    matplotlib.pyplot.close(fig)


def test_evaluate_says_why_it_cannot_score_a_perfect_cohort():
    """'Encountered all NA values' is not a usable error message."""
    import numpy as np
    import pytest as _pytest
    from pypasi import BandNegotiationClassifier, Triage
    from pypasi.datasets import make_conflict_signals

    X, y, axis, *_ = make_conflict_signals(n_samples=200, random_state=0)
    clf = BandNegotiationClassifier(axis=axis, n_bands=5, random_state=0).fit(X, y)
    tri = Triage().fit(clf, X, y)
    perfect = clf.predict(X)            # by construction the model is never wrong
    with _pytest.raises(ValueError, match="only one class of outcome"):
        tri.evaluate(X, perfect)


# ------------------------------------------- orientation (0.5.1)


def _orient_fixture():
    import numpy as np
    from pypasi import BandNegotiationClassifier
    from pypasi.datasets import make_conflict_signals

    X, y, axis, *_ = make_conflict_signals(n_samples=400, random_state=0)
    clf = BandNegotiationClassifier(axis=axis, n_bands=5, random_state=0).fit(X, y)
    rng = np.random.default_rng(0)
    degraded = np.zeros(len(X), int)
    bad = rng.choice(len(X), len(X) // 2, replace=False)
    degraded[bad] = 1
    return clf, X, y, degraded


def test_orient_on_uses_the_supplied_target():
    """A degradation detector must be orientable on degradation."""
    from pypasi import Triage

    clf, X, y, degraded = _orient_fixture()
    t = Triage(conflict_score="dg").fit(clf, X, y, orient_on=degraded)
    assert "the supplied target" in t.orientation_source_


def test_default_orientation_is_still_the_error_target():
    import numpy as np
    import warnings as _w
    from pypasi import Triage

    clf, X, y, _ = _orient_fixture()
    # the fixture classifier is perfect, so give it labels it gets wrong -
    # otherwise there is only one outcome and no orientation to learn
    noisy = np.asarray(y).copy()
    flip = np.random.default_rng(2).choice(len(noisy), len(noisy) // 5, replace=False)
    noisy[flip] = (noisy[flip] + 1) % len(np.unique(y))
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        t = Triage(conflict_score="dg").fit(clf, X, noisy)
    assert "wrong prediction" in t.orientation_source_


def test_orient_on_length_is_validated():
    import pytest as _pytest
    from pypasi import Triage

    clf, X, y, degraded = _orient_fixture()
    with _pytest.raises(ValueError, match="orient_on has"):
        Triage().fit(clf, X, y, orient_on=degraded[:10])


def test_a_near_chance_orientation_warns():
    """The failure this exists for: a coin-flip sign silently inverts a descriptor."""
    import numpy as np
    import pytest as _pytest
    from pypasi import Triage, WeakOrientationWarning

    clf, X, y, _ = _orient_fixture()
    coin = np.random.default_rng(1).integers(0, 2, len(X))
    with _pytest.warns(WeakOrientationWarning, match="close to arbitrary"):
        Triage().fit(clf, X, y, orient_on=coin)


def test_orientation_margin_is_recorded():
    from pypasi import Triage

    clf, X, y, degraded = _orient_fixture()
    t = Triage(conflict_score="dg").fit(clf, X, y, orient_on=degraded)
    assert t.orientation_margin_ is not None
    assert 0.0 <= t.orientation_margin_ <= 0.5


def test_flipping_the_target_flips_the_sign_and_the_auroc():
    """Orientation is the whole difference between p and 1-p."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from pypasi import Triage

    clf, X, y, degraded = _orient_fixture()
    a = Triage(conflict_score="dg").fit(clf, X, y, orient_on=degraded)
    b = Triage(conflict_score="dg").fit(clf, X, y, orient_on=1 - degraded)
    assert a.conflict_sign_ == -b.conflict_sign_
    ta = a.assess(X, y).table["conflict"].to_numpy()
    tb = b.assess(X, y).table["conflict"].to_numpy()
    assert roc_auc_score(degraded, ta) == pytest.approx(
        1.0 - roc_auc_score(degraded, tb), abs=1e-9)
