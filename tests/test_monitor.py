"""Batch-level drift monitoring: limits, localisation, and the honest warnings."""

import json
import warnings

import numpy as np
import pytest

from pypasi import BandNegotiationClassifier, ControlProfile, ControlReport
from pypasi.datasets import make_conflict_signals
from pypasi.monitor import MONITORED_STATISTICS

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


@pytest.fixture(scope="module")
def cohort():
    """One acquisition, split into a reference part and unseen rows from it.

    "In control" means *this same process*, so the clean batches below are rows
    of the same acquisition that the limits were not built from - not a fresh
    draw from the generator, which is a different process and which the chart is
    supposed to notice.
    """
    X, y, axis, _ = make_conflict_signals(
        n_samples=3600, noise=0.85, signal_strength=0.45, random_state=0)
    return X[:2400], y[:2400], axis, X[2400:]


@pytest.fixture(scope="module")
def fitted(cohort):
    X, y, axis = cohort[:3]
    return BandNegotiationClassifier(
        axis=axis, n_bands=7, axis_name="cm-1", random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def profile(cohort, fitted):
    X = cohort[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ControlProfile.fit(fitted, X, n_batches=20, laser_nm=633.0,
                                  random_state=0)


def _fresh(n, seed, corrupt=None):
    """Unseen rows of the reference acquisition, optionally corrupted."""
    rng = np.random.default_rng(seed)
    X, _, _, _ = make_conflict_signals(
        n_samples=3600, noise=0.85, signal_strength=0.45, random_state=0)
    X = X[2400:][rng.choice(1200, size=min(n, 1200), replace=n > 1200)]
    if corrupt is not None:
        X = X.copy()
        X[:, corrupt] *= 1.8
    return X


# ------------------------------------------------------------------- fitting


def test_profile_reports_its_configuration(profile):
    assert profile.n_batches_ == 20
    assert len(profile.labels_) == 7
    assert profile.axis_name_ == "cm-1"
    frame = profile.to_frame()
    assert len(frame) == 7 * len(profile.statistics)
    assert (frame["ucl"] >= frame["lcl"]).all()
    assert {"statistic", "label", "centre", "sd", "lcl", "ucl"} <= set(frame.columns)


def test_random_split_warns_that_limits_will_be_tight(cohort, fitted):
    """The caveat is part of the API, not a footnote in the docs."""
    with pytest.warns(UserWarning, match="random split"):
        ControlProfile.fit(fitted, cohort[0], n_batches=20, random_state=0)


def test_explicit_batches_do_not_warn(cohort, fitted):
    X = cohort[0]
    runs = [X[i::10] for i in range(10)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ControlProfile.fit(fitted, batches=runs)


def test_few_batches_warn_about_the_standard_deviation(cohort, fitted):
    X = cohort[0]
    with pytest.warns(UserWarning, match="poorly determined"):
        ControlProfile.fit(fitted, batches=[X[i::4] for i in range(4)])


def test_quantile_limits_need_no_distribution(cohort, fitted):
    X = cohort[0]
    runs = [X[i::10] for i in range(10)]
    p = ControlProfile.fit(fitted, batches=runs, method="quantile")
    f = p.to_frame()
    assert (f["lcl"] <= f["centre"]).all() and (f["ucl"] >= f["centre"]).all()
    assert p.check(_fresh(400, 7), name="q").n_out_of_control >= 0


# -------------------------------------------------------------------- checks


def test_clean_batches_rarely_breach(profile):
    """Rarely, not never.

    Fourteen band-statistics charted at three sigma will throw the occasional
    single flag on a good batch - that is what a three-sigma limit means. The
    property worth testing is the rate, not any one draw.
    """
    reps = [profile.check(_fresh(400, 100 + i), name=f"clean {i}") for i in range(8)]
    assert isinstance(reps[0], ControlReport)
    assert reps[0].n_samples == 400
    assert len(reps[0].table) == 7 * len(profile.statistics)
    in_control = sum(r.is_in_control for r in reps)
    assert in_control >= 6, [r.summary() for r in reps if not r.is_in_control]
    assert np.median([r.drift_score for r in reps]) < profile.k


def test_corrupted_batch_is_flagged_in_the_right_interval(profile):
    """The output that matters is a wavenumber interval, not a score."""
    rep = profile.check(_fresh(600, 12, corrupt=slice(200, 260)), name="dirty")
    assert not rep.is_in_control
    assert rep.drift_score > profile.k
    worst = rep.worst
    # features 200:260 of a 400-point 400-1800 cm-1 axis fall in 1100-1310
    assert worst["cm-1_lo"] <= 1200 <= worst["cm-1_hi"], rep.summary()
    assert worst["label"] in rep.bands_out_of_control()


def test_report_summary_names_the_interval(profile):
    rep = profile.check(_fresh(400, 13, corrupt=slice(200, 260)), name="run 9")
    text = rep.summary()
    assert "run 9" in text and "cm-1" in text and "z =" in text
    assert "IN CONTROL" not in text


def test_out_of_control_is_sorted_worst_first(profile):
    rep = profile.check(_fresh(600, 15, corrupt=slice(200, 260)))
    z = rep.out_of_control["z"].abs().to_numpy()
    assert (np.diff(z) <= 1e-9).all()


def test_wavelength_columns_appear_when_the_laser_is_known(profile):
    rep = profile.check(_fresh(300, 16))
    assert {"nm_lo", "nm_hi"} <= set(rep.table.columns)
    assert (rep.table["nm_lo"] > 600).all() and (rep.table["nm_hi"] < 800).all()


# ------------------------------------------------------------ batch size


def test_limits_widen_for_a_smaller_batch(cohort, fitted):
    """The charted value is a batch mean, so its spread depends on batch size.

    Without rescaling, a batch an order of magnitude smaller than the reference
    runs is judged against limits built for a much steadier statistic, and
    false alarms follow. This is the property that makes the chart usable when
    batch sizes vary, which in a laboratory they always do.
    """
    X = cohort[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = ControlProfile.fit(fitted, X, n_batches=4, random_state=0)  # ~600 each
    small = [_fresh(60, 200 + i) for i in range(8)]
    scaled = sum(p.check(b, scale_limits=True).n_out_of_control for b in small)
    unscaled = sum(p.check(b, scale_limits=False).n_out_of_control for b in small)
    assert scaled < unscaled, (scaled, unscaled)
    assert p.check(small[0]).limit_scale > 1.0


# ----------------------------------------------------------------- sequences


def test_in_control_summary_says_so(profile):
    reps = [profile.check(_fresh(400, 600 + i), name="ok") for i in range(6)]
    assert any("IN CONTROL" in r.summary() for r in reps)


def test_check_many_gives_one_row_per_batch(profile):
    batches = [_fresh(200, 300 + i) for i in range(4)]
    batches += [_fresh(200, 400 + i, corrupt=slice(200, 260)) for i in range(2)]
    frame = profile.check_many(batches, [f"run {i}" for i in range(6)])
    assert len(frame) == 6
    assert {"batch", "drift_score", "n_out_of_control", "in_control"} <= set(frame.columns)
    assert all(f"z_{lab}" in frame.columns for lab in profile.labels_)
    assert not frame["in_control"].iloc[4:].any()
    assert frame["drift_score"].iloc[4:].min() > frame["drift_score"].iloc[:4].max()


def test_variance_decomposition_separates_the_two_components(profile):
    """Sampling noise shrinks with batch size; a real batch effect does not."""
    assert profile.sigma_within_ is not None
    assert (profile.sigma_within_ >= 0).all()
    assert (profile.sigma_between_ >= 0).all()
    small = profile.effective_sd(50)
    large = profile.effective_sd(5000)
    assert (large <= small + 1e-12).all()
    # and it stops shrinking once the between-batch part dominates
    assert (large >= profile.sigma_between_ - 1e-12).all()


def test_check_many_rejects_mismatched_names(profile):
    with pytest.raises(ValueError, match="names for"):
        profile.check_many([_fresh(100, 1)], ["a", "b"])


# --------------------------------------------------------------- persistence


def test_profile_round_trips_through_json(profile, fitted, tmp_path):
    path = tmp_path / "profile.json"
    profile.save(path)
    json.loads(path.read_text())                     # must be plain JSON
    loaded = ControlProfile.load(path, estimator=fitted)
    X = _fresh(300, 21)
    a = profile.check(X, name="x").table
    b = loaded.check(X, name="x").table
    assert np.allclose(a["z"].to_numpy(dtype=float), b["z"].to_numpy(dtype=float),
                       equal_nan=True)


def test_loaded_profile_refuses_a_different_configuration(profile, cohort, tmp_path):
    """A profile is a set of numbers attached to specific bands; say so loudly."""
    X, y, axis = cohort[:3]
    other = BandNegotiationClassifier(
        axis=axis, n_bands=5, axis_name="cm-1", random_state=0).fit(X, y)
    path = tmp_path / "p.json"
    profile.save(path)
    with pytest.raises(ValueError, match="different configuration"):
        ControlProfile.load(path, estimator=other)


def test_loaded_profile_without_an_estimator_says_so(profile, tmp_path):
    path = tmp_path / "p.json"
    profile.save(path)
    bare = ControlProfile.load(path)
    with pytest.raises(RuntimeError, match="no fitted estimator"):
        bare.check(_fresh(100, 22))


# -------------------------------------------------------------------- errors


def test_unknown_statistic_is_rejected(cohort, fitted):
    with pytest.raises(ValueError, match="unknown statistic"):
        ControlProfile.fit(fitted, cohort[0], statistics=("vibes",))


@pytest.mark.parametrize("kw,msg", [
    ({"method": "eyeball"}, "method must be"),
    ({"k": 0.0}, "k must be positive"),
    ({"n_batches": 1}, "at least 2 reference batches"),
    ({"n_batches": 10 ** 6}, "asked for"),
])
def test_bad_arguments_are_rejected(cohort, fitted, kw, msg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=msg):
            ControlProfile.fit(fitted, cohort[0], **kw)


def test_fit_needs_data(fitted):
    with pytest.raises(ValueError, match="either X or batches"):
        ControlProfile.fit(fitted)


def test_check_rejects_a_single_spectrum(profile):
    with pytest.raises(ValueError, match="must be 2-D"):
        profile.check(np.zeros(400))


def test_every_advertised_statistic_can_be_charted(cohort, fitted):
    X = cohort[0]
    runs = [X[i::10] for i in range(10)]
    p = ControlProfile.fit(fitted, batches=runs, statistics=MONITORED_STATISTICS)
    rep = p.check(_fresh(300, 31))
    assert set(rep.table["statistic"]) == set(MONITORED_STATISTICS)


# ------------------------------------------------------------------ figures


@pytest.mark.parametrize("dark", [False, True])
def test_control_figures_render(profile, dark, tmp_path):
    from pypasi import viz

    rep = profile.check(_fresh(400, 41, corrupt=slice(200, 260)), name="run 41")
    frame = profile.check_many([_fresh(150, 50 + i) for i in range(4)])
    for i, fig in enumerate([viz.plot_control_chart(rep, dark=dark),
                             viz.plot_control_chart(rep, dark=dark,
                                                    statistic="mean_stress"),
                             viz.plot_control_trend(frame, dark=dark)]):
        out = tmp_path / f"m{i}_{int(dark)}.png"
        fig.savefig(out, dpi=60)
        assert out.stat().st_size > 1000
        matplotlib.pyplot.close(fig)


def test_control_chart_works_for_an_in_control_batch(profile, tmp_path):
    from pypasi import viz

    fig = viz.plot_control_chart(profile.check(_fresh(300, 42), name="quiet"))
    fig.savefig(tmp_path / "quiet.png", dpi=60)
    matplotlib.pyplot.close(fig)
