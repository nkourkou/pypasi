import numpy as np
import pytest

from pypasi import BandSet, degrade, noise_level, noise_limit


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    axis = np.linspace(400, 1800, 500)
    peaks = rng.uniform(450, 1750, 12)
    clean = sum(np.exp(-0.5 * ((axis - p) / 10) ** 2) for p in peaks)
    X = clean[None, :] + rng.normal(0, 0.01, (60, axis.size))
    X = (X - X.min(1, keepdims=True)) / (
        X.max(1, keepdims=True) - X.min(1, keepdims=True)
    )
    return axis, X, BandSet.equal_width(axis, 7)


def test_plant_ground_truth_and_clean_rows_untouched(data):
    axis, X, bands = data
    p = degrade.plant(X, axis, bands=bands, random_state=1)
    assert p.degraded.sum() == 30
    assert np.array_equal(p.X[p.degraded == 0], X[p.degraded == 0])
    assert set(p.kind[p.degraded == 1]) <= set(degrade.PAPER_FAMILIES)
    assert np.all(
        (p.severity[p.degraded == 1] >= 0.3) & (p.severity[p.degraded == 1] <= 1.0)
    )
    local = p.kind == "local contamination"
    assert np.all(p.where[local] >= 0) and np.all(p.where[~local] == -1)


def test_plant_is_reproducible(data):
    axis, X, bands = data
    a = degrade.plant(X, axis, bands=bands, random_state=3)
    b = degrade.plant(X, axis, bands=bands, random_state=3)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.kind, b.kind)


def test_local_fault_stays_in_its_interval(data):
    axis, X, bands = data
    p = degrade.plant(
        X,
        axis,
        bands=bands,
        families=["local contamination"],
        fraction=1.0,
        random_state=2,
    )
    for i in range(len(X)):
        b = bands[int(p.where[i])]
        inside = (axis >= b.lo) & (axis < b.hi)
        assert np.allclose(p.X[i, ~inside], X[i, ~inside])
        assert not np.allclose(p.X[i, inside], X[i, inside])


def test_local_family_needs_intervals(data):
    axis, X, _ = data
    with pytest.raises(ValueError, match="band intervals"):
        degrade.plant(
            X, axis, families=["local contamination"], fraction=1.0, random_state=0
        )


def test_every_family_changes_the_spectrum(data):
    axis, X, bands = data
    for name in degrade.FAMILIES:
        p = degrade.plant(
            X[:5],
            axis,
            bands=bands,
            families=[name],
            fraction=1.0,
            severity=(1.0, 1.0),
            random_state=0,
        )
        assert not np.allclose(p.X, X[:5]), name


def test_custom_family(data):
    axis, X, bands = data

    def offset(x, axis, s, rng, edges):
        return x + s, -1

    fam = degrade.Family("offset", "global", offset, "adds s")
    p = degrade.plant(
        X, axis, families=[fam], fraction=1.0, severity=(0.5, 0.5), random_state=0
    )
    assert np.allclose(p.X, X + 0.5)
    assert set(p.kind) == {"offset"}


def test_unknown_family_is_rejected(data):
    axis, X, _ = data
    with pytest.raises(ValueError, match="unknown family"):
        degrade.plant(X, axis, families=["gremlins"])


def test_axis_shift_moves_a_peak(data):
    axis = np.linspace(400, 1800, 1401)
    x = np.exp(-0.5 * ((axis - 1000) / 5) ** 2)
    y = degrade.axis_shift(x, axis, 4.0)[0]
    assert axis[np.argmax(y)] == pytest.approx(1004.0, abs=1.0)


def test_noise_level_tracks_added_noise(data):
    axis, X, _ = data
    rng = np.random.default_rng(0)
    base = noise_level(X)
    noisy = noise_level(X + rng.normal(0, 0.05, X.shape))
    assert np.all(noisy > base)
    assert np.median(noisy) == pytest.approx(0.05, rel=0.3)
    assert noise_limit(X, q=0.5) == pytest.approx(np.median(base))


def test_noise_level_ignores_isolated_spikes(data):
    axis, X, _ = data
    spiky = degrade.cosmic_rays(X, 3, 5.0, rng=0)
    assert np.allclose(noise_level(spiky), noise_level(X), rtol=0.25)


def test_noise_level_validates_window():
    with pytest.raises(ValueError):
        noise_level(np.zeros((2, 50)), window=8)


def test_stress_test_reports_every_family():
    from pypasi import PLSDA, BandNegotiationClassifier, Triage
    from pypasi.datasets import make_conflict_signals

    X, y, axis, _ = make_conflict_signals(
        n_samples=600, n_classes=3, n_points=300, random_state=0
    )
    X = (X - X.min(1, keepdims=True)) / (
        X.max(1, keepdims=True) - X.min(1, keepdims=True)
    )
    bands = BandSet.equal_width(axis, 5)
    clf = BandNegotiationClassifier(
        bands=bands,
        base_estimator=PLSDA(n_components=3),
        topology="complete",
        random_state=0,
    ).fit(X[:300], y[:300])
    tri = Triage(n_components=3).fit(clf, X[300:450], y[300:450])
    tab = degrade.stress_test(tri, X[450:], y[450:], axis, random_state=0)
    assert list(tab["family"])[-1] == "all"
    assert set(tab["family"][:-1]) == set(degrade.PAPER_FAMILIES)
    assert tab["n"].iloc[-1] == 75
    assert tab[["conflict", "1 - confidence", "novelty"]].notna().all().all()
