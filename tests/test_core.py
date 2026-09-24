"""Bands, topology, logits, divergences, geometry, gates and regimes."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pypasi import (
    AbsoluteGate,
    BandSet,
    H1,
    H2,
    Plain,
    QuantileGate,
    RankGate,
    Topology,
    negotiate,
)
from pypasi.divergence import center, js_divergence, kl_divergence, softmax
from pypasi.geometry import (
    band_decision_geometry,
    decision_geometry,
    early_decision_geometry,
    eredg,
)
from pypasi.logits import to_logits
from pypasi.perturb import perturb_bands


@pytest.fixture
def axis():
    return np.linspace(400, 1800, 400)


# ------------------------------------------------------------------- bands


def test_equal_width_covers_range_without_overlap(axis):
    bs = BandSet.equal_width(axis, 7)
    assert bs.n_bands == 7
    assert bs.labels == [f"B{i}" for i in range(1, 8)]
    all_features = np.concatenate([b.features for b in bs])
    assert all_features.size == np.unique(all_features).size
    assert all_features.size == axis.size


def test_last_band_includes_its_upper_edge(axis):
    bs = BandSet.equal_width(axis, 4)
    assert axis.size - 1 in bs[-1].features.tolist()


def test_overlapping_bands_are_rejected(axis):
    from pypasi.bands import Band

    dup = np.arange(10)
    with pytest.raises(ValueError, match="overlaps"):
        BandSet(axis, [Band(0, "B1", 400, 500, dup), Band(1, "B2", 500, 600, dup)])


def test_empty_band_is_rejected(axis):
    # axis points are ~3.5 apart, so [401, 402) contains none of them
    with pytest.raises(ValueError, match="no features"):
        BandSet.from_edges(axis, [401.0, 402.0, 1800.0])


def test_peak_informed_places_edges_away_from_peaks(axis):
    rng = np.random.default_rng(0)
    peaks = [600.0, 1000.0, 1500.0]
    X = np.stack(
        [sum(np.exp(-0.5 * ((axis - p) / 20) ** 2) for p in peaks) + rng.normal(0, 0.01, axis.size)
         for _ in range(30)]
    )
    bs = BandSet.peak_informed(axis, X, 3)
    assert bs.n_bands == 3
    for p in peaks:
        owner = [b for b in bs if b.lo <= p <= b.hi]
        assert owner, f"peak at {p} fell outside every band"
        b = owner[0]
        # the peak should not sit right on a boundary
        assert min(abs(p - b.lo), abs(p - b.hi)) > 10


def test_peak_informed_falls_back_when_no_peaks(axis):
    flat = np.ones((5, axis.size))
    bs = BandSet.peak_informed(axis, flat, 5)
    assert bs.n_bands == 5


def test_split_and_feature_lookup(axis):
    bs = BandSet.equal_width(axis, 5)
    X = np.random.default_rng(0).normal(size=(9, axis.size))
    blocks = bs.split(X)
    assert len(blocks) == 5
    assert sum(b.shape[1] for b in blocks) == axis.size
    assert bs.band_of_feature(0) is bs[0]
    assert bs.band_of_feature(axis.size - 1) is bs[-1]


# ---------------------------------------------------------------- topology


@pytest.mark.parametrize("build,expected_edges", [
    (lambda: Topology.chain(5), 4),
    (lambda: Topology.complete(5), 10),
    (lambda: Topology.knn(5, k=2), 7),
])
def test_topology_edge_counts(build, expected_edges):
    t = build()
    assert t.n_edges == expected_edges
    assert np.allclose(t.adjacency, t.adjacency.T)
    assert np.all(np.diag(t.adjacency) == 0)


def test_chain_is_connected_and_isolated_graph_is_not():
    assert Topology.chain(6).is_connected
    assert not Topology.from_edges(4, [(0, 1)]).is_connected


def test_from_edges_can_extend_the_chain():
    t = Topology.from_edges(5, [(0, 4)], include_chain=True)
    assert t.n_edges == 5
    assert 4 in t.neighbours(0)


def test_radius_topology_respects_distance():
    t = Topology.radius([0.0, 1.0, 2.0, 10.0], radius=1.5)
    assert 1 in t.neighbours(0)
    assert 3 not in t.neighbours(0)


def test_resolve_accepts_strings_and_instances():
    assert Topology.resolve("chain", 4).name == "chain"
    assert Topology.resolve(Topology.complete(4), 4).n_edges == 6
    with pytest.raises(ValueError, match="unknown topology"):
        Topology.resolve("mesh", 4)


def test_asymmetric_adjacency_is_rejected():
    with pytest.raises(ValueError, match="symmetric"):
        Topology(np.array([[0.0, 1.0], [0.0, 0.0]]))


# ------------------------------------------------------------------ logits


def _toy(n=60, d=8, classes=2, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, n)
    X = rng.normal(size=(n, d)) + y[:, None]
    return X, y


def test_binary_logits_reproduce_predict_proba():
    """[0, d] must match the model's own probabilities; [-d, +d] would not."""
    X, y = _toy(classes=2)
    m = LogisticRegression().fit(X, y)
    L = to_logits(m, X)
    assert L.shape == (X.shape[0], 2)
    np.testing.assert_allclose(softmax(L), m.predict_proba(X), rtol=1e-9, atol=1e-12)


def test_doubled_binary_logits_would_not_match():
    X, y = _toy(classes=2)
    m = LogisticRegression().fit(X, y)
    d = m.decision_function(X)
    doubled = softmax(np.column_stack([-d, d]))
    assert not np.allclose(doubled, m.predict_proba(X), atol=1e-6)


def test_multiclass_decision_function_passes_through():
    X, y = _toy(classes=3)
    m = LogisticRegression().fit(X, y)
    np.testing.assert_allclose(to_logits(m, X), m.decision_function(X))


def test_predict_proba_fallback_for_estimators_without_decision_function():
    X, y = _toy(classes=3)
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    assert not hasattr(rf, "decision_function")
    L = to_logits(rf, X, n_classes=3)
    np.testing.assert_allclose(softmax(L), rf.predict_proba(X), atol=1e-9)


def test_logits_reject_wrong_class_count():
    X, y = _toy(classes=2)
    m = LogisticRegression().fit(X, y)
    with pytest.raises(ValueError, match="3 classes were expected"):
        to_logits(m, X, n_classes=3)


def test_unsupported_estimator_is_rejected():
    class Dumb:
        pass

    with pytest.raises(TypeError, match="neither decision_function"):
        to_logits(Dumb(), np.zeros((2, 2)))


# ------------------------------------------------------------- divergences


def test_softmax_is_stable_and_normalised():
    p = softmax(np.array([[1000.0, 1001.0], [-1000.0, -1001.0]]))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert np.all(np.isfinite(p))


def test_js_is_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    p = softmax(rng.normal(size=(20, 4)))
    q = softmax(rng.normal(size=(20, 4)))
    np.testing.assert_allclose(js_divergence(p, q), js_divergence(q, p))
    assert np.all(js_divergence(p, q) <= np.log(2) + 1e-9)
    np.testing.assert_allclose(js_divergence(p, p), 0.0, atol=1e-12)


def test_kl_is_zero_only_for_identical_distributions():
    p = softmax(np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(kl_divergence(p, p), 0.0, atol=1e-12)
    assert kl_divergence(p, softmax(np.array([[3.0, 2.0, 1.0]])))[0] > 0


def test_centering_does_not_mutate_input():
    a = np.arange(12.0).reshape(3, 4)
    before = a.copy()
    center(a)
    np.testing.assert_array_equal(a, before)


# ---------------------------------------------------------------- geometry


def _traj(seed=0, n=4, t=11, k=5, c=3):
    return np.random.default_rng(seed).normal(size=(n, t, k, c))


def test_dg_is_invariant_to_uniform_logit_shifts():
    h = _traj()
    shifted = h + np.random.default_rng(1).normal(size=(4, 11, 5, 1)) * 50
    np.testing.assert_allclose(decision_geometry(h), decision_geometry(shifted), atol=1e-9)


def test_stationary_trajectory_has_zero_dg():
    h = np.ones((3, 8, 4, 2))
    np.testing.assert_allclose(decision_geometry(h), 0.0)


def test_band_dg_sums_to_total_dg():
    h = _traj()
    np.testing.assert_allclose(band_decision_geometry(h).sum(axis=1), decision_geometry(h))


def test_early_dg_never_exceeds_total():
    h = _traj()
    assert np.all(early_decision_geometry(h, 3) <= decision_geometry(h) + 1e-9)


def test_eredg_masks_inactive_trajectories():
    h = _traj(n=10)
    h[0] = 1.0  # a completely stationary sample
    vals, delta = eredg(h, 3, gate_quantile=0.2)
    assert np.isnan(vals[0])
    assert np.isfinite(delta)
    assert np.isfinite(vals[1:]).any()


# ------------------------------------------------------------------- gates


def test_absolute_gate_thresholds():
    s = np.array([[0.0, 0.5, 1.0]])
    np.testing.assert_array_equal(
        AbsoluteGate(0.4, protect_min=0).mask(s), [[False, True, True]]
    )


def test_quantile_gate_calibrates_from_reference():
    rng = np.random.default_rng(0)
    ref = rng.random((200, 5))
    g = QuantileGate(q=0.9).fit(ref)
    assert 0.85 < g.tau_ < 0.95
    assert g.mask(ref).mean() < 0.15


def test_quantile_gate_refuses_to_run_unfitted():
    with pytest.raises(RuntimeError, match="must be fitted"):
        QuantileGate().mask(np.zeros((2, 3)))


def test_rank_gate_mutes_a_fixed_share():
    s = np.array([[0.1, 0.9, 0.5, 0.2]])
    m = RankGate(fraction=0.5, protect_min=0).mask(s)
    assert m.sum() == 2
    assert m[0, 1] and m[0, 2]


def test_protect_min_prevents_total_collapse():
    """Every band above threshold, yet one must survive."""
    s = np.array([[9.0, 8.0, 7.0]])
    guarded = AbsoluteGate(1.0, protect_min=1).mask(s)
    assert guarded.sum() == 2
    assert not guarded[0, 2]  # the quietest band is spared
    unguarded = AbsoluteGate(1.0, protect_min=0).mask(s)
    assert unguarded.all()


# ----------------------------------------------------------------- regimes


def _setup(n=40, k=6, c=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, k, c)) * 2.0


def test_plain_never_changes_weights():
    L = _setup()
    r = negotiate(L, regime="plain", random_state=0)
    np.testing.assert_allclose(r.weights_final, r.weight_history[:, 0, :])


def test_h2_is_distinguishable_from_h1():
    """Additive reinforcement changes weight ratios, so H2 must diverge from H1.

    A multiplicative rule would scale every surviving band equally, which the
    sum-normalised aggregation cancels, collapsing H2 onto H1.
    """
    L = _setup()
    gate = QuantileGate(q=0.6).fit(np.abs(np.random.default_rng(1).normal(size=(50, 6))) * 0.05)
    a = negotiate(L, regime=H1(), gate=gate, random_state=0)
    b = negotiate(L, regime=H2(), gate=gate, random_state=0)
    assert not np.allclose(a.logit_history, b.logit_history)
    assert not np.allclose(a.weights_final, b.weights_final)


def test_h2_lets_a_silenced_band_recover():
    """A muted band sits at weight zero; addition can lift it, multiplication cannot."""
    reg = H2(gamma=0.25)
    w = np.zeros((1, 3))
    muted = np.array([[False, False, False]])
    frozen = np.zeros((1, 3), dtype=bool)
    L = np.zeros((1, 3, 2))
    _, w_new = reg.propose(L, L, w, muted, frozen)
    assert np.all(w_new > 0)


def test_unknown_regime_is_rejected():
    with pytest.raises(ValueError, match="unknown regime"):
        negotiate(_setup(), regime="H9")


def test_gated_regime_requires_a_gate():
    with pytest.raises(ValueError, match="needs a gate"):
        negotiate(_setup(), regime="H1", gate=None)


# --------------------------------------------------------------- negotiate


def test_negotiation_is_reproducible():
    L = _setup()
    kw = dict(regime="plain", max_iter=12)
    a = negotiate(L, random_state=7, **kw)
    b = negotiate(L, random_state=7, **kw)
    np.testing.assert_array_equal(a.logit_history, b.logit_history)
    np.testing.assert_array_equal(a.y_pred, b.y_pred)


def test_history_shapes_line_up():
    L = _setup(n=9, k=5, c=4)
    r = negotiate(L, regime="plain", max_iter=10)
    assert r.logit_history.shape == (9, 11, 5, 4)
    assert r.stress_history.shape == (9, 10, 5)
    assert r.weight_history.shape == (9, 11, 5)
    assert r.proba.shape == (9, 4)
    assert r.band_dg.shape == (9, 5)


def test_geometry_requires_a_recorded_trajectory():
    r = negotiate(_setup(), regime="plain", record_trajectory=False)
    with pytest.raises(RuntimeError, match="record_trajectory=True"):
        _ = r.dg


def test_frozen_bands_never_move():
    L = _setup(n=5, k=4)
    frozen = np.zeros((5, 4), dtype=bool)
    frozen[:, 1] = True
    r = negotiate(L, regime="plain", frozen=frozen, random_state=0)
    np.testing.assert_allclose(r.logits_final[:, 1, :], L[:, 1, :])


def test_topology_changes_the_outcome():
    L = _setup()
    chain = negotiate(L, topology="chain", regime="plain", random_state=0)
    full = negotiate(L, topology="complete", regime="plain", random_state=0)
    assert not np.allclose(chain.logits_final, full.logits_final)


def test_zero_temperature_is_strictly_downhill():
    L = _setup()
    r = negotiate(L, regime="plain", temperature=0.0, max_iter=20, random_state=0)
    assert np.all(np.diff(r.energy_history, axis=1) <= 1e-9)


def test_collapse_is_flagged_when_the_guard_is_disabled():
    L = _setup(n=6, k=4) * 6.0
    r = negotiate(L, regime="H1", gate=AbsoluteGate(1e-9, protect_min=0), random_state=0)
    assert r.collapsed.any()
    # a collapsed sample gets a uniform distribution, not a silent tie-break
    np.testing.assert_allclose(r.proba[r.collapsed].sum(axis=1), 1.0)
    assert np.allclose(r.proba[r.collapsed], 1.0 / L.shape[2])


def test_default_guard_prevents_collapse():
    L = _setup(n=6, k=4) * 6.0
    r = negotiate(L, regime="H1", gate=AbsoluteGate(1e-9), random_state=0)
    assert not r.collapsed.any()


# -------------------------------------------------------------- perturbation


@pytest.mark.parametrize("mode", ["noise", "spike", "silence", "swap"])
def test_perturbation_hits_the_same_band_count_in_every_mode(mode):
    L = _setup(n=12, k=8)
    _, _, affected = perturb_bands(L, 0.5, mode, random_state=0)
    assert np.all(affected.sum(axis=1) == 4)


def test_silence_mode_freezes_what_it_damages():
    L = _setup(n=6, k=8)
    _, frozen, affected = perturb_bands(L, 0.25, "silence", random_state=0)
    np.testing.assert_array_equal(frozen, affected)


def test_zero_fraction_is_a_no_op():
    L = _setup()
    d, frozen, affected = perturb_bands(L, 0.0, "noise", random_state=0)
    np.testing.assert_array_equal(d, L)
    assert not frozen.any() and not affected.any()


def test_unaffected_bands_are_untouched():
    L = _setup(n=7, k=6)
    d, _, affected = perturb_bands(L, 0.5, "noise", random_state=3)
    np.testing.assert_allclose(d[~affected], L[~affected])
