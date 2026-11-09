# Changelog

## 0.5.5 (unreleased: SoftwareX revision)

- **New** `BandNegotiationClassifier.band_accuracy(X, y)`: each band's own accuracy against the majority-class rate, with a binomial test. The audit needs class information in every band, and this is the check.

- **New** `docs/parameter_selection_guide.md`: a default and the conditions for departing from it, for scaling, band estimator, band count, width, content and placement, topology and regime, summarised in Section 2.3 of the paper. It is measured on the Ho et al. cohort over five patient-disjoint splits, with a coarser-sampling test for spectral resolution.
- **New** `benchmarks/parameter_guide.ipynb`, which produces the guide's tables and checks them against the values quoted.

- **New** `pypasi.degrade`: eight fault families that act on spectra rather than on band evidence. Five are the families of the paper (axis shift, resolution loss, throughput dip, short exposure, local contamination) and three are new (cosmic rays, fluorescence, intensity loss). It also provides `plant()`, which returns the ground truth for every spectrum, and `stress_test()`, which reports conflict, 1 − confidence and novelty AUROCs and the paired accuracy cost per family. A new fault is one function wrapped in `Family`. With the default families and seed, `plant()` reproduces the defects of the SoftwareX notebook bit for bit.
- **New** `docs/degradation.md`, and `benchmarks/short_exposure.ipynb` for its section on what the audit cannot see.
- **Changed** `examples/softwarex_examples.ipynb` (the paper's notebook, now in the release) plants its defects with `pypasi.degrade`. It also reports the accuracy cost of the defects on the same spectra before and after them (4.4 points; the submitted manuscript compared two different halves of the cohort, 3.6) and checks 28 manuscript numbers.

- **New** `benchmarks/method_paper_comparison.ipynb`: the configuration of Kourkoumelis (2026) (L2 normalization with standardization, logistic-regression band models, chain topology, eREDG), reproduced in pypasi and changed one component at a time to the defaults. Conflict rises from 0.54 to 0.76 on the planted defects and from 0.60 to 0.75 on the real 2018 → 2019 change, in all five splits.

- **New** `benchmarks/outlier_comparison.ipynb`: conflict against PCA Hotelling's T², the Q residual, an isolation forest and a local outlier factor, on the planted defects and on the real 2018 → 2019 change, over five splits. Adding conflict to T² + Q + confidence raises the held-out AUROC by 12.9 points on the planted defects and by 4.9 points on the real change, in all five splits.

- **New** `benchmarks/instrument_change.ipynb`: the audit on the documented 2018 → 2019 change in the Ho et al. instrument. Conflict separates the 2019 spectra from held-out 2018 spectra (AUROC 0.75 ± 0.05 per spectrum and 0.86 per batch of 100, over five splits); confidence stays at chance (0.50).

- **New** `pypasi.quality.noise_level` (robust spread of the Savitzky–Golay residual) and `noise_limit` (a limit learned from clean spectra). They cover faults that act identically on every channel, which the band audit cannot see by construction.

- **Changed** `ControlProfile.fit(random_state=0)` by default, so the same reference gives the same limits on every run (it was `None`). The paper's numbers are unchanged: its notebooks always passed a seed.
- **New** CI runs Python 3.11–3.14 on Linux and 3.12 on macOS and Windows, and `pyproject.toml` lists the supported versions. Python 3.10 reached end of life in October 2026 and is no longer supported (the suite still passes on it).
- **Changed** `pypasi.viz` raises an `ImportError` that explains how to install the optional matplotlib dependency (`pip install 'pypasi[viz]'`). Nothing else needs matplotlib; the CLI skips figures without it.

### Packaging

- `requires-python = ">=3.11"`, matching the tested range (3.11–3.14). The classifiers are all valid PyPI classifiers, so the package can be uploaded.
- The description names Raman spectra explicitly. The README installs from a tagged release.
- Build outputs (`build/`, `dist/`) and the data folder (`Bacteria/`) are no longer tracked, and a benchmark script that belongs to a later study has been removed.

- **New** `benchmarks/bench_runtime.ipynb`: the wall time and peak memory of every user-facing step on 1,000 spectra, against the same base estimator fitted once on the full spectrum. It runs on the Ho et al. data (`DATA_DIR`) or on synthetic data of any width (`DATA_DIR = None`).

### `BandSet.peak_informed` returned fewer bands than requested

The default `min_width` was a tenth of the range whatever the band count, and boundaries closer than that were dropped without a warning. On the Ho et al. bacteria cohort a request for 7 bands returned 6, and 10 or 14 returned 9. Boundaries were also chosen between the most prominent *peaks*, so a cluster of strong peaks produced crowded cuts that were then pruned.

- **Changed** boundaries are the deepest *troughs* of the smoothed mean spectrum, taken greedily under the width constraint. If too few qualify, the widest bands are halved.
- **Changed** the default `min_width` is `0.5 * (hi - lo) / n_bands`.
- **New** a `UserWarning` whenever fewer bands than requested can be placed, and a `ValueError` when `min_width * n_bands` exceeds the range.
- `equal_width`, the default everywhere in the package and in the paper, is unaffected.

## 0.5.4

### Per-band residual statistics in the control chart

`ControlProfile` charted only what the negotiation produces: how much each band
disagrees with its neighbours and how sure each band is of itself. Those
statistics answer whether a batch *disturbs the model*. They do not answer
whether the *input* changed in a band the model happens to tolerate, which is
what a classical multivariate chart of PCA residuals does, for the whole
spectrum at once.

- **New** statistics `residual_q` (batch mean of the log mean squared PCA
  residual inside each band) and `residual_mean` (batch mean of the signed
  residual inside each band, which keeps the direction Q throws away), listed
  in `RESIDUAL_STATISTICS` and accepted by `ControlProfile.fit(statistics=...)`.
- **New** `ControlProfile.fit(..., residual_model=...)`: a fitted
  `NoveltyDetector`, required for the residual statistics. Fit it on the
  estimator's training spectra, not on the reference batches.
- Residual and negotiation statistics can be charted together under one
  calibrated limit; the empirical multiplicity correction covers all of them.
- A residual-only profile does not run the negotiation at all.
- `attach`, `load` and `check` take `residual_model=`; the profile's fingerprint
  records the residual model's rank and width and refuses a mismatched one.

## 0.5.3

### Novelty and PCA-LDA were not reproducible

`NoveltyDetector` and `PCALDA` built their `PCA` without a solver or a seed.
scikit-learn's `svd_solver="auto"` selects the **randomized** solver once the
cohort exceeds 500 in its larger dimension, which is every spectroscopic
cohort, and that solver draws from the **global NumPy stream** when nothing
seeds it. Two fits of one cohort therefore returned different models depending
only on how much unrelated code had run first. Measured on a 2000 x 1000
cohort: the Hotelling T-squared limit moved by 4% and the Q residual limit by
0.2% between two fits of identical data.

Control limits that depend on execution order cannot be filed, which is the
one thing a quality-control chart has to be able to do.

- **Changed** `NoveltyDetector` and `PCALDA` now default to `svd_solver="full"`,
  the exact decomposition. Results are bit-identical across runs, processes and
  call orders, with no seed required. Cost at the shapes these classes are used
  at is small: fitting `NoveltyDetector` on 2000 x 1000 went from 0.40 s to
  0.57 s.
- **New** both classes accept `svd_solver` and `random_state`, so a cohort large
  enough to make the exact decomposition costly can opt back into
  `svd_solver="randomized"` with a seed and stay reproducible.
- **New** `tests/test_determinism.py`, which fits each affected class twice with
  different amounts of the global stream drained in between. Its first test
  asserts that the fixture is still large enough to select the randomized
  solver, so a future scikit-learn release cannot silently disarm the rest.

**This changes reported numbers.** Every novelty-derived quantity shifts, because
the exact decomposition is not the approximation that was being computed before.
Conflict, confidence, band stress and the control-chart localisation are
untouched: they never go through this PCA.

## 0.5.2 (obsolete)

### Policy calibration and orientation can come from different data

`Triage.fit` took one `X_cal` and used it for three things: the novelty limits,
the review threshold, and the direction of the conflict descriptor. Those have
incompatible requirements and the API had no way to say so.

The limits and the threshold need **clean, held-out** signals — calibrate them on
a cohort that is half corrupted and the outlier threshold moves out to
accommodate the very corruption it exists to flag. The orientation needs signals
**labelled for the event being flagged**, which in practice means a cohort with
known defects in it. One array cannot be both.

- **New** `Triage.fit(..., orientation_data=(X, target))`: learn the sign from a
  separate labelled cohort while the policy is calibrated on clean held-out data.
  Mutually exclusive with `orient_on`.

## 0.5.1

### `Triage` can be oriented on the event you actually want flagged

`Triage.fit` learned which direction of a conflict descriptor means trouble by
regressing it against **wrong predictions**, and only that. For a descriptor that
is near chance against errors, the sign is then decided by noise — and getting it
backwards turns a descriptor that separates at *p* into one that separates at
*1 − p*.

This was not hypothetical. On the Ho bacteria cohort, `dg` scored 0.467 against
errors on the calibration split, the sign flipped, and the descriptor screen
recorded **0.246** against planted degradation for a quantity that runs at
**0.754**. The published descriptor was selected over it on that basis.

- **New** `Triage.fit(..., orient_on=...)`: a binary target for the orientation
  step. The default is unchanged (wrong predictions), so existing code behaves
  as before.
- **New** `Triage.orientation_margin_`: how far the deciding AUROC sat from 0.5.
- **New** `WeakOrientationWarning`, raised when that margin is under 0.05 — the
  regime in which the sign is close to arbitrary.

## 0.5.0

Three changes, one of which affects numbers already published with 0.4.x.

### Class evidence is now typed, and calibrated by default

`to_logits` preferred `decision_function` for every estimator. That is an exact
log-odds for logistic regression and LDA, and an arbitrary scale for an SVM
margin or a PLS-DA response — yet confidence weights, stress, the gate threshold
and DG are all computed from `softmax` of it. Two estimators on two scales
therefore produced conflict numbers that were never comparable.

- **New** `pypasi.evidence`: `to_evidence` returns `(logits, EvidenceSpec)`,
  where the spec records the kind (`exact_logit`, `log_proba`,
  `uncalibrated_score`, `calibrated_log_proba`), the source method, and the
  scale. `EvidenceSpec.warn_if_uncalibrated()` makes the problem loud.
- **New** `TemperatureCalibrator`: one scalar temperature fitted by held-out
  negative log-likelihood. Monotone and order-preserving, so no band's
  prediction changes; only the sharpness of its distribution does.
- **New** `BandNegotiationClassifier(calibration=...)`, default `"auto"`: fits a
  temperature when the base estimator's evidence is not already an exact
  log-odds. `"always"` calibrates regardless, `"per_band"` fits one per band,
  `None` reproduces 0.4.x and warns.
- **New** `BandNegotiationClassifier.evidence_report`.

**Migration.** With the default base estimator, `"auto"` is a no-op and nothing
changes. With `PLSDA`, `LinearSVC` or another score-scale estimator, band
evidence is now temperature-corrected and `calibration_split` (default 0.25) of
the training data is held back to fit it. Pass `calibration=None` for the old
numbers, or `calibration_data=(X, y)` to calibrate without spending training
data.

**Measured effect** on synthetic conflict signals, conflict against a planted
band-local defect: PLS-DA 0.620 → 0.641, linear SVM 0.569 → 0.615, and no change
for logistic regression or PCA-LDA. Roughly half of PLS-DA's apparent advantage
over linear SVM was a units artefact.

### Auditing a classifier the library did not fit

- **New** `pypasi.external`: `audit_external(host, X, y, ...)` returns an
  `ExternalAudit` that leaves the host model's predictions untouched and reports
  band-resolved conflict, novelty and the worst interval alongside them.
  `ControlProfile.fit` accepts one directly.
- `ExternalAudit.validate(X, y)` measures conflict against the *host's* errors,
  because that relationship is host-specific and does not transfer.

### Control limits account for how many things are being charted

Seven bands on two statistics is fourteen simultaneous tests; six statistics is
forty-two. At a nominal `k=3` the measured batch-level false-alarm rate on
synthetic signals was 0.075 with two statistics and 0.155 with six, against a
nominal 0.003.

- **New** `ControlProfile(multiplicity=...)`, default `"empirical"`: calibrates
  the threshold from the distribution of the largest `|z|` each reference batch
  scores against a centre computed from the others. Measured rates fall to 0.020
  and 0.015 respectively, with the corrupted-batch drift score unchanged at 10.4
  against a threshold of 3.92 — the sensitivity is not what was traded.

  The centre is held out and the spread is not, deliberately: a leave-one-out
  standard deviation over n−1 batches collapses when the removed batch was
  carrying the spread, and an early version of this produced a threshold of 26
  sigma — a chart that can never fire. `resolution_` records the finest
  false-alarm rate the reference batches can resolve; at 30 batches that is
  0.033, which is why the measured rate lands near 0.02 rather than at the
  nominal 0.003. More reference batches is the only fix, and the profile says so
  rather than implying a precision it does not have.
- `"sidak"` and `"bonferroni"` apply the parametric correction; `"none"`
  reproduces 0.4.x. `"empirical"` never returns a threshold looser than
  `"sidak"`, which is what keeps it safe when there are too few batches to
  calibrate from.
- **New** `ControlProfile.k_effective`, `.n_tests`, `.limits_report()`.

**Migration.** Code comparing `report.drift_score` to a hard-coded `k` should
use `report.is_in_control`, or compare against `profile.k_effective`.

### Minority evidence

- **New** `AuditResult.minority_evidence()`: for each band the gate silenced,
  the class its own evidence supported before it was overruled. H1 suppresses
  precisely the region carrying an unusual signal, and until now the negotiated
  output kept no trace of what that region had argued for.
