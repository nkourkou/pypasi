# Changelog

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
