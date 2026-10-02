# pypasi

**Process-aware signal inference.** Classify Raman spectroscopy (or any 1-D) signals by negotiated consensus
among band-level models, then audit *how* the decision was reached: which regions
of the signal disagreed, how long they held out, and what the raw data looks like
there.

Conventional classifiers report a label and a probability. Neither tells you
whether the prediction came from agreement across the signal or from the forced
reconciliation of contradictory regions. `pypasi` partitions a signal into
contiguous bands, gives each its own classifier, and lets those band agents
negotiate over an interaction graph. The trajectory they take is measurable, and
it is attributable back to specific regions of the spectrum.

The most direct use is monitoring. A fitted model keeps producing predictions
long after the data stops looking like what it was trained on, and accuracy —
which would say so — needs labels you do not have when a new batch arrives.
`pypasi` charts the band-level quantities the negotiation already produces as
process variables and reports **which wavenumber interval left its control
limits**, with no labels at all.

---

## Install

```bash
pip install -e .            # from a checkout
pip install -e ".[viz]"     # adds matplotlib helpers
```

Requires Python 3.10+, NumPy, SciPy, pandas and scikit-learn.

## Sixty seconds

```bash
pypasi demo --out results/demo --axis-name "cm-1"
```

Generates synthetic signals with conflict planted at a known position, fits the
model, compares all three regulatory regimes, and writes the band-conflict table
plus self-contained HTML reports.

In Python:

```python
from pypasi import BandNegotiationClassifier
from pypasi.datasets import make_conflict_signals

X, y, axis, spec = make_conflict_signals(n_samples=600, random_state=0)

clf = BandNegotiationClassifier(axis=axis, n_bands=7, axis_name="cm-1",
                                regime="H1", random_state=0).fit(X, y)

audit = clf.audit(X, y)
print(audit.conflict_table())        # which bands disagreed, and how much
print(audit.top_conflict_bands(3))   # ['B4', 'B2', 'B3']

trace = audit.trace_band("B4")       # the raw signal behind the worst band
audit.report("audit.html", sample=0) # interactive, self-contained
```

## What it does

### Bands

```python
from pypasi import BandSet

BandSet.equal_width(axis, 7, lo=400, hi=1800)      # uniform partition
BandSet.peak_informed(axis, X, 7)                  # boundaries in the troughs
BandSet.from_edges(axis, [400, 700, 1100, 1800])   # explicit
```

`peak_informed` places boundaries between the strongest peaks so that prominent
features are not split across two agents.

### Any scikit-learn classifier, including chemometric ones

```python
from pypasi import PLSDA, PCALDA
from sklearn.ensemble import RandomForestClassifier

BandNegotiationClassifier(axis=axis, base_estimator=PLSDA(n_components=8))
BandNegotiationClassifier(axis=axis, base_estimator=RandomForestClassifier())
```

scikit-learn ships the regression half of PLS but no discriminant wrapper, so
`pypasi` provides the two classifiers vibrational spectroscopy actually uses:

| Classifier | Why it is here |
| --- | --- |
| `PLSDA` | The workhorse of Raman and IR. Copes with far more variables than samples and with heavy collinearity. Exposes `vip_scores_`. |
| `PCALDA` | PCA then LDA - the standard route when spectra are wider than they are tall and LDA cannot be fitted directly. |

Both are ordinary estimators and both pass `check_estimator` in full. For the
rest, reach for scikit-learn directly: shrinkage LDA
(`LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")`) is well suited to
small classes, and elastic-net logistic regression and a linear SVM both work
well on normalised spectra.

**VIP scores.** `PLSDA.vip_scores_` gives Variable Importance in Projection per
wavenumber, the conventional measure of where a spectral model finds its
information, with 1.0 as the usual cut-off.

### Class evidence, and what scale it is on

Band agents exchange logits, and getting those out of an arbitrary estimator is
less obvious than it looks. Random forests and naive Bayes have no
`decision_function`, so `pypasi` falls back to `log(predict_proba)`. Binary
`decision_function` returns one number, and the natural-looking `[-d, +d]`
*doubles* the scale of a proper two-class logit vector, inflating every
divergence and breaking any threshold tuned elsewhere; `pypasi` uses `[0, d]`.

**That last construction reproduces the model's own probabilities exactly for a
logistic score, and not otherwise.** It is exact for binary and multinomial
logistic regression and for LDA, whose discriminant is a log posterior up to a
constant. It is *not* exact for an SVM margin, which lives in scaled-feature
units, nor for a PLS-DA response, which is a regression output onto one-hot
targets. For those, the softmax has an arbitrary temperature - and since
confidence weights, stress, the gate threshold and DG are all computed from that
softmax, two estimators on two scales produce conflict numbers that cannot be
compared.

The scales really do differ. On synthetic conflict signals the mean-centred
evidence has standard deviation 0.39 for PLS-DA and 7.86 for PCA-LDA, and mean
band confidence follows it from 0.51 to 0.88. A saturated softmax has little
room left to disagree, so the estimator with the gentler scale looks like the
one whose bands disagree most informatively - a statement about units.

Since 0.5.0 `pypasi.to_evidence` returns the values *and* an `EvidenceSpec`
saying what they are, and the estimator repairs the scale by default:

```python
from pypasi import BandNegotiationClassifier, PLSDA

clf = BandNegotiationClassifier(axis=axis, base_estimator=PLSDA(10)).fit(X, y)
print(clf.evidence_report)
#      estimator            source                   kind  temperature  nll_before  nll_after
#          PLSDA decision_function  calibrated_log_proba        5.657       0.739      0.348
```

A single temperature is fitted on data the band models never saw - one
parameter, monotone, order-preserving, so no band's prediction changes and only
the sharpness of its distribution moves. Per-band calibration is available
(`calibration="per_band"`) and is not the default, because making each band
individually well calibrated normalises away exactly how informative each band
is, which is the signal the audit consumes.

Note that an *exact* logit is not a *calibrated* one: L2 shrinks logistic
regression's coefficients, and its band models want a temperature near 7 on the
same data. `calibration="always"` repairs that too, at the cost of a held-out
split; `calibration="auto"` (the default) leaves exact-logit estimators alone so
that upgrading costs no training data. `calibration=None` reproduces 0.4.x
exactly and warns.

### Auditing a classifier you already have

Replacing a validated workflow is a high price for a diagnostic. `audit_external`
leaves the laboratory's own model in charge and fits a band ensemble alongside
it as an instrument:

```python
from pypasi import audit_external

ext = audit_external(house_plsda, X_fit, y_fit, axis=axis, n_bands=7)
ext.explain(todays_batch)
#   host_prediction  band_prediction  agreement  conflict  novelty  worst_band  ...
```

`host_prediction` is the answer of record and is never overridden. The
negotiation describes the *auxiliary* band system, not the internal reasoning of
the host model, and disagreement between the two is a flag to investigate rather
than a correction to apply. Degradation detection is a property of the spectra
and transfers to any host; the relationship between conflict and a *wrong host
prediction* does not, so `ext.validate(X, y)` measures it on labelled data
instead of assuming it.

### Interaction topology

The published method couples each band to its immediate neighbours. That cannot
express coupling between distant regions sharing a biochemical origin.

```python
from pypasi import Topology

Topology.chain(7)                                       # published default
Topology.complete(7)                                    # all-to-all
Topology.knn(7, k=2)                                    # two bands either side
Topology.radius(bands.centers, 300, decay=True)         # distance-weighted
Topology.from_edges(7, [(0, 6)], include_chain=True)    # chain plus long-range
```

Topology governs what an audit can see. A discordant band flanked by
uninformative neighbours has nothing to disagree with and stays invisible under
a chain; under a complete graph it does not.

### Regimes and gates

| Regime | Behaviour |
| --- | --- |
| `plain` | Unregulated diffusion toward the neighbourhood consensus |
| `H1` | Persistently discordant bands are silenced |
| `H2` | Suppression, plus reinforcement of coherent bands |

A **gate** decides what counts as discordant:

```python
from pypasi import QuantileGate, AbsoluteGate, RankGate

QuantileGate(q=0.975)    # threshold from clean training stress (default)
AbsoluteGate(tau=0.05)   # fixed threshold
RankGate(fraction=0.25)  # mute each signal's noisiest quarter
```

`QuantileGate` is the default because stress is a divergence between softmax
distributions: its scale depends on the number of classes and the sharpness of
the base learner, so a threshold tuned on one dataset rarely transfers to
another. Calibrating against the clean stress distribution makes the gate
comparable across problems.

Every gate takes `protect_min`, the number of least-discordant bands that can
never be silenced. It defaults to 1: if every band were silenced the weight
vector would be all zeros and the prediction would fall out of a tie-break rather
than out of inference. `protect_min=0` restores the unguarded behaviour.

### Geometry

From the recorded trajectory:

- **DG** — total reconciliation effort, the cumulative path length of all band
  agents in mean-centred logit space, and decomposable per band.
- **REDG** — the fraction of that effort spent in the opening iterations.
- **eREDG** — REDG with trivially inactive trajectories masked out, so the ratio
  is read only where there was real reconciliation to measure.

### Auditing, and the way back to the raw signal

```python
audit = clf.audit(X, y)
audit.summary()               # one row per signal
audit.conflict_table()        # one row per band
audit.trace_band("B4")        # raw spectra over exactly that interval
audit.to_csv("results/run")
audit.report("r.html", sample=12)
```

The conflict table separates two things that both raise stress and mean opposite
things. An **uninformative** band contributes near-uniform evidence that
disagrees with any confident consensus without contributing anything. A
**confidently wrong** band contributes sharp evidence for the wrong class. Only
the second is a real conflict. `band_confidence` distinguishes them and
`informed_conflict` combines them; with ground truth available, `excess_stress`
— conflict on wrong predictions minus conflict on right ones — is the decisive
column.

### Importance versus conflict

```python
importance, per_feature = clf.band_importance(X, y)   # PLS-DA VIP by default
```

Two different questions, and the pair is more informative than either alone.
Importance says *where a classifier finds information*; conflict says *where
inference runs into trouble*. A band high in both is informative but unstable -
the first place to look when a prediction fails. A band high in importance and
low in conflict is a dependable discriminant.

`trace_band` closes the loop: it returns the raw signal over exactly the
interval the table implicates, with per-class means, so a conflict score can be
checked against the spectrum that produced it.

### Comparing regimes

```python
from pypasi import compare_regimes
audits = clf.compare_regimes(X, y)          # one fit, three negotiations
print(compare_regimes(audits))
```

### Controlled perturbation

```python
from pypasi import perturb_bands
damaged, frozen, affected = perturb_bands(band_logits, 0.4, "spike", random_state=0)
```

Modes: `noise`, `spike`, `silence` (also freezes the band), `swap` (gives a band
another class's evidence). All corrupt the same number of bands at a matched
fraction, so modes are directly comparable.

## Monitoring a new batch, without labels

The question a laboratory actually has when spectra arrive: *has anything
changed, and where?*

```python
from pypasi import ControlProfile

profile = ControlProfile.fit(clf, X_reference, batches=runs, laser_nm=633)
report = profile.check(X_new_batch, name="2026-03-14")
print(report.summary())
```

```
2026-03-14: 480 spectra, 3 of 14 band-statistics outside 3 sigma
  worst: B5 (1196-1395 cm-1) on mean_stress, z = +8.4
  bands flagged: B5, B4, B6
```

`report.table` carries, per band and statistic, the reference centre and limits,
this batch's value, its standardised deviation and whether it breached — plus the
scattered wavelength of each band when `laser_nm` is given, because a throughput
problem lives at a place on the detector rather than at a Raman shift.

```python
frame = profile.check_many(batches, names)     # one row per batch, for a trend
viz.plot_control_chart(report)                 # this batch against its limits
viz.plot_control_trend(frame)                  # drift score across a run
profile.save("qc.json")                        # limits travel; re-attach a model
ControlProfile.load("qc.json", estimator=clf)
```

**This is not outlier detection.** `NoveltyDetector` asks whether a *spectrum* is
unlike the training data, per sample, in input space. This asks whether a *batch*
is disturbing the fitted model, answers per band, and hands back an interval in
cm⁻¹ you can take to the instrument. Run both.

### Three things that decide whether the chart is trustworthy

**Hold the reference out of fitting.** Band models fit their own training data
better than anything else, so a profile built on the spectra the estimator was
fitted on describes in-sample behaviour and flags every honest batch that
follows. The effect scales with overfitting — negligible with thousands of
training spectra per band, severe with a few hundred.

**Use real acquisition runs as reference batches.** Batches carved at random from
one cohort share an instrument, an operator and a session, so their spread
understates real batch-to-batch variation and the limits come out too tight.
`fit` warns when you pass `X` instead of `batches=`.

**Batch size is handled, not assumed.** The charted value is a batch mean, so a
Shewhart chart would shrink its limits as one over the square root of the batch
size — which is wrong whenever batches genuinely differ from each other, and
makes the chart cry wolf on any batch larger than the reference runs. `fit`
therefore splits each reference batch in half to separate the two components:

```
var(batch mean of m) = sigma_within^2 / m + sigma_between^2
```

The between-batch part is common to both halves and cancels in their difference,
leaving sampling noise; the spread of the batch means carries both. Limits then
shrink with batch size only as far as real batch effects allow, and stop.

Seven bands on two statistics is fourteen tests at once, so at three sigma a good
batch throws a single flag roughly once in thirty. A lone breach is a reason to
look; several bands at once, or the same band across consecutive batches, is a
reason to act.

## Quality-control triage

The practical use: flag predictions a clinician should not act on unreviewed.

```python
from pypasi import Triage

tri = Triage(review_rate=0.10).fit(clf, X_train, y_train)   # calibrate on clean data
report = tri.assess(X_new, y_new)                           # per-signal actions
print(report.action_counts)
```

**A prediction can be untrustworthy for two unrelated reasons, and conflating
them makes the flag useless.**

An **outlier** is a spectrum unlike anything in the training data - wrong
specimen, contamination, substrate change, a cosmic ray. It lives in input space,
and chemometrics has measured it for decades with the squared prediction error
and Hotelling's T² of a PCA model. Such a spectrum should be re-acquired; the
model was never entitled to an opinion about it.

**Internal disagreement** is different. The spectrum sits comfortably inside the
training distribution, yet its bands argue for different classes. This is a real
specimen that genuinely looks like two things at once - the case for a second
assay or an expert eye, not for re-measurement.

Crossing the two gives four actions rather than one flag:

| novelty | conflict | action | meaning |
| --- | --- | --- | --- |
| low | low | `accept` | coherent evidence, typical spectrum |
| low | **high** | `review` | ordinary spectrum, bands disagree - secondary validation |
| **high** | low | `remeasure` | specimen or acquisition is suspect |
| **high** | **high** | `reject` | both |

`review_rate` sets the conflict threshold as a quantile of the clean calibration
distribution, so the referral rate is a number you choose rather than one you
discover - which is the knob a clinic actually has.

### Proving it is not just outlier detection

```python
evidence = tri.evaluate(X_eval, y_eval)
```

Four questions, four answers:

- **`orthogonality`** - rank correlation between novelty and conflict, and the
  overlap between the two flags. Near zero is the claim.
- **`conditional`** - does conflict still separate right from wrong *after
  restricting to in-distribution signals*? If it does, it cannot be novelty
  detection wearing a different hat.
- **`among_confident`** - the manuscript's actual claim: is a conflicted
  prediction worth flagging *even when the classifier is confident about it*?
- **`incremental`** and **`risk_coverage`** - does it beat, or add to, the
  obvious baseline of low predicted probability? Weights for the combination are
  fitted on the calibration data, not chosen by hand.

**The direction is learned, not assumed.** Whether high or low eREDG indicates
trouble is dataset-dependent - the manuscript says so, and it is - so `fit`
determines the sign from calibration labels and records how it did:

```python
compare_conflict_scores(clf, X_cal, y_cal, X_eval, y_eval)
```

tries all four descriptors (`eredg`, `dg`, `stress`, `mute`) and reports which,
if any, is `worth_using`. Run this first on a new dataset. On the bundled
synthetic demo the honest answer is *none of them* - the model's own confidence
is the better flag there - and the tool says so rather than flattering itself.
Select a descriptor on a third split before reporting a number as final.

```bash
pypasi demo --triage --review-rate 0.12 --out results/demo
```

## Figures

```python
from pypasi import viz

viz.plot_bands(clf.bands_, X, y)                     # signal with the partition
viz.plot_band_conflict(audit)                        # conflict per band
viz.plot_importance_vs_conflict(audit, imp, perfeat) # the two questions together
viz.plot_stress_map(audit.result, clf.bands_, i)     # one signal, band x iteration
viz.plot_regime_comparison(table)                    # one panel per metric
viz.plot_dg_curve(frame)                             # effort under corruption
viz.plot_triage_map(report)                          # novelty x conflict, by action
viz.plot_risk_coverage(evidence["risk_coverage"])    # accuracy as cases are handed off
viz.plot_control_chart(report)                       # one batch against its limits
viz.plot_control_trend(frame)                        # drift score across batches
```

Every function takes `dark=True` and returns the figure. The palette is
colour-vision-deficiency validated: the categorical hues clear the all-pairs CVD
and normal-vision separation floors in both modes, signed quantities get a
diverging scale with a neutral zero, magnitudes get a single hue, and every
multi-series figure carries direct labels so identity never rests on colour
alone. Requires the `viz` extra.

## Command line

```bash
pypasi demo --out results/demo
pypasi run --x X.npy --y y.npy --axis axis.npy --out results/mine
pypasi bacteria --data-dir path/to/bacteria-ID --out results/bacteria
```

Common options: `--n-bands`, `--band-method {equal_width,peak_informed}`,
`--band-range LO HI`, `--topology {chain,complete,knn}`,
`--regime {plain,H1,H2}`, `--max-iter`, `--n-reports`, `--seed`,
`--triage --review-rate R --conflict-score {eredg,dg,stress,mute}`, and
`--monitor --monitor-batches N --laser-nm NM`.

`--monitor` runs the batch workflow end to end: it refits the model on part of
the training cohort so the rest can serve as a reference it has never seen,
learns the limits there, charts the evaluation cohort against them without
touching its labels, and writes the limits, the check, a reusable
`*_control_profile.json` and the chart.

## Data

`pypasi.datasets.make_conflict_signals` plants informative peaks and
*contradictory* ones at known positions and returns the mask of which signals
were contaminated — ground truth an audit should recover, which is what this
package's own tests check against.

`pypasi.datasets.load_bacteria` reads the public Raman dataset of Ho et al.
(2019) from a local directory. Nothing is downloaded automatically; get it from
<https://github.com/csho33/bacteria-ID>.

## Tests and conformance

```bash
pip install -e ".[dev,viz]"
pytest
```

`BandNegotiationClassifier`, `PLSDA` and `PCALDA` are checked against
scikit-learn's own `check_estimator` suite in the tests. The two chemometric
classifiers pass every check. The negotiation classifier passes every check when
`temperature=0`; with Metropolis acceptance switched on it deviates only on
`check_methods_subset_invariance` and `check_methods_sample_order_invariance`,
because a stochastic accept/reject step genuinely is not subset-invariant. Both
facts are asserted by tests rather than described.

## Benchmarks

```bash
python benchmarks/bench_vectorisation.py --out bench.csv --figure bench.png
```

The samples in a negotiation are independent, so the obvious implementation is a
loop over signals. `pypasi` instead advances the whole cohort through each
iteration at once. The benchmark measures the difference against a deliberately
naive reference, and refuses to report a single timing until the two agree to
floating point:

| signals | per-sample loop | batched | speedup |
| ---: | ---: | ---: | ---: |
| 100 | 0.79 s | 0.034 s | 23x |
| 500 | 3.99 s | 0.130 s | 31x |
| 1000 | 8.07 s | 0.236 s | 34x |
| 2500 | 20.27 s | 0.641 s | 32x |

Measured on 7 bands, 5 classes, 30 iterations; your numbers will differ, the
shape will not. Cost per signal is roughly flat for the loop and falls for the
batched engine, which is what makes cohort-scale auditing practical.


## Method

The negotiation, the regimes and the Decision Geometry descriptors follow
N. Kourkoumelis, Process-aware inference of biomedical Raman spectra classification, Chemometrics and Intelligent Laboratory Systems 279 (2026) 105914. https://doi.org/10.1016/j.chemolab.2026.105914. This package generalises that method beyond Raman spectroscopy and beyond nearest-neighbour coupling.

## Licence

MIT
