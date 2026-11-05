# Stress-testing the audit on your own data

`pypasi.degrade` plants known faults in spectra, so that you can measure what the audit detects on *your* instrument and samples before relying on it.

## Recipe

You need four disjoint sets of spectra, split by specimen (patient, isolate, batch), all **before** preprocessing: one to fit the model, one clean set to calibrate the audit, one to plant defects in so the audit learns which direction of conflict means trouble, and one held out for the test.

```python
import numpy as np
from pypasi import BandNegotiationClassifier, BandSet, PLSDA, Triage, degrade

def snv(X):                                   # the scaling your model expects
    X = np.asarray(X, float)
    return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)

bands = BandSet.equal_width(axis, 7)
clf = BandNegotiationClassifier(bands=bands, base_estimator=PLSDA(n_components=10),
                                topology="complete").fit(snv(X_fit), y_fit)

oriented = degrade.plant(X_orient, axis, bands=bands, random_state=1)
triage = Triage().fit(clf, snv(X_cal), y_cal,
                      orientation_data=(snv(oriented.X), oriented.degraded))

table = degrade.stress_test(triage, X_test, y_test, axis, preprocess=snv, random_state=0)
```

- Plant **before** preprocessing: a defect is something the instrument did, and scaling is what the analyst does afterwards.
- `table` has one row per family, plus `"all"`. It gives the AUROC of conflict, 1 − confidence and novelty against the clean spectra, and the paired accuracy cost (the same spectra before and after the fault).
- If `Triage.fit` emits a `WeakOrientationWarning`, the planted defects barely moved conflict on your data, so the direction of the score is uncertain; plant more spectra or check that every band classifies above chance (`clf.band_accuracy`).

To plant faults without running the audit, use `degrade.plant(X, axis, bands=bands, random_state=0)`. It returns the spectra and the ground truth: `degraded`, `kind`, `severity`, and `where` (the interval of a local fault).

## The families

Severity `s` is drawn uniformly from `severity` (default 0.3–1.0). The magnitudes assume spectra scaled to about [0, 1] per spectrum, so translate them to other units through the definitions below.

| Family | Scope | What `s` does | In the paper |
|---|---|---|---|
| `axis shift` | global | shifts the axis by 4 `s` axis units | yes |
| `resolution loss` | global | applies a Gaussian line shape, FWHM 7 `s` | yes |
| `throughput dip` | global | applies the envelope exp(−0.35 `s` g), with g a Gaussian at mid-axis, sd 22% of the span | yes |
| `short exposure` | global | adds noise of sd 0.02 `s` √(x + 0.05) | yes |
| `local contamination` | local | attenuates one interval by exp(−0.45 `s`) and adds noise of sd 0.02 `s` | yes |
| `cosmic rays` | global | adds 1 + Poisson(3 `s`) one-channel spikes | no |
| `fluorescence` | global | adds a decaying background of 0.5 `s` × the spectrum's range | no |
| `intensity loss` | global | scales the whole spectrum by exp(−0.7 `s`) | no |

Each family is a small function, and the transforms (`axis_shift`, `resolution_loss`, …) can be called on their own. Choose families with `families=[...]`.

**Add your own fault** using this function:

```python
def etaloning(x, axis, s, rng, edges):
    return x * (1 + 0.05 * s * np.sin(axis / 12.0)), -1      # (new spectrum, interval or -1)

fam = degrade.Family("etaloning", "global", etaloning, "5 s % fringes")
degrade.stress_test(triage, X, y, axis, families=[fam, "local contamination"], preprocess=snv)
```

## What the audit can see

Measured on the Ho et al. cohort (split 0):

| Family | Conflict | 1 − confidence | Novelty | Accuracy cost (pp) |
|---|---|---|---|---|
| local contamination | **0.924** | 0.449 | 0.562 | 13.5 |
| throughput dip | **0.908** | 0.460 | 0.597 | 6.2 |
| resolution loss | **0.804** | 0.363 | 0.168 | 1.4 |
| axis shift | **0.757** | 0.497 | 0.414 | 1.0 |
| short exposure | 0.485 | 0.506 | 0.500 | 0.5 |
| fluorescence | **0.996** | 0.390 | 0.665 | 28.1 |
| cosmic rays | 0.503 | 0.514 | 0.535 | 0.6 |
| intensity loss | 0.498 | 0.494 | 0.499 | 0.0 |

**Conflict follows the damage.** It is high for faults that change the shape of the spectrum unevenly across bands and cost accuracy. It is at chance for three kinds of fault:

- **Faults that the preprocessing removes.** Intensity loss is undone exactly by SNV.
- **Faults that do not change the prediction.** Cosmic rays at this size (standard despiking removes them), and the paper's short exposure.
- **Faults that act identically on every channel.** Extra white noise is the main example.

For the last group, use an acquisition-level check. `pypasi.noise_level` estimates the noise of each spectrum from its Savitzky–Golay residual, and `pypasi.noise_limit` learns a limit from clean spectra. Apply both to raw counts or min–max-scaled spectra, not after SNV. Integration time, laser power and total counts, where the instrument records them, are better still.

On the Ho et al. data, the planted short-exposure noise (sd 0.004–0.014 of the min–max range) is an order of magnitude below the noise already in the released spectra (about 0.10). It flips 0.5% of predictions, and no score detects it, not even `noise_level` (0.51). At eight times that level (accuracy cost 10.6 pp), `noise_level` reaches 0.83, novelty 0.66 and conflict 0.56.
