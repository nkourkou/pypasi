# Parameter selection guide

The settings that change what the pypasi audit can see are the **spectral scaling**, the **band estimator**, the **number, width and placement of bands**, the **interaction topology** and the **regime**. This guide gives a default for each and says when to depart from it. Section 2.3 of the SoftwareX paper summarises it in one paragraph.

Every number below comes from the Ho et al. bacteria cohort used in that paper:

- 1,000 channels over 382–1792 cm⁻¹, 1.41 cm⁻¹ per channel, 400–1792 cm⁻¹ used;
- PLS-DA(10) band models, evidence calibrated on held-out isolates;
- defects planted on held-out isolates.

The figures are the mean ± sd over five patient-disjoint splits (`SPLIT_REPEAT = 0–4`). Where a comparison is called consistent, it went the same way in all five splits. The figures are produced, and checked against this page, by `benchmarks/parameter_guide.ipynb`.

**Scope.** The defaults suit biomedical spectra in the fingerprint region, where the signatures of proteins, lipids, nucleic acids and carbohydrates spread class information across the whole range, so every band has something to say. Spectra whose information sits in a few sharp peaks (pharmaceutical, polymer or mineral identification) may leave bands without class information, and there the defaults must be checked (§5). Read the numbers as guidance for spectra like these, not as constants.

## Summary

| Setting | Default | Change it when | Evidence (this cohort) |
|---|---|---|---|
| Scaling | SNV (per-spectrum standardisation) before the band models | Never use min–max scaling alone | conflict 0.76 against 0.51 (§1) |
| Band estimator | PLS-DA, 10 components | Keep any classifier for prediction and audit it with `ExternalAudit` | PLS-DA gave the strongest conflict: 0.777 against 0.706 for linear SVM (paper, §3.4) |
| Number and width of bands | About 7 equal-width bands (199 cm⁻¹ here) | Narrower bands (down to about 70 cm⁻¹) to localise narrow changes, at a cost of 2–3 accuracy points; never fewer than 5 | localisation hit 0.65 at 7 bands, 0.82 at 10–14; 3 bands near chance (§2) |
| Band content | Every band classifies above chance on its own (`band_accuracy`) | Merge or drop bands that are at chance | each band 0.52–0.86 against a chance rate of 0.20 (§2) |
| Points per band | At least about 20 | On coarser instruments, keep the band width in cm⁻¹; if a band then holds fewer than about 20 points, use fewer bands | local defect 0.92 → 0.87 at 8× coarser sampling (§2) |
| Band placement | Equal width | Peak-informed when a strong peak falls on a boundary | no gain in detection: 0.753 against 0.777 (§3) |
| Topology | `"complete"` | `"knn"` above about 14 bands, to cut the cost; avoid `"chain"` | chain 0.72 against 0.76; audit time 5.1 s against 1.3 s per 1,000 spectra at 28 bands (§4) |
| Regime | `"H1"` (keeps the overruled band's evidence) | Rarely; plain, H1 and H2 detect equally | 0.772–0.778 (§4) |
| Before use | `degrade.stress_test` on held-out spectra of your own | – | §5 |

## 1. Scaling decides whether the method works

Inter-band conflict compares the class evidence of bands. A per-spectrum min–max scaling ties every band to the two most extreme channels of the spectrum, so a defect in one band rescales all the others, and the bands no longer disagree in a useful way.

| 7 bands, complete | Conflict vs degraded acquisition | External accuracy |
|---|---|---|
| SNV | **0.760 ± 0.028** | 0.771 ± 0.026 |
| min–max only | 0.510 ± 0.013 | 0.725 ± 0.039 |

SNV was better in 5 of 5 splits, by 0.25. Under min–max scaling the orientation margin also collapsed (0.004), so the learned sign of the descriptor was close to arbitrary. The batch control chart is a different matter: it localised equally well under both scalings (hit 0.66), because it compares batch statistics with reference limits rather than bands with each other. **Apply SNV (or MSC, or an L2 norm followed by centring) before the band models**, and plant defects before the scaling when testing (§5).

## 2. How many bands

Per-spectrum detection has a broad plateau. Localisation improves with more bands, while accuracy and the link between conflict and wrong predictions decline.

| Bands | Width (cm⁻¹) | Channels per band | Conflict vs degradation | Localisation (lift over chance) | Conflict vs wrong prediction | External accuracy |
|---|---|---|---|---|---|---|
| 3 | 464 | 329 | 0.675 ± 0.038 | 1.3× | 0.672 | 0.782 |
| 5 | 279 | 197 | 0.726 ± 0.033 | 1.7× | 0.638 | 0.772 |
| **7** | **199** | **141** | **0.760 ± 0.028** | **4.6×** | 0.583 | 0.771 |
| 10 | 139 | 98 | 0.756 ± 0.036 | 3.6× (hit 0.82) | 0.532 | 0.767 |
| 14 | 100 | 70 | 0.754 ± 0.033 | 5.7× (hit 0.82) | 0.481 | 0.749 |

Here localisation is measured with a defect about 200 cm⁻¹ wide (one of the seven published intervals); defects of other widths are tested below. A hit is a named band that overlaps the damaged interval, and the lift divides the hit rate by the share of bands that overlap it.

- **Every band carries class information.** On held-out isolates, each of the seven band models classified the five species on its own at 0.52–0.86 accuracy (chance 0.20; negotiated ensemble 0.91). That is the precondition for the audit, since a band at chance has nothing to disagree about. Check it on your own data with `band_accuracy` (§5).
- **Below five bands the audit cannot localise.** Three bands localise barely above chance (hit 0.54 against a chance rate of 0.43). They also detect 0.08 worse than seven, consistently.
- **From 7 to 14 bands detection does not change** (differences of 0.003–0.006, no consistent winner). The hit rate rises from 0.65 to 0.82.
- **More bands cost accuracy.** Each band model sees less of the spectrum. External accuracy falls from 0.839 (3 bands) to 0.754 (28 bands) on split 0; full-spectrum PLS-DA reaches 0.842. Conflict also becomes less related to wrong predictions (0.67 → 0.48), because narrow bands disagree for reasons that do not change the answer.

### Band width against the size of the change

The table above uses one defect width, about 200 cm⁻¹, so it cannot show whether the best band width depends on the width of the change. To test this, a local defect 50, 100, 200 or 400 cm⁻¹ wide was planted at a random position: per spectrum on half the qc isolates, and per batch in 42 localisation trials. The results are means over three patient splits.

Per-spectrum detection (conflict AUROC, sd 0.02–0.06):

| Bands (width, cm⁻¹) | Defect 50 cm⁻¹ | 100 | 200 | 400 |
|---|---|---|---|---|
| 5 (279) | 0.639 | 0.767 | 0.876 | 0.951 |
| 7 (199) | 0.638 | 0.763 | 0.882 | 0.962 |
| 10 (139) | 0.630 | 0.749 | 0.859 | 0.943 |
| 14 (100) | 0.628 | 0.737 | 0.849 | 0.936 |
| 20 (70) | 0.620 | 0.727 | 0.843 | 0.930 |

Batch localisation (hit rate, with lift over chance):

| Bands (width, cm⁻¹) | Defect 50 cm⁻¹ | 100 | 200 | 400 |
|---|---|---|---|---|
| 5 (279) | 0.73 (3.1×) | 0.60 (2.3×) | 0.60 (1.7×) | 0.71 (1.5×) |
| 7 (199) | 0.86 (4.9×) | 0.82 (4.0×) | 0.73 (2.6×) | 0.74 (1.7×) |
| 10 (139) | 0.89 (6.2×) | 0.83 (4.8×) | 0.83 (3.3×) | **0.77** (1.9×) |
| 14 (100) | 0.83 (8.0×) | 0.82 (5.7×) | 0.83 (3.9×) | 0.64 (1.8×) |
| 20 (70) | **0.95** (10.8×) | **0.97** (8.1×) | **0.90** (4.7×) | 0.56 (1.7×) |

- **Detection does not depend on matching the band width to the defect.** For every defect width, 5–7 bands detect a local defect best and 20 bands worst, by 0.02–0.03, which is within the split-to-split spread. How visible a defect is depends on its own size (0.64 at 50 cm⁻¹, 0.96 at 400 cm⁻¹), not on the partition.
- **Localisation is sharper with narrower bands**, down to the narrowest tested (70 cm⁻¹), for any defect up to about 200 cm⁻¹. The exception is a change much wider than the bands: a 400 cm⁻¹ defect spread over six 70 cm⁻¹ bands was named correctly in only 0.56 of trials, and 7–10 bands did best.
- **The rule is therefore a trade-off, not a matching rule.** Seven bands gave the best per-spectrum detection overall: five bands matched them on local defects but were 0.03 worse across all five defect families, in 5 of 5 splits. Narrower bands, down to about 70 cm⁻¹ while keeping at least about 20 channels per band, name a narrow change more precisely and cost 2–3 points of external accuracy. Only a change broader than three or four bands is localised worse by narrow bands.

### Coarser spectral sampling

To see how the choice scales with resolution, adjacent channels were averaged after the defects were planted. The defects are optical; the detector samples afterwards.

| Sampling (cm⁻¹ per channel) | Bands | Channels per band | Local contamination | Localisation hit | External accuracy |
|---|---|---|---|---|---|
| 1.4 | 7 | 141 | 0.924 | 0.62 | 0.797 |
| 2.8 | 7 | 70 | 0.907 | 0.57 | 0.799 |
| 5.6 | 7 | 35 | 0.880 | 0.50 | 0.804 |
| 11.3 | 7 | 18 | 0.866 | 0.62 | 0.782 |
| 5.6 | 14 | 17 | 0.808 | 0.74 | 0.806 |
| 11.3 | 14 | 9 | 0.795 | 0.64 | 0.791 |

Split 0 only, PLS-DA(10); the last row uses PLS-DA(5), because 9 channels cannot support 10 components. Reducing to PLS-DA(5) at 7 bands changed nothing material (0.851 against 0.866).

**Keep the band width in cm⁻¹ when the sampling changes.** At a fixed 7 bands, an eight-fold coarser sampling (18 channels per band) still detects local contamination at 0.87. Splitting into more, narrower bands on a coarse axis lowers detection. Around 20 channels per band is a reasonable floor. The detection of global defects falls with coarser sampling (0.777 → 0.651 overall), partly because binning itself hides sub-channel shifts and blur.

## 3. Equal width or peak-informed

`BandSet.peak_informed` places boundaries in the deepest troughs of the mean spectrum, so that strong peaks are not split between two bands. On this cohort it gave a slightly higher in-cohort accuracy (0.918 against 0.877 at 7 bands, split 0). External accuracy was similar (0.800 against 0.797), and detection was no better (0.753 against 0.777). **Equal width is the default** because the result does not depend on the training mean spectrum and the bands are easy to report. Use peak-informed bands when a strong band sits on an equal-width boundary.

Before 0.5.5, `peak_informed` could return fewer bands than requested without a warning. A request for 7 returned 6, and 10 or 14 returned 9. It now places the number requested, and warns if `min_width` makes that impossible.

## 4. Topology and regime

| 7 bands | Conflict vs degradation | Local contamination | Localisation hit |
|---|---|---|---|
| complete | **0.760 ± 0.028** | **0.877** | **0.65** |
| knn (k = 2) | 0.756 ± 0.028 | 0.867 | 0.51 |
| chain | 0.715 ± 0.040 | 0.759 | 0.55 |

- **Chain is worst**, consistently: 0.045 lower at 7 bands and 0.025 at 10, in 5 of 5 splits. In a chain a damaged band is compared only with its two neighbours, so a local defect has less to disagree with. (Chain is the topology of the original method [Kourkoumelis 2026].)
- **Complete and kNN detect equally**, but kNN localises worse at 7 bands. The complete graph costs more as bands are added: auditing 1,000 spectra on two cores took 0.27 s (complete) and 0.23 s (kNN) at 7 bands, 0.95 s and 0.56 s at 14, and 5.1 s and 1.3 s at 28 (`benchmarks/bench_runtime.ipynb`). Use `"knn"` beyond about 14 bands.
- **The regime hardly matters for detection.** At 7 bands, complete: plain 0.778, H1 0.777, H2 0.772; localisation 0.69 / 0.62 / 0.71. With 42 trials the hit rate has a standard error of about 0.075, so these localisation differences are within noise. H1 is the default because it records the overruled band's evidence (`minority_evidence`).

## 5. Check it on your own data

The numbers above belong to one cohort. Before relying on a configuration:

1. **Check that every band carries class information**: `clf.band_accuracy(X_heldout, y_heldout)` scores each band's own model against the majority-class rate. On this cohort every band passed (0.52–0.86 against 0.20). Merge or drop bands that do not.
2. **Split by specimen or patient**, never by spectrum. A spectrum-level split was worth 7.1 accuracy points of leakage on this cohort.
3. **Plant defects with `pypasi.degrade`** on a held-out set, before your scaling. Learn the conflict sign on one held-out group (`Triage.fit(..., orientation_data=...)`) and read AUROCs on another.
4. **Compare two or three configurations, not thirty**, and change the default only if a competitor wins by a margin you set in advance. The analysis behind the paper used 0.05; no configuration above cleared it.
5. **Read localisation with its uncertainty.** Forty-two trials give a 95% interval of about ±0.15 on the hit rate.
