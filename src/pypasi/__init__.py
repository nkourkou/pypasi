"""pypasi - process-aware signal inference.

Classify 1-D signals by negotiated consensus among band-level models, then audit
how the decision was reached: which regions of the signal disagreed, how long
they held out, and what the raw data looks like there.

Quick start
-----------
>>> from pypasi import BandNegotiationClassifier
>>> from pypasi.datasets import make_conflict_signals
>>> X, y, axis, _ = make_conflict_signals(n_samples=200, random_state=0)
>>> clf = BandNegotiationClassifier(axis=axis, n_bands=7, regime="H1", random_state=0)
>>> clf.fit(X, y)                                    # doctest: +ELLIPSIS
BandNegotiationClassifier(...)
>>> audit = clf.audit(X, y)
>>> audit.top_conflict_bands(1)                      # doctest: +SKIP
['B4']
>>> trace = audit.trace_band("B4")                   # raw signal behind the conflict

Monitoring a new batch, without labels
--------------------------------------
>>> from pypasi import ControlProfile
>>> profile = ControlProfile.fit(clf, X, n_batches=10)   # doctest: +SKIP
>>> print(profile.check(X_new).summary())                # doctest: +SKIP
"""

from . import degrade, quality
from .audit import AuditResult, BandTrace, band_importance, compare_regimes
from .bands import Band, BandSet
from .chemometrics import PCALDA, PLSDA, vip_scores
from .divergence import js_divergence, kl_divergence, softmax
from .estimator import BandNegotiationClassifier
from .evidence import (
    EVIDENCE_KINDS,
    EvidenceSpec,
    TemperatureCalibrator,
    UncalibratedEvidenceWarning,
    to_evidence,
)
from .external import ExternalAudit, audit_external
from .gates import AbsoluteGate, Gate, QuantileGate, RankGate
from .geometry import (
    band_decision_geometry,
    decision_geometry,
    early_decision_geometry,
    eredg,
    redg,
)
from .logits import to_logits
from .monitor import RESIDUAL_STATISTICS, ControlProfile, ControlReport
from .negotiate import NegotiationResult, negotiate
from .perturb import perturb_bands
from .quality import noise_level, noise_limit
from .regimes import H1, H2, Plain, Regime
from .topology import Topology
from .triage import (
    NoveltyDetector,
    Triage,
    WeakOrientationWarning,
    compare_conflict_scores,
    risk_coverage,
)

__version__ = "0.5.5"

__all__ = [
    "EVIDENCE_KINDS",
    "H1",
    "H2",
    "PCALDA",
    "PLSDA",
    "RESIDUAL_STATISTICS",
    "AbsoluteGate",
    "AuditResult",
    "Band",
    "BandNegotiationClassifier",
    "BandSet",
    "BandTrace",
    "ControlProfile",
    "ControlReport",
    "EvidenceSpec",
    "ExternalAudit",
    "Gate",
    "NegotiationResult",
    "NoveltyDetector",
    "Plain",
    "QuantileGate",
    "RankGate",
    "Regime",
    "TemperatureCalibrator",
    "Topology",
    "Triage",
    "UncalibratedEvidenceWarning",
    "WeakOrientationWarning",
    "__version__",
    "audit_external",
    "band_decision_geometry",
    "band_importance",
    "compare_conflict_scores",
    "compare_regimes",
    "decision_geometry",
    "degrade",
    "early_decision_geometry",
    "eredg",
    "js_divergence",
    "kl_divergence",
    "negotiate",
    "noise_level",
    "noise_limit",
    "perturb_bands",
    "quality",
    "redg",
    "risk_coverage",
    "softmax",
    "to_evidence",
    "to_logits",
    "vip_scores",
]
