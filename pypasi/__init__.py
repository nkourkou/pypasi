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
from .monitor import ControlProfile, ControlReport
from .negotiate import NegotiationResult, negotiate
from .perturb import perturb_bands
from .regimes import H1, H2, Plain, Regime
from .topology import Topology
from .triage import NoveltyDetector, Triage, compare_conflict_scores, risk_coverage

__version__ = "0.5.0"

__all__ = [
    "__version__",
    "AbsoluteGate",
    "audit_external",
    "AuditResult",
    "Band",
    "band_decision_geometry",
    "band_importance",
    "BandNegotiationClassifier",
    "BandSet",
    "BandTrace",
    "compare_conflict_scores",
    "compare_regimes",
    "ControlProfile",
    "ControlReport",
    "decision_geometry",
    "early_decision_geometry",
    "eredg",
    "EVIDENCE_KINDS",
    "EvidenceSpec",
    "ExternalAudit",
    "Gate",
    "H1",
    "H2",
    "js_divergence",
    "kl_divergence",
    "negotiate",
    "NegotiationResult",
    "NoveltyDetector",
    "PCALDA",
    "perturb_bands",
    "Plain",
    "PLSDA",
    "QuantileGate",
    "RankGate",
    "redg",
    "Regime",
    "risk_coverage",
    "softmax",
    "TemperatureCalibrator",
    "to_evidence",
    "to_logits",
    "Topology",
    "Triage",
    "UncalibratedEvidenceWarning",
    "vip_scores",
]
