"""Chemometric classifiers for spectroscopic data.

scikit-learn ships the regression half of PLS but no discriminant wrapper, and
PLS-DA is the default classifier in most Raman and infrared workflows. The same
goes for PCA-LDA, which is ubiquitous in biospectroscopy because spectra are
wide and short - thousands of wavenumbers, tens of samples - and LDA cannot be
fitted directly in that regime.

Both classes here are ordinary scikit-learn estimators, so they work as
``base_estimator`` for :class:`~pypasi.estimator.BandNegotiationClassifier` and
anywhere else.

Both also expose ``feature_importances_``, which lets
:meth:`~pypasi.estimator.BandNegotiationClassifier.band_importance` contrast
where a model finds information against where inference runs into conflict -
two different questions that are easily confused.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["PLSDA", "PCALDA", "vip_scores"]


def vip_scores(pls: PLSRegression) -> np.ndarray:
    """Variable Importance in Projection for a fitted :class:`PLSRegression`.

    VIP summarises how much each input variable contributes to the latent
    components that explain the response. The conventional cut-off is 1.0:
    variables above it carry more than their share of the explanatory power.

    Parameters
    ----------
    pls
        A fitted ``PLSRegression``.

    Returns
    -------
    numpy.ndarray
        One score per input feature.

    Notes
    -----
    Defined as

    .. math::

        \\mathrm{VIP}_j = \\sqrt{ p \\frac{\\sum_a w_{ja}^2 \\, \\mathrm{SS}_a}
                                        {\\sum_a \\mathrm{SS}_a} }

    with :math:`\\mathrm{SS}_a` the sum of squares of the response explained by
    component :math:`a`, and :math:`w_{ja}` the normalised loading weights.
    """
    check_is_fitted(pls, "x_weights_")
    t = pls.x_scores_          # (n_samples, n_components)
    w = pls.x_weights_         # (n_features, n_components)
    q = pls.y_loadings_        # (n_targets, n_components)
    n_features = w.shape[0]

    ss = np.diag(t.T @ t @ q.T @ q).reshape(-1)     # per-component explained SS
    total = ss.sum()
    if total <= 0:
        return np.zeros(n_features)
    norm_w2 = w**2 / np.sum(w**2, axis=0, keepdims=True)
    return np.sqrt(n_features * (norm_w2 @ ss) / total)


class PLSDA(ClassifierMixin, TransformerMixin, BaseEstimator):
    """Partial Least Squares Discriminant Analysis.

    Classes are one-hot encoded and regressed on with PLS; the fitted responses
    are the class evidence, and the prediction is their argmax. This is the
    workhorse classifier of vibrational spectroscopy: it copes with far more
    variables than samples and with heavily collinear ones, which is exactly the
    shape of a spectral matrix.

    Parameters
    ----------
    n_components : int
        Number of latent variables. Clamped to what the data can support.
    scale : bool
        Standardise features before fitting. Usually left on for spectra whose
        regions differ in scale, and off after vector normalisation.
    center_scores : bool
        Mean-centre the class scores across classes before returning them from
        ``decision_function``. Softmax is invariant to this, but it keeps the
        scores on a symmetric scale that is easier to read.

    Attributes
    ----------
    classes_ : ndarray
    pls_ : PLSRegression
        The underlying fitted model.
    vip_scores_ : ndarray
        VIP score per feature, see :func:`vip_scores`.
    feature_importances_ : ndarray
        Alias of ``vip_scores_``, for interoperability.
    n_components_ : int
        Components actually used after clamping.

    Examples
    --------
    >>> from pypasi.chemometrics import PLSDA
    >>> from pypasi.datasets import make_conflict_signals
    >>> X, y, _, _ = make_conflict_signals(n_samples=90, random_state=0)
    >>> m = PLSDA(n_components=6).fit(X, y)
    >>> m.vip_scores_.shape
    (400,)

    References
    ----------
    Barker & Rayens (2003), *Partial least squares for discrimination*,
    J. Chemometrics 17:166-173.
    """

    def __init__(self, n_components: int = 10, *, scale: bool = True,
                 center_scores: bool = True):
        self.n_components = n_components
        self.scale = scale
        self.center_scores = center_scores

    def fit(self, X, y):
        X, y = validate_data(self, X, y, accept_sparse=False, dtype="numeric",
                             ensure_min_samples=2, ensure_min_features=1,
                             multi_output=False, y_numeric=False)
        X = np.asarray(X, dtype=float)
        self.classes_ = unique_labels(y)
        if self.classes_.size < 2:
            raise ValueError("PLSDA needs at least two classes")

        # PLS cannot extract more components than the data supports.
        self.n_components_ = int(
            max(1, min(self.n_components, X.shape[0] - 1, X.shape[1]))
        )
        Y = (y[:, None] == self.classes_[None, :]).astype(float)

        self.scaler_ = StandardScaler().fit(X) if self.scale else None
        Xs = self.scaler_.transform(X) if self.scale else X

        self.pls_ = PLSRegression(n_components=self.n_components_, scale=False)
        self.pls_.fit(Xs, Y)
        self.vip_scores_ = vip_scores(self.pls_)
        self.feature_importances_ = self.vip_scores_
        return self

    def _scores(self, X):
        """Per-class PLS response, always ``(n_samples, n_classes)``."""
        check_is_fitted(self, "pls_")
        X = validate_data(self, X, reset=False, accept_sparse=False, dtype="numeric")
        Xs = self.scaler_.transform(X) if self.scaler_ is not None else X
        scores = np.asarray(self.pls_.predict(Xs), dtype=float)
        if scores.ndim == 1:
            scores = scores[:, None]
        if self.center_scores:
            scores = scores - scores.mean(axis=1, keepdims=True)
        return scores

    def decision_function(self, X):
        """Class evidence.

        Two classes give a single margin and more give one score per class,
        following the scikit-learn convention. :func:`pypasi.logits.to_logits`
        converts the binary form back to a proper two-class logit vector.
        """
        scores = self._scores(X)
        if self.classes_.size == 2:
            return scores[:, 1] - scores[:, 0]
        return scores

    def predict_proba(self, X):
        """Softmax of the class scores.

        PLS responses are regression outputs, not probabilities. The softmax
        gives a usable ranking and a proper simplex, but it is not calibrated;
        treat it as relative evidence rather than a likelihood.
        """
        from .divergence import softmax

        return softmax(self._scores(X))

    def predict(self, X):
        check_is_fitted(self, "pls_")
        return self.classes_[np.argmax(self._scores(X), axis=1)]

    def transform(self, X):
        """Project onto the latent variables."""
        check_is_fitted(self, "pls_")
        X = validate_data(self, X, reset=False, accept_sparse=False, dtype="numeric")
        Xs = self.scaler_.transform(X) if self.scaler_ is not None else X
        return self.pls_.transform(Xs)

    def __sklearn_tags__(self):  # pragma: no cover - sklearn plumbing
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags


class PCALDA(ClassifierMixin, TransformerMixin, BaseEstimator):
    """PCA for dimensionality reduction, then Linear Discriminant Analysis.

    The standard route to LDA on spectra, where the number of wavenumbers
    usually exceeds the number of samples and LDA cannot be fitted directly.
    PCA first compresses to a rank the data can support.

    Parameters
    ----------
    n_components : int
        PCA components retained. Clamped to what the data can support.
    scale : bool
        Standardise before PCA.
    shrinkage : {None, "auto"} or float
        Passed to LDA. ``"auto"`` applies Ledoit-Wolf shrinkage, which is worth
        trying when classes are small.
    svd_solver : str
        Passed to :class:`~sklearn.decomposition.PCA`. The default ``"full"``
        computes the exact decomposition, so a refitted model reproduces its
        predecessor exactly. scikit-learn's ``"auto"`` selects the randomized
        solver at spectroscopic shapes and seeds it from the global NumPy
        stream, which makes the fit depend on execution order.
    random_state : int, Generator or None
        Seed for ``svd_solver="randomized"``. Ignored by the default solver.

    Attributes
    ----------
    classes_ : ndarray
    pca_ : PCA
    lda_ : LinearDiscriminantAnalysis
    feature_importances_ : ndarray
        Absolute discriminant loading per original feature, obtained by mapping
        the LDA coefficients back through the PCA rotation.
    """

    def __init__(self, n_components: int = 20, *, scale: bool = True, shrinkage=None,
                 svd_solver: str = "full", random_state=None):
        self.n_components = n_components
        self.scale = scale
        self.shrinkage = shrinkage
        self.svd_solver = svd_solver
        self.random_state = random_state

    def fit(self, X, y):
        X, y = validate_data(self, X, y, accept_sparse=False, dtype="numeric",
                             ensure_min_samples=2, ensure_min_features=1,
                             multi_output=False, y_numeric=False)
        X = np.asarray(X, dtype=float)
        self.classes_ = unique_labels(y)
        if self.classes_.size < 2:
            raise ValueError("PCALDA needs at least two classes")

        self.n_components_ = int(
            max(1, min(self.n_components, X.shape[0] - 1, X.shape[1]))
        )
        self.scaler_ = StandardScaler().fit(X) if self.scale else None
        Xs = self.scaler_.transform(X) if self.scale else X

        self.pca_ = PCA(n_components=self.n_components_,
                        svd_solver=self.svd_solver,
                        random_state=self.random_state).fit(Xs)
        solver = "lsqr" if self.shrinkage is not None else "svd"
        self.lda_ = LinearDiscriminantAnalysis(
            solver=solver, shrinkage=self.shrinkage
        ).fit(self.pca_.transform(Xs), y)

        coef = np.atleast_2d(self.lda_.coef_)          # (n_classes, n_components)
        self.feature_importances_ = np.abs(coef @ self.pca_.components_).max(axis=0)
        return self

    def _project(self, X):
        check_is_fitted(self, "lda_")
        X = validate_data(self, X, reset=False, accept_sparse=False, dtype="numeric")
        Xs = self.scaler_.transform(X) if self.scaler_ is not None else X
        return self.pca_.transform(Xs)

    def decision_function(self, X):
        check_is_fitted(self, "lda_")
        return self.lda_.decision_function(self._project(X))

    def predict_proba(self, X):
        check_is_fitted(self, "lda_")
        return self.lda_.predict_proba(self._project(X))

    def predict(self, X):
        check_is_fitted(self, "lda_")
        return self.lda_.predict(self._project(X))

    def transform(self, X):
        """Project onto the LDA discriminant axes."""
        check_is_fitted(self, "lda_")
        return self.lda_.transform(self._project(X))

    def __sklearn_tags__(self):  # pragma: no cover - sklearn plumbing
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags
