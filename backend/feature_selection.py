from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.feature_selection import chi2, mutual_info_classif


@dataclass
class HybridFeatureSelector:
    """Combine Chi-Square and Mutual Information scores for feature ranking."""

    k_features: int = 2000
    chi2_weight: float = 0.5
    mi_weight: float = 0.5

    def fit(self, x_matrix, y_labels) -> "HybridFeatureSelector":
        chi2_scores, _ = chi2(x_matrix, y_labels)
        mi_scores = mutual_info_classif(x_matrix, y_labels, discrete_features=True)
        chi2_scores = self._normalize(chi2_scores)
        mi_scores = self._normalize(mi_scores)
        self.combined_scores_ = (
            self.chi2_weight * chi2_scores + self.mi_weight * mi_scores
        )
        self.selected_indices_ = np.argsort(self.combined_scores_)[::-1][
            : self.k_features
        ]
        return self

    def transform(self, x_matrix) -> np.ndarray:
        return x_matrix[:, self.selected_indices_]

    def fit_transform(self, x_matrix, y_labels) -> np.ndarray:
        self.fit(x_matrix, y_labels)
        return self.transform(x_matrix)

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        min_value = np.min(scores)
        max_value = np.max(scores)
        if max_value == min_value:
            return np.zeros_like(scores)
        return (scores - min_value) / (max_value - min_value)

    def summary(self) -> Tuple[int, float, float]:
        return (self.k_features, self.chi2_weight, self.mi_weight)
