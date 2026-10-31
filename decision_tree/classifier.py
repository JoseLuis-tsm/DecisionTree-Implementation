"""Classification tree skeleton."""
from .base import BaseDecisionTree

import numpy as np  # conteos de clases, logaritmos, promedios


class DecisionTreeClassifier(BaseDecisionTree):
    """Tree variant for classification.

    Parameters:
        max_depth (int or None): See ``BaseDecisionTree``.
        min_samples_split (int): See ``BaseDecisionTree``.
        criterion (str): Supported values: 'gini', 'entropy'.

    Each hook below documents what it receives and must return; see
    ``BaseDecisionTree`` for the full contract.
    """

    # Criterios que acepta este árbol
    _criteria = ("gini", "entropy")

    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        super().__init__(max_depth, min_samples_split, criterion)

    def _build_tree(self, X, y, depth):
        """Recursively grow the tree.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at this node.
            y (ndarray, shape (n_samples,)): Targets at this node.
            depth (int): Current depth (root is 0).

        Returns:
            Node: Leaf (when a stopping rule triggers) or internal node with
            two subtrees built from the best split.
        """
        # Misma lógica que regresión: se reutiliza la de BaseDecisionTree
        return super()._build_tree(X, y, depth)

    def _impurity(self, y):
        """Impurity of ``y`` per ``self.criterion``: Gini impurity 1 - sum(p_k^2) or entropy -sum(p_k * log2(p_k)), where p_k is the fraction of samples of class k.

        Args:
            y (ndarray, shape (n_samples,)): Targets of a node.

        Returns:
            float: Impurity (0 if pure).
        """
        _, counts = np.unique(y, return_counts=True)  # n_k: muestras de cada clase
        p = counts / len(y)                           # p_k = n_k / n

        if self.criterion == "gini":
            return float(1 - np.sum(p ** 2))           # 1 - sum(p_k^2)
        if self.criterion == "entropy":
            return float(-np.sum(p * np.log2(p)))      # -sum(p_k log2 p_k), p_k > 0 siempre
        raise ValueError(f"Unsupported criterion: {self.criterion}")

    def _best_split(self, X, y):
        """Search all features/thresholds for the largest impurity decrease.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at the node.
            y (ndarray, shape (n_samples,)): Targets at the node.

        Returns:
            tuple (int, float) or None: ``(feature_index, threshold)``, or
            None if no split reduces impurity.
        """
        # Misma búsqueda que regresión solo cambia _impurity, que se llama desde ahí
        return super()._best_split(X, y)

    def _leaf_value(self, y):
        """Return the majority (most frequent) class of y.

        Args:
            y (ndarray, shape (n_samples,)): Targets that reached the leaf.

        Returns:
            The prediction for the leaf.
        """
        classes, counts = np.unique(y, return_counts=True)  # clases y cuátas veces aparece cada una
        return classes[np.argmax(counts)]                   # la más frecuente (empate: la primera)

    def _score(self, y_true, y_pred):
        """Metric for ``eval``: accuracy = fraction of correct predictions.

        Args:
            y_true (ndarray, shape (n_samples,)): True targets.
            y_pred (ndarray, shape (n_samples,)): Predicted targets.

        Returns:
            float: Score (higher is better).
        """
        return float(np.mean(y_true == y_pred))  # promedio de aciertos (True = 1, False = 0)
