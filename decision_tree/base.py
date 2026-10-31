"""Base class for decision trees with a scikit-learn like API."""
from abc import ABC, abstractmethod

import numpy as np  # arreglos y operaciones vectoriales

from .node import Node  # estructura de cada nodo del árbol


class BaseDecisionTree(ABC):
    """Skeleton shared by every tree variant (classification, regression, ...).

    Hyperparameters are set in ``__init__``; ``fit``, ``predict`` and ``eval``
    follow the scikit-learn conventions. Subclasses implement the abstract
    hooks; the TODOs are left to the students.

    Parameters:
        max_depth (int or None): Maximum depth of the tree. The root has depth
            0. ``None`` means nodes grow until they are pure or smaller than
            ``min_samples_split``.
        min_samples_split (int): Minimum number of samples a node needs to be
            considered for splitting. Must be >= 2.
        criterion (str): Name of the impurity measure used to choose splits
            (e.g. 'gini', 'entropy', 'mse'). Valid values depend on the
            subclass.

    Attributes:
        root (Node or None): Root of the fitted tree; ``None`` before ``fit``.
    """

    # Criterios válidos; cada subclase pone los suyos
    _criteria = ()

    def __init__(self, max_depth=None, min_samples_split=2, criterion=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        # Hiperparámetros extra (Fase 4); se cambian con set_params
        self.min_samples_leaf = 1           # mínimo de muestras en cada hijo
        self.max_features = None            # features a probar por split (None = todas)
        self.min_impurity_decrease = 0.0    # ganancia mínima para aceptar un split
        self.random_state = None            # semilla para elegir features al azar
        # Se llenan en fit
        self.n_features_ = None             # número de columnas de X
        self.feature_types_ = None          # "numeric" o "categorical" por columna
        self._rng = None                    # generador de números aleatorios

    def get_params(self):
        """Return the hyperparameters.

        Returns:
            dict: Maps 'max_depth', 'min_samples_split' and 'criterion' to
            their current values. Add new hyperparameters here.
        """
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "criterion": self.criterion,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "min_impurity_decrease": self.min_impurity_decrease,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """Update hyperparameters by keyword, e.g. ``set_params(max_depth=5)``.

        Args:
            **params: Hyperparameter names and new values.

        Returns:
            BaseDecisionTree: ``self``, to allow chaining.

        Raises:
            ValueError: If a name is not a known hyperparameter.
        """
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        """Build the tree from training data.

        TODO (students): implement this method. It should
          1. validate the hyperparameters (e.g. ``max_depth`` is None or a
             positive int, ``min_samples_split >= 2``, ``criterion`` is
             supported) and raise ``ValueError`` otherwise;
          2. convert ``X`` and ``y`` to arrays and check that they have the
             same number of samples;
          3. call ``self._build_tree(X, y, depth=0)`` and store the returned
             node in ``self.root``;
          4. return ``self``.

        Args:
            X (array-like, shape (n_samples, n_features)): Training features.
            y (array-like, shape (n_samples,)): Targets (class labels for
                classification, real values for regression).

        Returns:
            BaseDecisionTree: ``self`` (the fitted estimator).
        """
        # 1. Validar hiperparámetros
        self._validate_params()

        # 2. Convertir a arreglos y revisar formas
        X = np.asarray(X, dtype=float)  # features como matriz de floats
        y = np.asarray(y)               # etiquetas tal cual (pueden ser texto)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be 1D (n_samples,)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if X.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample")

        # Guardar información que usa la búsqueda de spilts
        self.n_features_ = X.shape[1]                         # columnas vistas en fit
        self.feature_types_ = self._detect_feature_types(X)   # numérica o categórica
        self._rng = np.random.default_rng(self.random_state)  # azar reproducible

        # 3. Construir el árbol desde la raíz (profundidad 0)
        self.root = self._build_tree(X, y, depth=0)

        # 4. Regresar el modelo entrenado
        return self

    def predict(self, X):
        """Predict a target for each row of ``X``.

        TODO (students): implement this method. It should raise an error if
        the model has not been fitted (``self.root is None``) and, for each
        sample, walk from ``self.root``: go left when
        ``x[feature_index] <= threshold`` else right, until a leaf is
        reached, then return that leaf's ``value``.

        Args:
            X (array-like, shape (n_samples, n_features)): Samples to predict.

        Returns:
            array, shape (n_samples,): Predicted target for every sample.
        """
        # No se puede predecir sin entrenar
        if self.root is None:
            raise RuntimeError("Model is not fitted yet. Call fit first.")

        X = np.asarray(X, dtype=float)  # convertir a matriz de floats
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X must have {self.n_features_} features")

        # Recorre cada fila y guarda su predicción
        predictions = []
        for x in X:
            predictions.append(self._predict_one(x))  # bajar por el árbol
        return np.array(predictions)

    def eval(self, X, y):
        """Score the model on ``(X, y)``.

        TODO (students): implement this method. It should call
        ``self.predict(X)`` and return ``self._score(y, y_pred)``.

        Args:
            X (array-like, shape (n_samples, n_features)): Samples to evaluate.
            y (array-like, shape (n_samples,)): True targets for ``X``.

        Returns:
            float: Metric value (accuracy for classification, R^2 for
            regression; higher is better).
        """
        y_pred = self.predict(X)  # predicciones del modelo
        y = np.asarray(y)         # Etiquetas reales como arreglo
        if y.shape[0] != y_pred.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        return self._score(y, y_pred)  # accuracy o R cuadarda según la subclase

    @abstractmethod
    def _build_tree(self, X, y, depth):
        """Recursively grow the tree and return its root ``Node``.

        TODO (students): create a leaf (``Node(value=self._leaf_value(y))``)
        when a stopping condition holds: ``depth`` reached ``max_depth``,
        fewer than ``min_samples_split`` samples, ``y`` is pure, or
        ``_best_split`` finds no useful split. Otherwise split the data on the
        returned (feature, threshold) and recurse on both parts with
        ``depth + 1``.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at this node.
            y (ndarray, shape (n_samples,)): Targets at this node.
            depth (int): Depth of this node (root is 0).

        Returns:
            Node: Subtree built from this data.
        """
        # Lógica común a clasificación y regresión; las subclases la llaman con super()
        n_samples = len(y)  # muestras que llegaron a este nodo

        # 1. Condiciones de paro
        reached_depth = self.max_depth is not None and depth >= self.max_depth  # ya no puede crecer
        too_few = n_samples < self.min_samples_split                            # muy pocas muestras
        is_pure = len(np.unique(y)) == 1                                        # un solo valor en y
        if reached_depth or too_few or is_pure:
            return Node(value=self._leaf_value(y))

        # 2. Buscar el mejor split
        split = self._best_split(X, y)
        if split is None:  # ningún split útil
            return Node(value=self._leaf_value(y))
        feature_index, threshold = split

        # 3. Partir los datos: izquierda x <= t, derecha x > t
        left_mask = X[:, feature_index] <= threshold

        # 4. Construir cada hijo un nivel más abajo
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        # Nodo interno con su regla, sus hijos y la predicción de este nodo
        return Node(feature_index=feature_index, threshold=threshold,
                    left=left, right=right, value=self._leaf_value(y))

    @abstractmethod
    def _impurity(self, y):
        """Compute the impurity of a set of targets.

        TODO (students): implement the measure selected by ``self.criterion``
        (e.g. Gini or entropy for classification, MSE or MAE for regression).
        Lower means more homogeneous; it is used to score candidate splits.

        Args:
            y (ndarray, shape (n_samples,)): Targets of a node.

        Returns:
            float: Impurity of ``y`` (0 for a perfectly pure node).
        """

    @abstractmethod
    def _best_split(self, X, y):
        """Find the split that most reduces impurity.

        TODO (students): for each feature and candidate threshold (e.g.
        midpoints between sorted unique values), partition ``y`` into left
        (``x <= threshold``) and right, compute the weighted impurity
        ``(n_l * imp_l + n_r * imp_r) / n`` and keep the split with the
        largest gain over ``self._impurity(y)``. Skip splits leaving a side
        empty.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at the node.
            y (ndarray, shape (n_samples,)): Targets at the node.

        Returns:
            tuple (int, float) or None: ``(feature_index, threshold)`` of the
            best split, or ``None`` if no split improves impurity.
        """
        # Lógica común a clasificación y regresión; las subclases la llaman con super()
        n_samples = len(y)                   # muestras en el nodo
        parent_impurity = self._impurity(y)  # impureza antes de partir

        best_gain = 0.0    # solo se aceptan splits que mejoren (ganancia > 0)
        best_split = None  # (feature, umbral) del mejor split encontrado

        features = self._candidate_features(X.shape[1])  # todas o un subconjunto al azar

        # Recorre cada feature candidata
        for feature_index in features:
            column = X[:, feature_index]                                      # valores de esa columna
            thresholds = self._candidate_thresholds(column, feature_index)   # umbrales a probar

            # Prueba cada umbral y se queda con el de mayor ganancia
            for threshold in thresholds:
                left_mask = column <= threshold  # True = va a la izquierda
                y_left = y[left_mask]            # etiquetas del hijo izquierdo
                y_right = y[~left_mask]          # etiquetas del hijo derecho
                n_left = len(y_left)
                n_right = len(y_right)

                # Descartar si algún hijo queda con menos de min_samples_leaf (incluye vacío)
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                # Impureza ponderada de los hijos: (n_l * imp_l + n_r * imp_r) / n
                weighted = (n_left * self._impurity(y_left)
                            + n_right * self._impurity(y_right)) / n_samples
                gain = parent_impurity - weighted  # cuánto bajó la impureza

                if gain > best_gain:
                    best_gain = gain
                    best_split = (int(feature_index), float(threshold))

        # Sin split, o la mejora no alcanza min_impurity_decrease
        if best_split is None or best_gain < self.min_impurity_decrease:
            return None
        return best_split

    @abstractmethod
    def _leaf_value(self, y):
        """Compute the prediction stored in a leaf.

        TODO (students): return the majority class for classification or the
        mean of ``y`` for regression.

        Args:
            y (ndarray, shape (n_samples,)): Targets that reached the leaf.

        Returns:
            The single predicted value for the leaf.
        """

    @abstractmethod
    def _score(self, y_true, y_pred):
        """Compute the metric reported by ``eval``.

        TODO (students): e.g. accuracy for classification, R^2 for regression.

        Args:
            y_true (ndarray, shape (n_samples,)): Ground-truth targets.
            y_pred (ndarray, shape (n_samples,)): Predicted targets.

        Returns:
            float: Score (higher is better).
        """

    # Revisa que todos los hiperpaámetros tengan valores válidos
    def _validate_params(self):
        if self.max_depth is not None and (not isinstance(self.max_depth, int) or self.max_depth < 0):
            raise ValueError("max_depth must be None or an int >= 0")
        if not isinstance(self.min_samples_split, int) or self.min_samples_split < 2:
            raise ValueError("min_samples_split must be an int >= 2")
        if self.criterion not in self._criteria:
            raise ValueError(f"criterion must be one of {self._criteria}")
        if not isinstance(self.min_samples_leaf, int) or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an int >= 1")

        # max_features puede ser None, "sqrt", un int >= 1 o un float en (0, 1]
        mf = self.max_features
        is_valid = (mf is None
                    or mf == "sqrt"
                    or (isinstance(mf, int) and mf >= 1)
                    or (isinstance(mf, float) and 0 < mf <= 1))
        if not is_valid:
            raise ValueError("max_features must be None, 'sqrt', an int >= 1 or a float in (0, 1]")

        if not isinstance(self.min_impurity_decrease, (int, float)) or self.min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease must be a number >= 0")

    # Marca cada columna como "categorical" o "numeric"
    def _detect_feature_types(self, X):
        types = []
        # Revisa los valores distintos de cada columna
        for j in range(X.shape[1]):
            values = np.unique(X[:, j])          # valores distintos de la columna j
            if np.all(np.isin(values, [0, 1])):  # solo hay 0 y/o 1
                types.append("categorical")
            else:
                types.append("numeric")
        return types

    # Regresa los umbrales a probar para una columna, según su tipo
    def _candidate_thresholds(self, column, feature_index):
        values = np.unique(column)  # valores distintos, ya ordenados
        if len(values) < 2:         # columna constante: no se puede partir
            return np.array([])

        # Categórica (one-hot): un solo split one-vs-rest, x == 0 a la izquierda, x == 1 a la derecha
        if self.feature_types_[feature_index] == "categorical":
            return np.array([0.5])

        # Numérica: puntos medios entre valores consecutivos
        return (values[:-1] + values[1:]) / 2

    # Regresa los índices de las features a revisar en este split
    def _candidate_features(self, n_features):
        k = self._n_features_to_try(n_features)  # cuántas features revisar
        if k == n_features:
            return np.arange(n_features)                             # todas, en orden
        return self._rng.choice(n_features, size=k, replace=False)  # k distintas al azar

    # Traduce max_features a un número concreto de features
    def _n_features_to_try(self, n_features):
        if self.max_features is None:
            return n_features                                 # todas
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))           # raíz cuadrada
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)         # número fijo
        return max(1, int(self.max_features * n_features))   # fracción del total

    # Recorre el árbol con una sola muestra x hasta llegar a una hoja y regresa su valor
    def _predict_one(self, x):
        node = self.root
        # Baja nivel por nivel mientras no sea hoja
        while not node.is_leaf:
            if x[node.feature_index] <= node.threshold:  # regla del nodo
                node = node.left
            else:
                node = node.right
        return node.value

    # Profundidad del árbol entrenado
    def get_depth(self):
        if self.root is None:
            raise RuntimeError("Model is not fitted yet. Call fit first.")
        return self._node_depth(self.root)

    # Profundidad del subárbol que empieza en node
    def _node_depth(self, node):
        if node.is_leaf:
            return 0
        return 1 + max(self._node_depth(node.left), self._node_depth(node.right))

    # Número de hojas del árbol entrenado
    def get_n_leaves(self):
        if self.root is None:
            raise RuntimeError("Model is not fitted yet. Call fit first.")
        return self._count_leaves(self.root)

    # Número de hojas del subárbol que empieza en node
    def _count_leaves(self, node):
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)


# Separa los datos al azar en entrenamiento y prueba; test_size es la fracción para prueba
def train_test_split(X, y, test_size=0.2, random_state=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n_samples = len(y)

    n_test = int(round(test_size * n_samples))  # muestras para prueba
    if n_test < 1 or n_test >= n_samples:
        raise ValueError("test_size leaves an empty train or test set")

    rng = np.random.default_rng(random_state)  # azar reproducible
    indices = rng.permutation(n_samples)       # índices revueltos
    test_idx = indices[:n_test]                # primeros n_test para prueba
    train_idx = indices[n_test:]               # el resto para entrenamiento
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# Validación cruzada k-fold: entrena k veces y regresa el score de cada fold
def cross_val_score(model, X, y, k=5, random_state=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n_samples = len(y)
    if k < 2 or k > n_samples:
        raise ValueError("k must be between 2 and n_samples")

    rng = np.random.default_rng(random_state)  # azar reproducible
    indices = rng.permutation(n_samples)       # índices revueltos
    folds = np.array_split(indices, k)         # k grupos de tamaño casi igual

    scores = []
    # En cada vuelta, el fold i es validación y los demás son entrenamiento
    for i in range(k):
        val_idx = folds[i]                           # índices de validación
        train_mask = np.ones(n_samples, dtype=bool)  # todos marcados para entrenar
        train_mask[val_idx] = False                  # menos los de validación
        model.fit(X[train_mask], y[train_mask])
        scores.append(model.eval(X[val_idx], y[val_idx]))
    return np.array(scores)
