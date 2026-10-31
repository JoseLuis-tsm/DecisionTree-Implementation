import numpy as np
import pytest

from decision_tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                           train_test_split, cross_val_score)


# ---------- Impureza ----------

# Gini y entropía con valores calculados a mano
def test_classifier_impurity_values():
    clf = DecisionTreeClassifier()
    y = np.array([0, 0, 1, 1])
    assert clf._impurity(y) == pytest.approx(0.5)                # 1 - (0.25 + 0.25)
    assert clf._impurity(np.array([1, 1, 1])) == pytest.approx(0.0)  # nodo puro
    clf.set_params(criterion="entropy")
    assert clf._impurity(y) == pytest.approx(1.0)                # -2 * 0.5 * log2(0.5)
    assert clf._impurity(np.array([1, 1, 1])) == pytest.approx(0.0)


# MSE alrededor de la media y MAE alrededor de la mediana
def test_regressor_impurity_values():
    reg = DecisionTreeRegressor()
    y = np.array([1.0, 2.0, 6.0])
    assert reg._impurity(y) == pytest.approx(14 / 3)  # media 3: (4 + 1 + 9) / 3
    reg.set_params(criterion="mae")
    assert reg._impurity(y) == pytest.approx(5 / 3)   # mediana 2: (1 + 0 + 4) / 3


# ---------- Árbol de clasificación ----------

# El mejor umbral separa las clases en el punto medio 6.5
def test_numeric_split_and_predict():
    X = [[2.0], [3.0], [10.0], [11.0]]
    y = [0, 0, 1, 1]
    clf = DecisionTreeClassifier().fit(X, y)
    assert clf.root.feature_index == 0
    assert clf.root.threshold == pytest.approx(6.5)
    assert list(clf.predict([[1.0], [12.0]])) == [0, 1]
    assert clf.eval(X, y) == 1.0


# Funciona con etiquetas de texto
def test_string_labels():
    X = [[1.0], [2.0], [8.0], [9.0]]
    y = ["no", "no", "yes", "yes"]
    clf = DecisionTreeClassifier().fit(X, y)
    assert list(clf.predict([[0.0], [10.0]])) == ["no", "yes"]


# Gini y entropía dan el mismo árbol en un caso separable
def test_gini_vs_entropy():
    X = [[1, 5], [2, 4], [3, 7], [8, 1], [9, 2], [10, 3]]
    y = [0, 0, 0, 1, 1, 1]
    gini = DecisionTreeClassifier(criterion="gini").fit(X, y)
    entropy = DecisionTreeClassifier(criterion="entropy").fit(X, y)
    assert gini.eval(X, y) == 1.0
    assert entropy.eval(X, y) == 1.0
    assert gini.get_depth() == entropy.get_depth()
    assert gini.get_n_leaves() == entropy.get_n_leaves()


#  Atributos categóricos one-hot

# Columnas one-hot: [rojo, verde, azul]; la clase es 1 solo si el color es verde
def test_categorical_one_vs_rest():
    X = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    y = [0, 1, 0, 0, 1, 0]
    clf = DecisionTreeClassifier().fit(X, y)
    assert clf.feature_types_ == ["categorical", "categorical", "categorical"]
    assert clf.root.feature_index == 1        # columna "verde"
    assert clf.root.threshold == 0.5          # split one-vs-rest
    assert clf.eval(X, y) == 1.0


# Una categoría no vista llega como puros ceros y se va a la rama "no es la categoría"
def test_unseen_category():
    X = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    y = [0, 1, 0, 0, 1, 0]
    clf = DecisionTreeClassifier().fit(X, y)
    assert list(clf.predict([[0, 0, 0]])) == [0]  # no es verde -> clase 0


# Mezcla de una columna numérica con un bloque one-hot
def test_mixed_numeric_and_categorical():
    # columnas: [edad, es_estudiante, es_empleado]
    X = [[18, 1, 0], [22, 1, 0], [35, 0, 1], [40, 0, 1], [50, 0, 1], [20, 1, 0]]
    y = [1, 1, 0, 0, 0, 1]
    clf = DecisionTreeClassifier().fit(X, y)
    assert clf.feature_types_ == ["numeric", "categorical", "categorical"]
    assert clf.eval(X, y) == 1.0


# ---------- Hiperparámetros ----------

# max_depth=0 deja solo la raíz como hoja con la clase mayoritaria
def test_max_depth_zero():
    clf = DecisionTreeClassifier(max_depth=0).fit([[1], [2], [3]], [0, 1, 1])
    assert clf.root.is_leaf
    assert clf.get_depth() == 0
    assert list(clf.predict([[100]])) == [1]


# max_depth limita la profundidad
def test_max_depth_limits_tree():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = rng.integers(0, 2, size=60)
    clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert clf.get_depth() <= 2


# min_samples_split mayor que n impide partir la raíz
def test_min_samples_split():
    clf = DecisionTreeClassifier(min_samples_split=10).fit([[1], [2], [3]], [0, 1, 1])
    assert clf.root.is_leaf


# min_samples_leaf=2 prohíbe el split en 1.5 (dejaría 1 muestra a la izquierda)
def test_min_samples_leaf():
    X = [[1], [2], [3], [4], [5]]
    y = [0, 1, 1, 1, 1]
    assert DecisionTreeClassifier().fit(X, y).root.threshold == pytest.approx(1.5)
    clf = DecisionTreeClassifier().set_params(min_samples_leaf=2).fit(X, y)
    assert clf.root.threshold == pytest.approx(2.5)


# min_impurity_decrease alto impide cualquier split
def test_min_impurity_decrease():
    clf = DecisionTreeClassifier().set_params(min_impurity_decrease=0.9)
    clf.fit([[1], [2], [3], [4]], [0, 0, 1, 1])
    assert clf.root.is_leaf


# max_features con la misma semilla da el mismo árbol
def test_max_features_reproducible():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 6))
    y = (X[:, 0] + X[:, 3] > 0).astype(int)
    a = DecisionTreeClassifier().set_params(max_features="sqrt", random_state=42).fit(X, y)
    b = DecisionTreeClassifier().set_params(max_features="sqrt", random_state=42).fit(X, y)
    assert list(a.predict(X)) == list(b.predict(X))
    c = DecisionTreeClassifier().set_params(max_features=0.5, random_state=0).fit(X, y)
    assert c.eval(X, y) > 0.5


# Valores inválidos lanzan ValueError en fit
@pytest.mark.parametrize("params", [
    {"max_depth": -1},
    {"max_depth": 2.5},
    {"min_samples_split": 1},
    {"min_samples_leaf": 0},
    {"max_features": 0},
    {"max_features": 1.5},
    {"max_features": "log"},
    {"min_impurity_decrease": -0.1},
    {"criterion": "mse"},
])
def test_invalid_params(params):
    clf = DecisionTreeClassifier().set_params(**params)
    with pytest.raises(ValueError):
        clf.fit([[1], [2]], [0, 1])


# Errores de uso: predecir sin entrenar, formas que no coinciden
def test_usage_errors():
    clf = DecisionTreeClassifier()
    with pytest.raises(RuntimeError):
        clf.predict([[1]])
    with pytest.raises(ValueError):
        clf.fit([[1], [2]], [0])           # distinto número de filas
    clf.fit([[1, 2], [3, 4]], [0, 1])
    with pytest.raises(ValueError):
        clf.predict([[1, 2, 3]])           # distinto número de columnas


#  Validación

# train_test_split respeta tamaños
def test_train_test_split():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    assert len(X_tr) == 7 and len(X_te) == 3
    assert len(y_tr) == 7 and len(y_te) == 3


# cross_val_score regresa un score por fold
def test_cross_val_score():
    X = [[i] for i in range(20)]
    y = [0] * 10 + [1] * 10
    scores = cross_val_score(DecisionTreeClassifier(), X, y, k=4, random_state=0)
    assert len(scores) == 4
    assert np.all((scores >= 0) & (scores <= 1))


#Árbol de regresión

# Función escalón: el corte debe quedar en 4.5 y la hoja guarda la media
def test_regressor_step_function():
    X = [[i] for i in range(10)]
    y = [0.0] * 5 + [10.0] * 5
    reg = DecisionTreeRegressor().fit(X, y)
    assert reg.root.threshold == pytest.approx(4.5)
    assert list(reg.predict([[0], [9]])) == [0.0, 10.0]
    assert reg.eval(X, y) == pytest.approx(1.0)


# La hoja guarda el promedio de y
def test_regressor_leaf_is_mean():
    reg = DecisionTreeRegressor(max_depth=0).fit([[1], [2], [3]], [1.0, 2.0, 6.0])
    assert reg.predict([[5]])[0] == pytest.approx(3.0)


# R^2 = 0 cuando el modelo siempre predice la media
def test_regressor_r2_of_mean_is_zero():
    reg = DecisionTreeRegressor()
    y = np.array([1.0, 2.0, 3.0])
    assert reg._score(y, np.full(3, 2.0)) == pytest.approx(0.0)


# MAE también construye un árbol útil, y los categóricos funcionan en regresión
def test_regressor_mae_and_categorical():
    X = [[1, 0], [1, 0], [0, 1], [0, 1]]
    y = [5.0, 5.0, 20.0, 20.0]
    reg = DecisionTreeRegressor(criterion="mae").fit(X, y)
    assert reg.eval(X, y) == pytest.approx(1.0)


# El regresor rechaza criterios de clasificación
def test_regressor_invalid_criterion():
    with pytest.raises(ValueError):
        DecisionTreeRegressor(criterion="gini").fit([[1], [2]], [1.0, 2.0])
