"""Decision tree classifier built from scratch with entropy and information gain."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype


def entropy(y) -> float:
    """Shannon entropy (base 2) of a label array."""
    _, counts = np.unique(np.asarray(y), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def information_gain(parent_y, children_y) -> float:
    """Entropy of the parent minus the weighted entropy of the children."""
    n = len(parent_y)
    weighted = sum(len(c) / n * entropy(c) for c in children_y)
    return entropy(parent_y) - weighted


class _Node:
    __slots__ = ("prediction", "counts", "feature", "threshold", "children", "default")

    def __init__(self, prediction, counts):
        self.prediction = prediction
        self.counts = counts
        self.feature = None
        self.threshold = None
        self.children = {}

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTree:
    """ID3/C4.5-style classification tree that works on any pandas DataFrame.

    Numeric columns are split with a binary ``<= threshold`` test; all other
    columns (object, category, bool) get one branch per category.

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree (the root is depth 0). ``None`` = unlimited.
    min_samples_split : int
        Minimum number of samples a node needs to be considered for splitting.
    min_samples_leaf : int
        Minimum number of samples every child of a split must contain.
    min_gain : float
        Minimum information gain required to make a split.
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_gain=1e-12):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.root_ = None

    def fit(self, df: pd.DataFrame, target: str):
        """Fit on ``df``; ``target`` is the name of the label column."""
        if target not in df.columns:
            raise KeyError(f"target column {target!r} not found in DataFrame")
        X = df.drop(columns=[target])
        y = df[target].to_numpy()
        self.target_ = target
        self.features_ = list(X.columns)
        self.numeric_ = {c: is_numeric_dtype(X[c]) and not is_bool_dtype(X[c]) for c in X.columns}
        self.classes_ = np.unique(y)
        self.root_ = self._build(X.reset_index(drop=True), y, depth=0)
        return self

    def _build(self, X, y, depth):
        classes, counts = np.unique(y, return_counts=True)
        node = _Node(classes[counts.argmax()], dict(zip(classes.tolist(), counts.tolist())))

        if (
            len(classes) == 1
            or len(y) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node

        best = self._best_split(X, y)
        if best is None:
            return node
        gain, feature, threshold, masks = best
        if gain < self.min_gain:
            return node

        node.feature, node.threshold = feature, threshold
        for key, mask in masks.items():
            node.children[key] = self._build(X[mask].reset_index(drop=True), y[mask], depth + 1)
        return node

    def _best_split(self, X, y):
        best, best_gain = None, -1.0
        for feature in self.features_:
            col = X[feature]
            if self.numeric_[feature]:
                cand = self._best_threshold(col.to_numpy(dtype=float), y)
                if cand is None:
                    continue
                gain, thr = cand
                if gain > best_gain:
                    v = col.to_numpy(dtype=float)
                    best_gain = gain
                    best = (gain, feature, thr, {"<=": v <= thr, ">": v > thr})
            else:
                values = col.to_numpy()
                masks = {v: values == v for v in pd.unique(values)}
                if len(masks) < 2 or any(m.sum() < self.min_samples_leaf for m in masks.values()):
                    continue
                gain = information_gain(y, [y[m] for m in masks.values()])
                if gain > best_gain:
                    best_gain, best = gain, (gain, feature, None, masks)
        return best

    def _best_threshold(self, v, y):
        """Best midpoint threshold for a numeric column (sorted sweep, O(n log n))."""
        order = np.argsort(v, kind="stable")
        v, y = v[order], y[order]
        n = len(y)
        classes = self.classes_
        onehot = (y[:, None] == classes[None, :]).astype(float)
        left = np.cumsum(onehot, axis=0)
        total = left[-1]
        parent = entropy(y)

        def H(counts, sizes):
            p = counts / sizes[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                return -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=1)

        idx = np.arange(n - 1)
        valid = v[idx] != v[idx + 1]
        nl = idx + 1
        nr = n - nl
        valid &= (nl >= self.min_samples_leaf) & (nr >= self.min_samples_leaf)
        if not valid.any():
            return None
        idx, nl, nr = idx[valid], nl[valid], nr[valid]
        gains = parent - (nl / n) * H(left[idx], nl.astype(float)) - (nr / n) * H(total - left[idx], nr.astype(float))
        k = gains.argmax()
        return float(gains[k]), float((v[idx[k]] + v[idx[k] + 1]) / 2)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict a label for every row of ``df`` (target column optional)."""
        self._check_fitted()
        rows = df.to_dict("records")
        return np.array([self._predict_row(self.root_, r) for r in rows])

    def _predict_row(self, node, row):
        while not node.is_leaf:
            val = row[node.feature]
            if node.threshold is not None:
                key = "<=" if val <= node.threshold else ">"
            else:
                key = val
            if key not in node.children:
                break
            node = node.children[key]
        return node.prediction

    def score(self, df: pd.DataFrame, target: str | None = None) -> float:
        """Accuracy on ``df``."""
        target = target or self.target_
        return float((self.predict(df) == df[target].to_numpy()).mean())

    def depth(self) -> int:
        self._check_fitted()

        def d(n):
            return 0 if n.is_leaf else 1 + max(d(c) for c in n.children.values())

        return d(self.root_)

    def n_leaves(self) -> int:
        self._check_fitted()

        def c(n):
            return 1 if n.is_leaf else sum(c(x) for x in n.children.values())

        return c(self.root_)

    def export_text(self) -> str:
        """Return the tree as indented text."""
        self._check_fitted()
        lines = []

        def leaf(n):
            return f"-> {n.prediction}  {n.counts}"

        def rec(n, indent):
            for key, child in n.children.items():
                if n.threshold is not None:
                    cond = f"{n.feature} {key} {n.threshold:.4g}"
                else:
                    cond = f"{n.feature} == {key}"
                if child.is_leaf:
                    lines.append(f"{indent}|-- {cond} {leaf(child)}")
                else:
                    lines.append(f"{indent}|-- {cond}")
                    rec(child, indent + "|   ")

        if self.root_.is_leaf:
            lines.append(leaf(self.root_))
        else:
            rec(self.root_, "")
        return "\n".join(lines)

    def display(self):
        """Print the tree."""
        print(self.export_text())

    __str__ = export_text

    def _check_fitted(self):
        if self.root_ is None:
            raise RuntimeError("tree is not fitted yet; call fit() first")
