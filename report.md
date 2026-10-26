# Report: Decision Tree from Scratch

**GitHub:** <ADD YOUR REPO LINK HERE>
**PyPI (bonus):** <ADD LINK AFTER UPLOAD, or "not deployed">

## Description
I implemented a decision tree classifier from scratch (only `numpy`/`pandas`, no scikit-learn) in the package `entropytree`. It splits nodes by maximising **information gain** based on **Shannon entropy**, and it works on any (clean) pandas DataFrame.

### Method
- `entropy(y) = -Σ p·log2(p)`
- `information_gain = entropy(parent) − Σ (n_child/n_parent)·entropy(child)`
- At each node, every feature is evaluated and the one with the highest gain is chosen.
  - **Numeric columns:** binary split `<= threshold`; candidates are midpoints between consecutive sorted unique values, computed in one sorted sweep with cumulative class counts.
  - **Categorical / bool / object columns:** multiway split, one branch per category.
- A node becomes a leaf (predicting the majority class) when it is pure, reaches `max_depth`, has fewer than `min_samples_split` samples, or no split gives enough gain / respects `min_samples_leaf`.
- Unseen categories at prediction time fall back to the majority class of the current node.

### Hyperparameters (tree size)
`max_depth`, `min_samples_split`, `min_samples_leaf`, `min_gain`.

### API
```python
tree = DecisionTree(max_depth=3).fit(df, target="label")
tree.predict(df); tree.score(df); print(tree)
```

### Bonus: displaying the tree
`print(tree)` / `tree.display()` prints an indented text tree with the split conditions and class counts at leaves, e.g.:
```
|-- outlook == sunny
|   |-- temp <= 77.5 -> yes  {'no': 1, 'yes': 2}
|   |-- temp > 77.5 -> no  {'no': 2}
|-- outlook == overcast -> yes  {'yes': 4}
|-- outlook == rain
|   |-- windy == False -> yes  {'yes': 3}
|   |-- windy == True -> no  {'no': 2}
```

### Bonus: packaging
The project is a standard `pyproject.toml` package (`src/` layout) ready to be built with `python -m build` and uploaded with `twine`.

### Testing
`pytest` suite (9 tests) checks entropy/gain values, the classic "play tennis" root split (outlook), numeric thresholds, depth limits, leaf-size limits, unseen categories and error handling.

### Assumptions
Data is already cleaned (no missing values), as instructed.
