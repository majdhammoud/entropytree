# entropytree

A decision tree classifier written from scratch (only `numpy` and `pandas`), using **entropy** and **information gain**. It works directly on a pandas DataFrame.

```python
import pandas as pd
from entropytree import DecisionTree

df = pd.read_csv("data.csv")
tree = DecisionTree(max_depth=3, min_samples_split=2, min_samples_leaf=1)
tree.fit(df, target="label")

tree.predict(df)      # numpy array of labels
tree.score(df)        # accuracy
print(tree)           # or tree.display()
```

## Hyperparameters
| name | meaning |
|---|---|
| `max_depth` | maximum depth of the tree (`None` = unlimited) |
| `min_samples_split` | minimum samples in a node to try splitting it |
| `min_samples_leaf` | minimum samples in each child of a split |
| `min_gain` | minimum information gain to accept a split |

## How it works
- `entropy(y) = -sum(p * log2 p)`
- `gain = entropy(parent) - sum(|child|/|parent| * entropy(child))`
- Numeric columns: binary split `<= t`, where `t` is the best midpoint between sorted unique values.
- Categorical / bool / object columns: one branch per category.
- Assumes clean data (no NaNs).

## Install / test
```
pip install .
pip install pytest && pytest
```
