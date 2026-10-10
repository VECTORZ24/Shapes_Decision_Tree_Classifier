"""Decision tree on the 'Triangles and Squares' dataset (IT430, Chapter 2).

Target (class) : Shape
Features       : Color, Outline, Dot
Criterion      : entropy -> same information-gain idea used in the course slides
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import LeaveOneOut, cross_val_score

# ------------------------------------------------------------------ 1. Load
df = pd.read_csv("shapes.csv").drop(columns="#")
print(df, "\n")

X = df[["Color", "Outline", "Dot"]]   # attributes / features
y = df["Shape"]                       # class label (target)

# ------------------------------------------------- 2. Encode categorical data
# scikit-learn trees need numbers -> one-hot encode the categorical columns
X_enc = pd.get_dummies(X)
print("Encoded features:", list(X_enc.columns), "\n")

# ----------------------------------------- 3. Information gain (as in slides)
def entropy(labels):
    p = labels.value_counts(normalize=True)
    return -(p * np.log2(p)).sum()

base = entropy(y)
print(f"Entropy of the whole dataset I = {base:.3f} bits")
for col in X.columns:
    i_res = sum(len(g) / len(df) * entropy(g["Shape"]) for _, g in df.groupby(col))
    print(f"Gain({col}) = {base - i_res:.3f}")
print()

# ------------------------------------------------------------------ 4. Train
# 14 rows only, so we train on everything (the slides do the same).
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X_enc, y)

print("Training accuracy:", clf.score(X_enc, y))

# Leave-one-out CV gives a more honest estimate on such a tiny dataset
loo = cross_val_score(
    DecisionTreeClassifier(criterion="entropy", random_state=42),
    X_enc, y, cv=LeaveOneOut(),
)
print(f"Leave-one-out accuracy: {loo.mean():.2f}\n")

# ----------------------------------------------------------- 5. Show the tree
print(export_text(clf, feature_names=list(X_enc.columns)))

plt.figure(figsize=(14, 8))
plot_tree(
    clf,
    feature_names=list(X_enc.columns),
    class_names=clf.classes_,
    filled=True,
    rounded=True,
)
plt.title("Decision tree - Triangles and Squares")
plt.tight_layout()
plt.savefig("decision_tree_shapes.png", dpi=150)
print("Saved decision_tree_shapes.png")

# -------------------------------------------------------------- 6. Predict new
new = pd.DataFrame([{"Color": "red", "Outline": "dashed", "Dot": "yes"}])
new_enc = pd.get_dummies(new).reindex(columns=X_enc.columns, fill_value=0)
print("Prediction for red / dashed / dot=yes ->", clf.predict(new_enc)[0])
