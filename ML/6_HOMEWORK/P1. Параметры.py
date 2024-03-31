# import pandas as pd
import numpy as np
from math import floor

np.random.seed(42)

# Импортируйте необходимые классы и функции из соответствующих модулей sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.6, shuffle=True)
clf = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)

clf.fit(X_train, y_train)
preds = clf.predict(x_test)

acc = accuracy_score(y_test, preds)

print(acc)