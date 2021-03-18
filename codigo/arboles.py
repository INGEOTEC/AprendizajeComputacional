from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

X, y = load_iris(return_X_y=True)

arbol = tree.DecisionTreeClassifier(min_impurity_decrease=0.05).fit(X, y)

with open("tree.dot", "w") as fpt:
    _ = tree.export_graphviz(arbol)
    fpt.write(_)

!dot -Tpng tree.dot -o tree3.png