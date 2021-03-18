from sklearn import datasets
from sklearn import tree

X, y = datasets.load_boston(return_X_y=True)
X, y = datasets.load_diabetes(return_X_y=True)

arbol = tree.DecisionTreeRegressor(max_depth=2, min_samples_split=10, splitter="random").fit(X, y)

with open("tree.dot", "w") as fpt:
    _ = tree.export_graphviz(arbol)
    fpt.write(_)

!dot -Tpng tree.dot -o tree.png
