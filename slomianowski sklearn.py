import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score, classification_report
import pydotplus

df = pd.read_csv(r"cardio_train.csv", sep=';')
df = df.drop(columns = ['id'])

X = df.iloc[:, 0:11]
y = df.iloc[:, 11]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43)
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

export_graphviz(clf, out_file='sk_tree.dot', feature_names=X.columns, class_names=['0', '1'], filled=True)
graph = pydotplus.graph_from_dot_file('sk_tree.dot')
graph.write_png('sk_tree.png')

fig = plt.figure(figsize=(15, 10))
_ = plot_tree(clf, filled=True)
plt.show()