from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
''' 
Load the dataset. 
'''
df = pd.read_csv('emails.csv')

'''
Drop the 'Email no' column. 
'''
df = df.drop(columns=['Email No.'])

'''
Set the target column to the Prediction column. 
'''
target_column = 'Prediction'
y = df[target_column].to_numpy()

'''Set the rest of the columns as the input features
'''
X_df = df.drop(columns=[target_column])
X = X_df.to_numpy()

'''
Split the dataset
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


'''
Initialize the decision tree classifier object. 
'''
tree_height = 2
tree_clf = DecisionTreeClassifier(max_depth=tree_height)

'''
Train the decision tree classifier
'''
tree_clf.fit(X_train, y_train)


'''
Visualize the trained decision tree
'''
export_graphviz(
    tree_clf,
    out_file="decision_tree.dot",
    feature_names=X_df.columns.to_list(),
    class_names=['normal', 'spam'],
    rounded=True,
    filled=True
)

'''
Convert the .dot file to .png
'''
(graph,) = pydot.graph_from_dot_file('decision_tree.dot')
graph.write_png('decision_tree.png')

'''
Calculate the confusion matrix
'''
y_pred = tree_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['normal', "spam"])
disp.plot()
plt.savefig("confusion-matrix-dt.png")