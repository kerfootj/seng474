from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
from IPython.display import Image  
import pydotplus

iris = load_iris()

print(len(iris.data))
print(iris.data)
print(len(iris.target))
print(iris.target)

decision_tree_entropy = DecisionTreeClassifier(random_state=0, max_depth=2, criterion="entropy")
decision_tree_entropy = decision_tree_entropy.fit(iris.data, iris.target)
decision_tree_entropy.predict([[3.6,2.5,4.5,2.1]])
decision_tree_entropy

dot_data = StringIO()
export_graphviz(decision_tree_entropy, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
image = graph.create_png()

with open('graph.png', 'wb') as f:
  f.write(image)