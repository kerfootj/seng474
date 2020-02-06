# Refernece: http://bit.ly/2vycpoo

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO 
from IPython.display import Image
import matplotlib.pyplot as plt
import pydotplus, pickle, sys

def get_best_tree(data, criterion, name="", graph=False):

  x_train, y_train = data["training_data"], data["training_target"]
  x_test, y_test = data["test_data"], data["test_target"]
  
  clf = DecisionTreeClassifier(random_state=0, criterion=criterion)
  path = clf.cost_complexity_pruning_path(x_train, y_train)
  ccp_alphas, impurities = path.ccp_alphas, path.impurities

  clfs = []
  for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, criterion=criterion, ccp_alpha=ccp_alpha)
    clf.fit(x_train, y_train)
    clfs.append(clf)

  # Remove tree with only one node
  clfs = clfs[:-1]
  ccp_alphas = ccp_alphas[:-1]

  train_scores = [clf.score(x_train, y_train) for clf in clfs]
  test_scores = [clf.score(x_test, y_test) for clf in clfs]
  node_counts = [clf.tree_.node_count for clf in clfs]
  depth = [clf.tree_.max_depth for clf in clfs]

  if graph:
    # Graph num nodes and depth vs alpha
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()

    plot_name = "images/plot0_" + criterion + ".png" if name == "" else "images/" + name + "_plot0_" + criterion + ".png"
    plt.savefig(plot_name)

    # Graph accuracy vs alpha
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
    ax.legend()

    plot_name = "images/plot1_" + criterion + ".png" if name == "" else "images/" + name + "_plot1_" + criterion + ".png"
    plt.savefig(plot_name)

  i = test_scores.index(max(test_scores))
  return clfs[i], train_scores[i], test_scores[i], node_counts[i]

def visualize_tree(tree, feature_names, class_names, criterion, name=""):
  dot_data = StringIO()
  export_graphviz(
    tree, 
    out_file=dot_data, 
    filled=True, 
    rounded=True, 
    special_characters=True, 
    feature_names=feature_names, 
    class_names=class_names
  )

  graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
  image = graph.create_png()

  plot_name = "images/plot2_" + criterion + ".png" if name == "" else "images/" + name + "_plot2_" + criterion + ".png"

  with open(plot_name, 'wb') as f:
    f.write(image)

def main():
  if len(sys.argv) < 2:
    print('Usage: python main.py <file_name>')
    exit()

  file_name = sys.argv[1]
  graph = '-g' in sys.argv

  out_name = ''
  if '-n' in sys.argv:
    out_name = sys.argv[sys.argv.index('-n') + 1]

  data = None
  with open(file_name, 'rb') as f:
    data = pickle.load(f)

  entropy_tree, e_tr, e_te, e_nc = get_best_tree(data, criterion="entropy", graph=graph, name=out_name)
  gini_tree, g_tr, g_te, g_nc = get_best_tree(data, criterion="gini", graph=graph, name=out_name)

  print('{} {} {}'.format(e_tr, e_te, e_nc))
  print('{} {} {}'.format(g_tr, g_te, g_nc))

  if graph:
    feature_names, class_names = data["feature_names"], data["class_names"]
    visualize_tree(entropy_tree, feature_names, class_names, "entropy", name=out_name)
    visualize_tree(gini_tree, feature_names, class_names, "gini", name=out_name)

if __name__ == "__main__":
  main()