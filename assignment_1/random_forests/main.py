import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys, pickle

plot_num = 0

def generate_forests(data, criterion, max_features, name):
  x_train, y_train = data["training_data"], data["training_target"]
  x_test, y_test = data["test_data"], data["test_target"]

  estimators = [2, 10, 18, 26, 34, 42, 50, 58]
  depths = [1, 2, 3, 4, 5, 6]

  best_tree, best_score = None, 0
  X, Y, Z = [], [], []
  for estimator in estimators:
    x, y, z = [], [], []
    for depth in depths:
      clf = RandomForestClassifier(
        criterion=criterion, 
        n_estimators=estimator, 
        max_depth=depth, 
        max_features="sqrt", 
        random_state=0
      )
      clf.fit(x_train, y_train)
      score = clf.score(x_test, y_test)

      if (score > best_score):
        best_score = score
        best_tree = clf

      x.append(estimator)
      y.append(depth)
      z.append(score)

    X.append(x)
    Y.append(y)
    Z.append(z)

  plot_num += 1
  plt.figure(plot_num)
  plt.contour(X, Y, Z, 20)
  plt.colorbar()

  feature_name = 'none' if max_features == None else max_features
  plot_name = name + '_' + criterion + '_' + feature_name
  plt.savefig('images/' + plot_name + '.png')

  return best_tree, best_score


def get_best_forest(data, criterion, name):
 
  best_forest_sqrt, best_score_sqrt = generate_forests(data, criterion, 'sqrt', name)
  best_forest_none, best_score_none = generate_forests(data, criterion, None, name)

  if best_score_sqrt > best_score_none:
    return best_forest_sqrt, best_score_sqrt
  return best_forest_none, best_score_none


def main():
  if len(sys.argv) < 2:
    print('Usage: python main.py <file_name>')
    exit()

  file_name = sys.argv[1]

  out_name = ''
  if '-n' in sys.argv:
    out_name = sys.argv[sys.argv.index('-n') + 1]

  data = None
  with open('data/' + file_name, 'rb') as f:
    data = pickle.load(f)

  best_tree_gini, best_score_gini = get_best_forest(data, criterion="gini", name=out_name)
  best_tree_entr, best_score_entr = get_best_forest(data, criterion="entropy", name=out_name)

if __name__ == "__main__":
  main()