from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import sys, pickle

plot_num = 0

def generate_networks(data, solver, name):
  x_train, y_train = data["training_data"], data["training_target"]
  x_test, y_test = data["test_data"], data["test_target"]

  sizes = list(range(1, 30))

  clfs = []
  for size in sizes:
    clf = MLPClassifier(
      solver=solver, 
      hidden_layer_sizes=size, 
      alpha=0.00001,
      learning_rate='adaptive'
    )
    clf.fit(x_train, y_train)
    clfs.append(clf)

  train_scores = [clf.score(x_train, y_train) for clf in clfs]
  test_scores = [clf.score(x_test, y_test) for clf in clfs]

  fig, ax = plt.subplots()
  ax.set_xlabel("number of nodes in hidden layer")
  ax.set_ylabel("accuracy")
  ax.set_title("Accuracy vs Size of Hidden Layer for training and test sets")
  ax.plot(sizes, train_scores, marker='o', label="train", drawstyle="steps-post")
  ax.plot(sizes, test_scores, marker='o', label="test", drawstyle="steps-post")
  ax.legend()

  plot_name = name + '_' + solver
  plt.savefig('images/' + plot_name + '.png')

  i = test_scores.index(max(test_scores))

  return clfs[i], test_scores[i], sizes[i]

def get_best_network(data, name):
  best_network_sgd, best_score_sgd, best_attrs_sgd = generate_networks(data, 'sgd', name)
  best_network_adam, best_score_adam, best_attrs_adam = generate_networks(data, 'adam', name)

  if best_score_sgd > best_score_adam:
    return best_network_sgd, best_score_sgd, best_attrs_sgd, 'sgd'
  return best_network_adam, best_score_adam, best_attrs_adam, 'adam'

def main():
  if len(sys.argv) < 2:
    print('Usage: python main.py <file_name> -n <out_name>')
    exit()
  
  file_name = sys.argv[1]

  out_name = ''
  if '-n' in sys.argv:
    out_name = sys.argv[sys.argv.index('-n') + 1]

  data = None
  with open('data/' + file_name, 'rb') as f:
    data = pickle.load(f)

  network, score, size, solver = get_best_network(data, out_name)

  print('{}, {}, {}'.format(score, size, solver))

if __name__ == "__main__":
  main()