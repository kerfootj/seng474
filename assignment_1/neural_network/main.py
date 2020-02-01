from sklearn.neural_network import MLPClassifier
import sys, pickle

def main():
  if len(sys.argv) < 2:
    print('Usage: python main.py <file_name> -n <out_name>')
    exit()
  
  file_name = sys.argv[1]

  data = None
  with open('data/' + file_name, 'rb') as f:
    data = pickle.load(f)

  x_train, y_train = data["training_data"], data["training_target"]
  x_test, y_test = data["test_data"], data["test_target"]

  clf = MLPClassifier(
    solver='sgd', 
    alpha=0.001, 
    hidden_layer_sizes=(10), 
    random_state=0
  )

  clf.fit(x_train, y_train)
  score = clf.score(x_test, y_test)

  print(score)


if __name__ == "__main__":
  main()