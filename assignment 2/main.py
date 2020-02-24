from mnist_reader import load_mnist
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def filter_data(x, y):
  x.tolist()
  y.tolist()

  temp_x, temp_y = [], []
  for i in range(len(x)):
    if y[i] in [5, 7]:
      temp_x.append(x[i] / 255)
      temp_y.append(0 if y[i] == 5 else 1)
  return temp_x, temp_y

def process_data():
  x_train, y_train = load_mnist('data', kind='train')
  x_test, y_test = load_mnist('data', kind='t10k')

  x_train, y_train = filter_data(x_train, y_train)
  x_test, y_test = filter_data(x_test, y_test)

  return (x_train, y_train), (x_test, y_test)

def train_lr(train, test):
  x_train, y_train = train
  x_test, y_test = test

  c_0 = 0.0001
  a = 4

  c, train_scores, test_scores = [], [], []
  for i in range(10):
    print(i)
    C = c_0 * a ** i
    clf = LogisticRegression(penalty='l2', C=C).fit(x_train, y_train)
    train_scores.append(clf.score(x_train, y_train))
    test_scores.append(clf.score(x_test, y_test))
    c.append(C)

  fig, ax = plt.subplots()
  ax.set_xlabel("regularization strength")
  ax.set_ylabel("accuracy")
  ax.set_title("Accuracy vs regularization for training and testing sets")
  ax.plot(c, train_scores, marker='o', label="train")
  ax.plot(c, test_scores, marker='o', label="test")
  ax.legend()

  plt.savefig('images/logistic_regression.png')

if __name__ == '__main__':
  train, test = process_data()
  train_lr(train, test)