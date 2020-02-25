from mnist_reader import load_mnist
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

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

  return ((x_train, y_train), (x_test, y_test))

def plot(x_axis, train_scores, test_scores, title, name):
  fig, ax = plt.subplots()
  ax.set_xlabel("regularization strength")
  ax.set_ylabel("accuracy")
  ax.set_title(title)
  ax.plot(x_axis, train_scores, marker='o', label="train")
  ax.plot(x_axis, test_scores, marker='o', label="test")
  ax.legend()
  plt.savefig(name)

def train(data, classifier, c_0, a, name, itterations=10):
  train, test = data
  x_train, y_train = train
  x_test, y_test = test

  c, train_scores, test_scores = [], [], []
  for i in range(itterations):
    print(i)
    C = c_0 * a ** i
    
    clf = None
    if classifier == 'lr':
      clf = LogisticRegression(penalty='l2', C=C).fit(x_train, y_train)
    else:
      clf = SVC(kernel='linear', max_iter=200, C=C).fit(x_train, y_train)
    
    train_scores.append(clf.score(x_train, y_train))
    test_scores.append(clf.score(x_test, y_test))
    c.append(C)

  plot(
    c, 
    train_scores, 
    test_scores,
    f'{name.capitalize()} Accuracy vs Regularization',
    f'images/{name}.png'
    )

def flat(l):
  return [i for s in l for i in s]

def kfold(data, classifier, C, k):
  train, test = data
  x_train, y_train = train

  x_chunks = np.split(np.array(x_train), k)
  y_chunks = np.split(np.array(y_train), k)

  scores = []
  for i in range(k):
    print(i)
    xi_train = []
    yi_train = []
    
    for j in range(k):
      if j == i:
        continue
      xi_train.append(x_chunks[j])
      yi_train.append(y_chunks[j])

    xi_train = list(flat(xi_train))
    yi_train = list(flat(yi_train))
    
    xi_test = list(x_chunks[i])
    yi_test = list(y_chunks[i])

    clf = None
    if classifier == 'lr':
      clf = LogisticRegression(penalty='l2', C=C).fit(xi_train, yi_train)
    else:
      clf = SVC(kernel='linear', max_iter=200, C=C).fit(xi_train, yi_train)

    scores.append(clf.score(xi_test, yi_test))

  return scores

if __name__ == '__main__':
  data = process_data()
  # train(data, 'lr', 0.0001, 4,"logistic regression")
  # train(data, 'svm', 0.001, 2.7, "support vector machine")

  print(kfold(data, 'lr', 0.02, 5))