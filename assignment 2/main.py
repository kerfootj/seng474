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

  mid = int(len(x_train) / 2)
  return ((x_train, y_train), (x_test, y_test)), ((x_train[:mid], y_train[:mid]), (x_test, y_test))

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
      clf = SVC(kernel='linear', C=C).fit(x_train, y_train)
    
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
  x_train, y_train = data

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
      clf = LogisticRegression(penalty='l2', max_iter=200, C=C).fit(xi_train, yi_train)
    else:
      clf = SVC(kernel='linear', C=C).fit(xi_train, yi_train)

    scores.append(clf.score(xi_test, yi_test))

  return sum(scores) / len(scores)

def verfify_kfold(data, classifier, C):
  train, test = data
  x_train, y_train = train
  x_test, y_test = test

  clf = None
  if classifier == 'lr':
    clf = LogisticRegression(penalty='l2', max_iter=200, C=C).fit(x_train, y_train)
  else:
    clf = SVC(kernel='linear', max_iter=200, C=C).fit(x_train, y_train)

  train_score = clf.score(x_train, y_train)
  test_score = clf.score(x_test, y_test)

  print(f'{classifier} - training: {train_score} testing: {test_score}')

def train_kfold(data, classifier, c_0, a, k=5, itterations=10):
  train, test = data

  results = []
  print('training...')
  for i in range(itterations):
    print(f'itteration {i}')
    results.append(kfold(train, classifier, c_0 * a ** i, k))

  best = results.index(max(results))
  
  verfify_kfold(data, classifier, c_0 * a ** best)

if __name__ == '__main__':
  print('processing data...')
  data, half_data = process_data()
  print('done processing!')
  
  # train(data, 'lr', 0.0001, 4,"logistic regression")
  #train(half_data, 'svm', 0.04, 1.6, "support vector machine")

  # train_kfold(data, 'lr', 0.7, 1.1, k=5) # lr - training: 0.9738333333333333 testing: 0.9585
  train_kfold(half_data, 'svm', 0.04, 1.5, k=5) # svm - training: 0.9115 testing: 0.907 