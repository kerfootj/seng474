from mnist_reader import load_mnist
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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

def train(data, classifier, c_0, a, itterations=10, name=''):
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


if __name__ == '__main__':
  data = process_data()
  train(data, 'lr', 0.0001, 4, name="logistic regression")
  train(data, 'svm', 0.001, 2.7, name="support vector machine")