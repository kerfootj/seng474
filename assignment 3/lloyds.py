# references:
#     https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
#     https://gist.github.com/larsmans/4952848
#     https://www.geeksforgeeks.org/ml-k-means-algorithm/

import sys, csv, random
import numpy as np
import matplotlib.pyplot as plt

def initialize_uniform_random(data, k):
  return random.sample(data, k)

def initialize_kmeans_plusplus(data, k):
  mu = random.sample(data, 1)
  for c in range(k-1):
    # calculate probabilities
    distances = []
    for x in data:
      distances.append(closest(x, mu)[1] ** 2)
    prob = [d / sum(distances) for d in distances]

    # select next center
    val = random.random()
    total = 0
    for i in range(len(prob)):
      total += prob[i]
      if total > val:
        mu.append(data[i])
        break

  return mu

def closest(x, mu):
  # min([(index, distance), ...]) --> (best_index, best_distance)
  return min([(i, np.linalg.norm(x-m)) for i, m in enumerate(mu)], key=lambda z:z[1])

def cluster(X, mu):
  # key - index of mu
  # value - list of points in cluster
  clusters = {} 
  for i, x in enumerate(X):
    best = closest(x, mu)[0]
    if best in clusters:
      clusters[best].append(x)
    else:
      clusters[best] = [x]
  return clusters

def center(mu, clusters):
  new_mu = []
  for key in sorted(clusters.keys()):
    new_mu.append(np.mean(clusters[key], axis = 0))
  return new_mu

# update until cluster memberships do not change is equivalent to
# update until centers doesn't move
def has_converged(new_mu, mu):
  # convert coords to tuples and make a set for easy comparison 
  return set([tuple(x) for x in new_mu]) == set([tuple(x) for x in mu])

def kmeans(X, k, init, n=16):
  mu = initialize_uniform_random(X, k) if init == 'random' else initialize_kmeans_plusplus(X, k)
  for _ in range(16):
    clusters = cluster(X, mu)
    new_mu = center(mu, clusters)
    if has_converged(new_mu, mu):
      return clusters, mu
    mu = new_mu
  return clusters, mu

def calculate_error(clusters, mu):
  error = 0
  for key in clusters.keys():
    for x in clusters[key]:
      error += np.linalg.norm(x-mu[key])
  return error  

def plot_2d(clusters, init):
  fig, ax = plt.subplots()
  title = 'uniform random initialization' if init == 'random' else 'k-means++ initialization'
  ax.set_title(f'k-means - {title}')

  for key in clusters.keys():
    x = [a[0] for a in clusters[key]]
    y = [a[1] for a in clusters[key]]
    ax.scatter(x, y, marker='.')

  plt.savefig(f"images/{title.replace(' ', '_')}.png")

def process_data(src):
  data = None
  with open(src, 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    data = list(reader)
  
  data = [[float(y) for y in x] for x in data]
  return [np.array(x, dtype='float') for x in data]

if __name__ == '__main__':
  def usage():
    print('usage: python lloyds.py <k> <initialization> <data> --plot')
    sys.exit(1)

  if len(sys.argv) < 4:
    usage()

  k = int(sys.argv[1])
  init = sys.argv[2]
  data_src = sys.argv[3]
  plot_results = '--plot' in sys.argv

  data = process_data(data_src)
  dimension = len(data[0])

  clusters, mu = kmeans(data, k, init)

  print(calculate_error(clusters, mu))

  if plot_results and dimension == 2:
    plot_2d(clusters, init)
