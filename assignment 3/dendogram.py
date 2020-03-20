# references
#    https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
#    https://scikit-learn.org/stable/auto_examples/cluster/plot_digits_linkage.html#sphx-glr-auto-examples-cluster-plot-digits-linkage-py
#    https://scikit-learn.org/stable/auto_examples/cluster/plot_linkage_comparison.html#sphx-glr-auto-examples-cluster-plot-linkage-comparison-py
#    https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/

import sys, csv
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def process_data(src):
  data = None
  with open(src, 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    data = list(reader)
  
  data = [[float(y) for y in x] for x in data]
  return [np.array(x, dtype='float') for x in data]

if __name__ == '__main__':
  def usage():
    print('usage: python dendrogram.py <data>')
    sys.exit(1)

  if len(sys.argv) < 2:
    usage()

  data = process_data(sys.argv[1])
  dimension = len(data[0])

  for link in ['single', 'average']:
    plt.figure()
    plt.title(f'Dendogram - {link.capitalize()} Linkage')
    dendrogram(linkage(data, method=link), truncate_mode='lastp')
    plt.savefig(f'images/dendograms/{link}_{dimension}.png')

  # model = AgglomerativeClustering(linkage='single', n_clusters=1).fit(data)
  # plot_dendrogram(model, truncate_mode='lastp')
  # plot_clustering(data, model.labels_, "%s linkage" % linkage)
  
  