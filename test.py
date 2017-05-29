import sys
# print(__doc__)

print len(sys.argv)
if len(sys.argv) < 3:
	eps_set = 0.3
	min_samples_set = 3
else: 
	eps_set = float(sys.argv[1])
	min_samples_set = int(sys.argv[2])

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
#from sklearn.datasets.samples_generator import make_blobs
#from sklearn.preprocessing import StandardScaler

def read_data():
	import numpy as np
	f = open('data', 'r')
	f.readline()
	dt = []
	for line in f:
	    splt = line.split()
	    dt.append([float(splt[1]), float(splt[2])])
	return np.array(dt)

Xs = read_data()

#print "--------X--------"
#print Xs

#print "--------labels_true--------"
#print labels_true


X = Xs # StandardScaler().fit_transform(Xs)
#print "--------X transformed--------"
#print X

db = DBSCAN(eps=eps_set, min_samples=min_samples_set).fit(X)
#print "-------- db --------"
#print db
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

#print "--------core_sample_indices--------"
#print db.core_sample_indices_

#print "--------labels--------"
#print labels

#print max(labels)

result = []
for j in range(0, max(labels)+1):
	indices = [i for i, x in enumerate(labels) if x == j]
	new_indices = [x+1 for x in indices]
	result.append(new_indices)
	
print result

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = Xs[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = Xs[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()