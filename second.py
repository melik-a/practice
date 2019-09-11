#Agglomerative кластеризация
import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=2)
agl = AgglomerativeClustering(n_clusters=3)
y_ag = agl.fit_predict(X)

plt.scatter(X[y_ag==0,0],
            X[y_ag==0,1],
            c='lightblue',
            marker='o',
            s=40,
            edgecolors='black',
            label='кластер 1')
plt.scatter(X[y_ag==1,0],
            X[y_ag==1,1],
            c='red',
            marker='s',
            s=40,
            edgecolors='black',
            label='кластер 2')
plt.scatter(X[y_ag==2,0],
            X[y_ag==2,1],
            c='green',
            marker='v',
            s=40,
            edgecolors='black',
            label='кластер 3')
plt.grid()
#plt.show()

from scipy.cluster.hierarchy import dendrogram, ward
fig = plt.figure()
linkage_array = ward(X)

dendrogram(linkage_array)
plt.legend()
plt.show()
