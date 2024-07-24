import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
import pandas as pd

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

data = pd.read_csv('Space_Corrected.csv')
data = np.array([data['Datum'], data['Status_Rocket']])
data = np.stack((data[0], data[1]), axis=-1)


# plt.scatter(data.T[0], data.T[1], **plot_kwds)
# frame = plt.gca()
# frame.axes.get_xaxis().set_visible(False)
# frame.axes.get_yaxis().set_visible(False)
plt.xlabel("a")
plt.ylabel("b")


def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    # frame.axes.get_xaxis().set_visible(False)
    # frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    print('Clustering took {:.2f} s'.format(end_time - start_time))


#plot_clusters(data, cluster.KMeans, (), {'n_clusters':4})
#plot_clusters(data, cluster.AffinityPropagation, (), {'preference':-1000, 'damping':0.95})
#plot_clusters(data, cluster.MeanShift, (100,), {'cluster_all':False})
#plot_clusters(data, cluster.AgglomerativeClustering, (), {'n_clusters':6, 'linkage':'ward'})
#plot_clusters(data, cluster.DBSCAN, (), {'eps':30})
import hdbscan
plot_clusters(data, hdbscan.HDBSCAN, (), {'min_cluster_size':4})
plt.show()



