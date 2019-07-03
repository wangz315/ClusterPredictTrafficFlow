import networkx as nx # graph package
import numpy as np
import pandas as pd # csv handler
import random
import csv
import matplotlib as mpl
mpl.use('Agg')
from timeit import default_timer as timer
from sklearn import metrics
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.cluster import AffinityPropagation # AP cluster class
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from colorama import Fore
from colorama import Style
from timeit import default_timer as timer


def read_graph():
	G = nx.read_gml('data/citypulse_static.gml')

	# give index to each node
	index = 0
	for n in G.nodes():
		G.nodes[n]['index'] = index
		index += 1

	# give capacity (i.e. flow at time t) to each edge
	for i, j, d in G.edges(data = True):
		df = pd.read_csv('data/train/train_%s.csv' % (d['report']))
		# df = df[df['STAMP'] == stamp]
		d['capacity'] = df['FLOW'].values
		d['length'] = int(d['length'])

	return G


# stamp = index in stamps
def similarity_matrix(G, stamp):
	size = len(G)	
	# similarity matrix
	sm = np.zeros((size, size))
	
	for s in G.nodes():
		i = G.nodes[s]['index']
		for t in G.nodes():
			j = G.nodes[t]['index']
			sm[i][j] = sij(s, t, G, stamp)
			
		# if there is no flow at all, find the closest node
		if np.sum(sm[i]) <= 1:
			oute = G.out_edges(s, data = True)
			shortd = float('inf')
			shortnode = None
			for x, y, d in oute:
				if d['length'] < shortd:
					shortnode = y
					
			j = G.nodes[shortnode]['index']
			for a in range(size):
				if i != a:
					sm[i][a] = 0
			sm[i][j] = 1
			

	X = np.zeros((size, size))
	for s in G.nodes():
		for t in G.nodes():
			i = G.nodes[s]['index']
			j = G.nodes[t]['index']
			
			X[i][j] = (sm[i][j] + sm[j][i])/2
	
	return X


def sij(s, t, G, stamp):

	if s != t:  
		ine = G.in_edges(t, data = True)
		totalflow = 0
		for i, j, d in ine:
			totalflow += d['capacity'][stamp]
		
		path = nx.shortest_path(G, source=s, target=t, weight = 'length')
		
		flow = 65535 # the max possible flow
		for i in range(len(path) - 1):
			prev = path[i]
			curr = path[i+1]
			
			ed = G[prev][curr]
			
			if ed['capacity'][stamp] < flow:
				flow = ed['capacity'][stamp]
		
		if totalflow > 0:
			return flow/totalflow
		else:
			return 0
		
	else:            
		return 1


# =============================================================================
# draw graph
# =============================================================================
def plot_cluster(G, labels_):
	
	mpl.pyplot.figure(figsize=(8,6))
	labels = {}
	for idx, node in enumerate(G.nodes()):
		labels[node] = int(labels_[idx])
	pos=nx.get_node_attributes(G,'pos')
	nx.draw_networkx_nodes(G, pos, node_size=60, node_color = [x for x in labels_])
	nx.draw_networkx_edges(G, pos, arrows=False)
	nx.draw_networkx_labels(G, pos, labels, font_size=8)
	
	
	#mpl.pyplot.show()
	
	mpl.pyplot.savefig("images/cluster.png", dpi=100)


# =============================================================================
# compute cluster mean
# =============================================================================
def cluster_mean(sm, labels):
	membership = dict()
	
	for i in range(len(labels)):
		if labels[i] in membership:
			membership[labels[i]].append(i)
		else:
			membership[labels[i]] = []
			membership[labels[i]].append(i)

	simi = 0
	comb = 0 # combination size
	for k, v in membership.items():
		for i in range(len(v)):
			for j in range(i + 1, len(v)):
				simi += sm[v[i]][v[j]]
				comb += 1
				
	return simi/comb


def cluster_stamp():
	stamps = []
	month = 30

	for i in range(month): # days
		for j in range(24): # hours
			for k in range(12): # mins
				stamp = '2014-03-{:02d}T{:02d}:{:02d}:00'.format(i+1, j, k*5)
				stamps.append(stamp)

	for i in range(month): # days
		for j in range(24): # hours
			for k in range(12): # mins
				stamp = '2014-04-{:02d}T{:02d}:{:02d}:00'.format(i+1, j, k*5)
				stamps.append(stamp)

	start = timer()
	G = read_graph()
	end = timer()

	print('Read Graph Done: {}s.'.format(end - start))

	filename = 'data/cluster_stamp.csv'
	open('data/cluster_stamp.csv', 'w').close()

	csvfile = open(filename, 'wb')
	writer = csv.writer(csvfile)

	for i in range(len(stamps)):
		start = timer()
		X = similarity_matrix(G, i)
		af = AffinityPropagation(preference = np.min(X), affinity = 'precomputed').fit(X)
		writer.writerow(af.labels_)
		end = timer()
		print('{}/{}: {}s.'.format(i+1, len(stamps), end - start))

	csvfile.close()


def cluster_day():
	days = 60

	start = timer()
	G = read_graph()
	for i, j, d in G.edges(data = True):
		dayflow = [ sum(d['capacity'][x:x+12*24]) for x in range(0, len(d['capacity']), 12*24)]
		d['capacity'] = dayflow
	end = timer()

	print('Read Graph Done: {}s.'.format(end - start))

	filename = 'data/cluster_day.csv'
	open('data/cluster_day.csv', 'w').close()

	csvfile = open(filename, 'wb')
	writer = csv.writer(csvfile)

	for i in range(days):
		start = timer()
		X = similarity_matrix(G, i)
		af = AffinityPropagation(preference = np.min(X), affinity = 'precomputed').fit(X)
		writer.writerow(af.labels_)
		end = timer()
		print('{}/{}: {}s.'.format(i+1, days, end - start))

	csvfile.close()


def cluster_month():
	month = 30
	G = nx.read_gml('data/citypulse_static.gml')
	X = np.zeros((len(G), len(G)))
	clusters = []

	filename = 'data/cluster_day.csv'
	csvfile = open(filename, 'rb')
	reader = csv.reader(csvfile)

	for row in reader:
		clusters.append(row)

	csvfile.close()

	for i in range(month):
		for j in range(len(G)):
			for k in range(j+1, len(G)):
				if clusters[i][j] == clusters[i][k]:
					X[j][k] += 1
					X[k][j] += 1

	X = X / X.max()
	for i in range(len(G)):
		X[i][i] = 1

	af = AffinityPropagation(preference = np.min(X), affinity = 'precomputed').fit(X)
	n_clusters_ = len(af.cluster_centers_indices_)
	print('Estimated number of clusters: %d' % (n_clusters_))
	
	print("Silhouette Coefficient: %0.3f"
	  % metrics.silhouette_score(1-X, af.labels_, metric='precomputed'))

	mean = cluster_mean(X, af.labels_)
	print('Estimated mean similarity clusters: %f' % (mean))
	plot_cluster(G, af.labels_)

	filename = 'data/cluster.csv'
	open('data/cluster.csv', 'w').close()

	csvfile = open(filename, 'wb')
	writer = csv.writer(csvfile)
	writer.writerow(af.labels_)
	csvfile.close()


def apcluster(G, X):
	print(Fore.CYAN + 'AP Clustering...')
	
	start = timer()
	af = AffinityPropagation(preference = np.min(X), affinity = 'precomputed').fit(X)
	end = timer()
		
	n_clusters_ = len(af.cluster_centers_indices_)
	print('Estimated number of clusters: %d' % (n_clusters_))
	
	print("Silhouette Coefficient: %0.3f"
	  % metrics.silhouette_score(1-X, af.labels_, metric='precomputed'))
	
	
	mean = cluster_mean(X, af.labels_)
	print('Estimated mean similarity clusters: %f' % (mean))
	
	print('Time cost of AP is %f' % (end - start))
	plot_cluster(G, af.labels_)
	
	print(Style.RESET_ALL)
	
	return n_clusters_


def kmedoidcluster(G, X, n = 10):
	print(Fore.MAGENTA + 'K-medoids Clustering...')
	
	initial_medoids = random.sample(range(1, 100), n)
	
	start = timer()
	kmedoids_instance = kmedoids(1-X, initial_medoids, data_type='distance_matrix')
	kmedoids_instance.process()
	end = timer()
	
	clusters = np.array(kmedoids_instance.get_clusters())
	
	labels = np.zeros(len(G))
		
	for i in range(len(clusters)):
		for x in clusters[i]:
			labels[x] = i
			
	n_clusters_ = len(set(labels))
			
	print('Estimated number of clusters: %d' % (n_clusters_))
			
	print("Silhouette Coefficient: %0.3f"
	  % metrics.silhouette_score(1-X, labels, metric='precomputed'))
			
	mean = cluster_mean(X, labels)
	print('Estimated mean similarity clusters: %f' % (mean))
  
	print('Time cost of k-medoid is %f' % (end - start))

	print(Style.RESET_ALL)


def DBSCANcluster(G, X):
	print(Fore.GREEN + 'DBSCAN Clustering...')
	
	start = timer()
	db = DBSCAN(metric = 'precomputed', min_samples = 1, eps = 0.5).fit(1-X)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	end = timer()
	labels = db.labels_
	
	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)

	print('Estimated number of clusters: %d' % n_clusters_)
	print('Estimated number of noise points: %d' % n_noise_)
	
	mean = cluster_mean(X, labels)
	print('Estimated mean similarity clusters: %f' % (mean))
	
	if n_clusters_ > 1:
		print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(1-X, labels))
	else:
		print("Silhouette Coefficient does not apply: only 1 cluster")
		
	print('Time cost of DBSCAN is %f' % (end - start))
	
	print(Style.RESET_ALL)


def aggcluster(G, X, n = 10):
	print(Fore.YELLOW + 'Agglomerative Clustering...')
	
	start = timer()
	model = AgglomerativeClustering(affinity='precomputed', linkage='average', n_clusters=n)
	model.fit(1-X)
	end = timer()
	labels = model.labels_
	
	print('Estimated number of clusters: %d' % n)
	
	print("Silhouette Coefficient: %0.3f"
	  % metrics.silhouette_score(1-X, labels, metric='precomputed'))
	
	mean = cluster_mean(X, labels)
	print('Estimated mean similarity clusters: %f' % (mean))
	
	print('Time cost of Agglomerative is %f' % (end - start))
	
	print(Style.RESET_ALL)
