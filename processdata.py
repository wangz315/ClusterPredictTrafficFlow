import os
import collections
import csv
import progressbar
import pandas as pd # csv handler
import networkx as nx # graph package
import numpy as np
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def process_data(train, test, lags):
	# train: dict
	# test: dict

	X_train, y_train, X_test, y_test = [], [], [], []
	trainfeatures, testfeatures = [], []

	for k, v in train.items():
		trainfeatures.append(v)

	for k, v in test.items():
		testfeatures.append(v)

	trainfeatures = np.array(trainfeatures).T
	testfeatures = np.array(testfeatures).T

	scaler_train = MinMaxScaler(feature_range=(0,1)).fit(trainfeatures)
	trainfeatures = scaler_train.transform(trainfeatures)

	scaler_test = MinMaxScaler(feature_range=(0,1)).fit(testfeatures)
	testfeatures = scaler_test.transform(testfeatures)

	X, y = [], []
	for i in range(trainfeatures.shape[0]-lags-1):
		t=[]
		for j in range(lags):
			t.append(trainfeatures[[(i+j)], :])
		X.append(t)
		y.append(trainfeatures[i+lags,:])

	X, y = np.array(X), np.array(y)

	X_train = X.reshape(X.shape[0], lags, X.shape[3])
	y_train = y


	X, y = [], []
	for i in range(testfeatures.shape[0]-lags-1):
		t=[]
		for j in range(lags):
			t.append(testfeatures[[(i+j)], :])
		X.append(t)
		y.append(testfeatures[i+lags,:])

	X, y = np.array(X), np.array(y)

	X_test = X.reshape(X.shape[0], lags, X.shape[3])
	y_test = y

	return X_train, y_train, X_test, y_test, scaler_train, scaler_test


def process_cluster():

	G = nx.read_gml('data/citypulse_static.gml')
	filename = 'data/cluster.csv'
	csvfile = open(filename, 'rb')
	reader = csv.reader(csvfile)

	for row in reader:
		clusters = row

	csvfile.close()

	clusters = map(int, clusters)

	traindict = []
	testdict = []

	for i in range(max(clusters)+1):
		traindict.append(OrderedDict())
		testdict.append(OrderedDict())

	index = 0

	print('Gathering data:')
	bar = progressbar.ProgressBar(maxval=len(G), \
		widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()

	for n in G.nodes():
		train = OrderedDict()
		test = OrderedDict()
		G.nodes[n]['index'] = index
		inedges = G.in_edges(n, data = True)
		outedges = G.out_edges(n, data = True)

		for i, j, e in inedges:
			dftrain = pd.read_csv('data/train/train_%s.csv' % (e['report']))
			train[e['report']] = dftrain['FLOW'].values

			dftest = pd.read_csv('data/test/test_%s.csv' % (e['report']))
			test[e['report']] = dftest['FLOW'].values

		for i, j, e in outedges:
			dftrain = pd.read_csv('data/train/train_%s.csv' % (e['report']))
			train[e['report']] = dftrain['FLOW'].values

			dftest = pd.read_csv('data/test/test_%s.csv' % (e['report']))
			test[e['report']] = dftest['FLOW'].values


		bar.update(index+1)
		# print('Gathering data: %d/%d' % (index, len(G)))
		traindict[clusters[index]].update(train)
		testdict[clusters[index]].update(test)

		index += 1

	bar.finish()

	return traindict, testdict, clusters


def index_of_stamp(stamp, report):
	for i in range(len(report)):
		if stamp == report[i]:
			return i
	return -1


def process_static(debug = False):
	G = nx.DiGraph()

	metaData = pd.read_csv('data/raw/trafficMetaData.csv')

	# G: direct network, distance (length) on edge
	# find all nodes
	nodes = dict()
	for i in range(len(metaData['POINT_1_STREET'])):
		nodes[metaData['POINT_1_NAME'][i]] = (float(metaData['POINT_1_LAT'][i]), 
			float(metaData['POINT_1_LNG'][i]))
		nodes[metaData['POINT_2_NAME'][i]] = (float(metaData['POINT_2_LAT'][i]), 
			float(metaData['POINT_2_LNG'][i]))

	# insert nodes to the graph
	for node in nodes:
		G.add_node(str(node),pos=(nodes[node][0],nodes[node][1]))

	# add edges
	for i in range(len(metaData['POINT_1_STREET'])):
		s = metaData['POINT_1_NAME'][i]
		t = metaData['POINT_2_NAME'][i]
		l = int(metaData['DISTANCE_IN_METERS'][i])
		rep = metaData['REPORT_ID'][i]

		G.add_edge(str(s), str(t), length = str(l), report = str(rep))

	nx.write_gml(G, 'data/citypulse_static.gml')

	return G


def process_train(G):
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

	for s, t, d in G.edges(data=True):
		flow = collections.OrderedDict()
		# flow['stamp'] = 'flow'
		if os.path.exists('data/raw/traffic_feb_june/trafficData%s.csv' % d['report']):
			report = pd.read_csv('data/raw/traffic_feb_june/trafficData%s.csv' % d['report'])

			stamplength = len(stamps)
			stampcount = 0
			filecount = index_of_stamp('2014-03-01T00:00:00', report['TIMESTAMP']) # start of March 1st
			
			if filecount >= 0:
				while stampcount < stamplength:
					
						if report['TIMESTAMP'][filecount] == stamps[stampcount]:
							flow[stamps[stampcount]] = report['vehicleCount'][filecount]
							stampcount += 1
							filecount += 1
						else:
							flow[stamps[stampcount]] = 0
							stampcount += 1
			else:
				# print('March 1st does not found in report %s' % (d['report']))
				while filecount < 0:
					flow[stamps[stampcount]] = 0
					stampcount += 1
					filecount = index_of_stamp(stamps[stampcount], report['TIMESTAMP'])

				while stampcount < stamplength:
						if report['TIMESTAMP'][filecount] == stamps[stampcount]:
							flow[stamps[stampcount]] = report['vehicleCount'][filecount]
							stampcount += 1
							filecount += 1
						else:
							flow[stamps[stampcount]] = 0
							stampcount += 1

		else:
			print('report %s not found' % (d['report']))

		keys = flow.keys()
		values = flow.values()
		combined = np.vstack((keys, values)).T
		df = pd.DataFrame(combined, columns=['STAMP', 'FLOW'])
		# df = df.T
		df.to_csv('data/train/train_%s.csv' % (d['report']), index=False)


def process_test(G):
	stamps = []
	month = 30

	for i in range(month): # days
		for j in range(24): # hours
			for k in range(12): # mins
				stamp = '2014-05-{:02d}T{:02d}:{:02d}:00'.format(i+1, j, k*5)
				stamps.append(stamp)

	for s, t, d in G.edges(data=True):
		flow = collections.OrderedDict()
		# flow['stamp'] = 'flow'
		if os.path.exists('data/raw/traffic_feb_june/trafficData%s.csv' % d['report']):
			report = pd.read_csv('data/raw/traffic_feb_june/trafficData%s.csv' % d['report'])

			stamplength = len(stamps)
			stampcount = 0
			filecount = index_of_stamp('2014-05-01T00:00:00', report['TIMESTAMP']) # start of May 1st
			
			if filecount >= 0:
				while stampcount < stamplength:
					
						if report['TIMESTAMP'][filecount] == stamps[stampcount]:
							flow[stamps[stampcount]] = report['vehicleCount'][filecount]
							stampcount += 1
							filecount += 1
						else:
							flow[stamps[stampcount]] = 0
							stampcount += 1
			else:
				# print('March 1st does not found in report %s' % (d['report']))
				while filecount < 0:
					flow[stamps[stampcount]] = 0
					stampcount += 1
					filecount = index_of_stamp(stamps[stampcount], report['TIMESTAMP'])

				while stampcount < stamplength:
						if report['TIMESTAMP'][filecount] == stamps[stampcount]:
							flow[stamps[stampcount]] = report['vehicleCount'][filecount]
							stampcount += 1
							filecount += 1
						else:
							flow[stamps[stampcount]] = 0
							stampcount += 1

		else:
			print('report %s not found' % (d['report']))

		keys = flow.keys()
		values = flow.values()
		combined = np.vstack((keys, values)).T
		df = pd.DataFrame(combined, columns=['STAMP', 'FLOW'])
		# df = df.T
		df.to_csv('data/test/test_%s.csv' % (d['report']), index=False)


def main():
	G = process_static()
	process_train(G)
	process_test(G)


if __name__ == '__main__':
	main()
