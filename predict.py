import math
import warnings
import csv
import os
import progressbar
import networkx as nx
import numpy as np
import pandas as pd
from collections import OrderedDict
from processdata import process_data
from processdata import process_cluster
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def MAPE(y_true, y_pred):
	"""Mean Absolute Percentage Error
	Calculate the mape.

	# Arguments
		y_true: List/ndarray, ture data.
		y_pred: List/ndarray, predicted data.
	# Returns
		mape: Double, result data for train.
	"""

	y = [x for x in y_true if x > 0]
	y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

	num = len(y_pred)
	sums = 0

	for i in range(num):
		tmp = abs(y[i] - y_pred[i]) / y[i]
		sums += tmp

	mape = sums * (100 / num)

	return mape


def eva_regress(y_true, y_pred, cid):
	"""Evaluation
	evaluate the predicted results.

	# Arguments
		y_true: List/ndarray, ture data.
		y_pred: List/ndarray, predicted data.
	"""

	mape = MAPE(y_true, y_pred)
	vs = metrics.explained_variance_score(y_true, y_pred)
	mae = metrics.mean_absolute_error(y_true, y_pred)
	mse = metrics.mean_squared_error(y_true, y_pred)
	r2 = metrics.r2_score(y_true, y_pred)
	print('explained_variance_score:%f' % vs)
	print('mape:%f%%' % mape)
	print('mae:%f' % mae)
	print('mse:%f' % mse)
	print('rmse:%f' % math.sqrt(mse))
	print('r2:%f' % r2)

	d = dict()
	d['explained_variance_score'] = [vs]
	d['mape'] = [mape]
	d['mae'] = [mae]
	d['mse'] = [mse]
	d['rmse'] = [math.sqrt(mse)]
	d['r2'] = [r2]

	df = pd.DataFrame(d)

	df.to_csv('data/results/cluster_%d.csv' % (cid), index=False)


def plot_results(y_true, y_preds, names, cid):
	"""Plot
	Plot the true data and predicted data.

	# Arguments
		y_true: List/ndarray, ture data.
		y_pred: List/ndarray, predicted data.
		names: List, Method names.
	"""

	d = '2014-03-01T00:00:00'
	x = pd.date_range(d, periods=288, freq='5min')

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(x, y_true, label='True Data')
	for name, y_pred in zip(names, y_preds):
		ax.plot(x, y_pred, label=name)

	plt.legend()
	plt.grid(True)
	plt.xlabel('Time of Day')
	plt.ylabel('Flow')

	date_format = mpl.dates.DateFormatter("%H:%M")
	ax.xaxis.set_major_formatter(date_format)
	fig.autofmt_xdate()

	# plt.show()
	plt.savefig('images/plot_cluster_' + cid + '_results.png')


def predict_cluster():
	traindict, testdict, clusters = process_cluster()

	lag = 12
	for i in range(max(clusters)+1):
		cid = str(i)

		lstm = load_model('model/lstm_cluster_' + cid + '.h5')
		models = [lstm]
		names = ['LSTM']

		train = traindict[i]
		test = testdict[i]

		temptrain = dict()
		for k,v in train.items():
			temptrain[k] = v
			break

		train = temptrain

		temptest = dict()
		for k,v in test.items():
			temptest[k] = v
			break

		test = temptest

		print('Predicting cluster: %d/%d' % (i, max(clusters)+1))
		_, _, X_test, y_test, scaler_train, scaler_test = process_data(train, test, lag)
		
		y_preds = []
		for name, model in zip(names, models):
			# file = 'images/' + name + '_cluster_' + cid + '.png'
			# plot_model(model, to_file=file, show_shapes=True)
			predicted = model.predict(X_test)
			predicted = predicted.T[0]
			y_test = y_test.T[0]
			y_preds.append(predicted[:288])
			print(name)
			eva_regress(y_test, predicted, i)

		plot_results(y_test[:288], y_preds, names, cid)


		break


def main():
	predict_cluster()


if __name__ == '__main__':
	main()



















