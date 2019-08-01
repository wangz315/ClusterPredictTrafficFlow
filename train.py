import sys
import warnings
import argparse
import csv
import progressbar
import networkx as nx
import numpy as np
import pandas as pd
from collections import OrderedDict
from processdata import process_data
from processdata import process_cluster
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")


def get_lstm(units, shape):
	"""LSTM(Long Short-Term Memory)
	Build LSTM Model.

	# Arguments
		units: List(int), number of input, output and hidden units.
	# Returns
		model: Model, nn model.
	"""

	model = Sequential()
	model.add(LSTM(units[1], input_shape=(shape[1], shape[2]), return_sequences=True))
	model.add(LSTM(units[2]))
	model.add(Dropout(0.2))
	model.add(Dense(shape[2], activation='sigmoid'))

	return model


def train_model(model, X_train, y_train, name, config, cid):
	"""train
	train a single model.

	# Arguments
		model: Model, NN model to train.
		X_train: ndarray(number, lags), Input data for train.
		y_train: ndarray(number, ), result data for train.
		name: String, name of model.
	config: Dict, parameter for train.
	"""

	model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
	# early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
	hist = model.fit(
		X_train, y_train,
		batch_size=config["batch"],
		epochs=config["epochs"],
		validation_split=0.05)

	model.save('model/' + name + '_cluster_' + cid + '.h5')
	df = pd.DataFrame.from_dict(hist.history)
	df.to_csv('model/' + name + '_cluster_' + cid + '_loss.csv', encoding='utf-8', index=False)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--model",
		default="lstm",
		help="Model to train.")
	args = parser.parse_args()

	lag = 12
	config = {"batch": 256, "epochs": 600}

	traindict, testdict, clusters = process_cluster()

	for i in range(max(clusters)+1):
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

		print('Training cluster: %d/%d' % (i, max(clusters)+1))
		X_train, y_train, _, _, _, _ = process_data(train, test, lag)
		m = get_lstm([12, 64, 64, 1], X_train.shape)
		cid = str(i)
		train_model(m, X_train, y_train, args.model, config, cid)

		break


if __name__ == '__main__':
	main()





















