import pandas 
import math
 
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential
from keras.layers import Conv1D, Dense, LSTM, RepeatVector, TimeDistributed, Bidirectional, Flatten, Dropout, Reshape
from keras.callbacks import ModelCheckpoint

import numpy as np
from numpy import array

import matplotlib.pyplot as plt

import random
import time

from config import *
from dataPreprocessing import load_data

for appliance_name in APPLIANCE_CONFIG.keys():
	# appliance_name = 'kettle'
	if appliance_name != 'kettle':
		continue
	print(appliance_name)
	X_train, X_test, y_train, y_test = load_data(appliance_name)
	sequence_length = math.ceil(APPLIANCE_CONFIG[appliance_name]['window_width'] / SAMPLE_WIDTH)
	max_power = APPLIANCE_CONFIG[appliance_name]['max_power']

	X_train = X_train.values
	X_test = X_test.values
	y_train = y_train.values
	y_test = y_test.values

	X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
	X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
	y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
	y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))

	print('Training Data:')
	print(X_train.shape)
	print(y_train.shape)

	print('Test Data:')
	print(X_test.shape)
	print(y_test.shape)
	# models

	def RNN_model():
		'''Creates the RNN module described in the paper
		'''
		model = Sequential()

		# 1D Conv
		model.add(Conv1D(16, 4, activation="linear", input_shape=(sequence_length, 1), padding="same", strides=1))

		#Bi-directional LSTMs
		model.add(Bidirectional(LSTM(128, return_sequences=True, stateful=False), merge_mode='concat'))
		model.add(Bidirectional(LSTM(256, return_sequences=True, stateful=False), merge_mode='concat'))

		# Fully Connected Layers
		model.add(Dense(128, activation='tanh'))
		model.add(Dense(1, activation='linear'))

		model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
		# model.summary()
		# plot_model(model, to_file='model.png', show_shapes=True)

		return model

	def DAE_model():
		'''Creates the Auto encoder module described in the paper
		'''
		model = Sequential()

		# 1D Conv
		model.add(Conv1D(8, 4, activation="linear", input_shape=(sequence_length, 1), padding="same", strides=1))
		model.add(Flatten())

		# Fully Connected Layers
		model.add(Dropout(0.2))
		model.add(Dense((sequence_length-0)*8, activation='relu'))

		model.add(Dropout(0.2))
		model.add(Dense(128, activation='relu'))

		model.add(Dropout(0.2))
		model.add(Dense((sequence_length-0)*8, activation='relu'))

		model.add(Dropout(0.2))

		# 1D Conv
		model.add(Reshape(((sequence_length-0), 8)))
		model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))

		model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
		# model.summary()
		# plot_model(model, to_file='model.png', show_shapes=True)

		return model

	# ! models

	# model = RNN_model()


	# trainning
	list_model = [RNN_model(), DAE_model()]
	list_pred = []
	for index, model in enumerate(list_model):

		model.summary()
		BATCH_SIZE = 128
		num_epochs = 20
		
		model.load_weights("model_" + appliance_name + '_' +  str(index) + '_' + str(num_epochs) + 'epo.hdf5')

		'''
		def RMSE_seq(list_pred, list_truth):
			sq_error = []
			for i in range(list_pred.shape[0]):
				pred = [ x[0] for x in list_pred[i]]
				truth = [ x[0] for x in list_truth[i]]
				
				for j in range(len(pred)):
					sq_error.append((pred[j] - truth[j]) ** 2)
			
			return np.asarray(sq_error).mean() ** (1/2)  * max
		'''


		preds = model.predict(X_test, verbose=0)
		# y_test = y_test.flatten()
		list_pred.append(preds)
		def mean_abs_err(pred, ground_truth):
			# sum of error / number of sample
			sum_abs_error = 0
			for i in range(pred.shape[0]):
				for j in range(pred.shape[1]):
					sum_abs_error += abs(pred[i][j][0] - ground_truth[i][j][0])
					
			return sum_abs_error/(pred.shape[0] * pred.shape[1])



		def PTECA(pred, ground_truth, aggre_data):
			sum_abs_error = 0
			sum_aggre = 0
			for i in range(pred.shape[0]):
				for j in range(pred.shape[1]):
					sum_abs_error += abs(pred[i][j][0] - ground_truth[i][j][0])
					sum_aggre += aggre_data[i][j][0]
					
			return 1 - sum_abs_error/ (2 * sum_aggre) 
		
		mae = mean_abs_err(preds, y_val) * max
		print('MAE is {}'.format(mae))

		pteca = PTECA(preds, y_val, x_val) 
		print('pteca is {}'.format(pteca))
		# pteca = PTECA(preds, y_test, X_test) 
		# print('pteca is {}'.format(pteca))
		
	
	count  = 0
	preds = list_pred[0]
	
	for index, list in enumerate(preds):
		line1, = plt.plot([ i for i in range(sequence_length)], [ item[0]  for item in list], label='LSTM')
		line2, = plt.plot([ i for i in range(sequence_length)], [ item[0]  for item in list_pred[1][index]], label='AE')
		line3, = plt.plot([ i for i in range(sequence_length)], [ item[0]  for item in y_test[index]], label='Appliance')
		# line4, = plt.plot([ i for i in range(sequence_length)], [ item[0]  for item in X_test[index]], label='')
		
		# plt.legend({'LSTM', 'AE',  'Appliance', 'Main'})
		plt.legend(handles=[line1, line2, line3])
		plt.xlabel('Relative Time (6s)')
		plt.ylabel('Power (kw/h)')
		plt.show()
		count += 1
		if count  == 5:
			break
		

