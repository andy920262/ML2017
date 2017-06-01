import numpy as np
import sys
import keras
from keras.models import Model
from keras.layers import Embedding, Input, Dropout, Flatten, BatchNormalization

dir_path = sys.argv[1]
output_path = sys.argv[2]

def load_data(path, mode):
	data = []
	ids = []
	with open(path, 'r', encoding = 'latin1') as file:
		file.readline()
		for line in file:
			if mode == 'train' or mode == 'test':
				data.append(line[:-1].split(',')[1:])
			if mode == 'ids':
				data.append(line[:-1].split(',')[0])
	data = np.array(data, dtype = np.int64)
	if mode == 'train':
		data = (data[:, :-1], data[:, -1])
	return data


if __name__ == '__main__':
	#x_train, y_train = load_data(dir_path + 'train.csv', 'train')
	x_test = load_data(dir_path + 'test.csv', 'test')
	movie_ids = load_data(dir_path + 'movies.csv', 'ids')
	user_ids = load_data(dir_path + 'users.csv', 'ids')

	#user_train = np.array([np.where(user_ids == x) for x in x_train[:, 0]]).ravel()
	#movie_train = np.array([np.where(movie_ids == x) for x in x_train[:, 1]]).ravel()
	user_test = np.array([np.where(user_ids == x) for x in x_test[:, 0]]).ravel()
	movie_test = np.array([np.where(movie_ids == x) for x in x_test[:, 1]]).ravel()

	#idx = np.arange(user_train.shape[0])
	#np.random.shuffle(idx)
	#user_train, movie_train = user_train[idx], movie_train[idx]
	#y_train = y_train[idx]

	#x_train = [user_train, movie_train]
	#x_valid = [user_train[:50000], movie_train[:50000]]
	#y_valid = y_train[:50000]
	#x_train = [user_train[50000:], movie_train[50000:]]
	#y_train = y_train[50000:]

	# Build model
	user_input = Input(shape = (1,))
	user_vec = Embedding(user_ids.shape[0] + 1, 100)(user_input)
	user_vec = Flatten()(user_vec)
	user_vec = Dropout(0.2)(user_vec)

	movie_input = Input(shape = (1,))
	movie_vec = Embedding(movie_ids.shape[0] + 1, 100)(movie_input)
	movie_vec = Flatten()(movie_vec)
	movie_vec = Dropout(0.2)(movie_vec)

	out = keras.layers.dot([user_vec, movie_vec], 1)

	model = Model(inputs = [user_input, movie_input], outputs = out)
	model.compile(optimizer = 'adam', loss = 'mse')
	#model.fit(x_train, y_train, epochs = 50, batch_size = 10000, validation_data = (x_valid, y_valid))
	#model.fit(x_train, y_train, epochs = 50, batch_size = 10000)
	#model.save_weights('model.hdf5')
	model.load_weights('best.hdf5')

	predict = model.predict([user_test, movie_test])
	out = open(output_path, 'w')
	print('TestDataID,Rating', file = out)
	for i, x in enumerate(predict):
		print('{},{}'.format(i + 1, x[0]), file = out)
	exit()
