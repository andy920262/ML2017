from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.models import Sequential, load_model
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import RegexpTokenizer
import numpy as np
import pickle
import itertools
import sys

TEXT_LEN = 310
THRES = 0.2
EMBEDDING_DIM = 100
EPOCHS = 500
BATCH_SIZE = 128

def load_train_data(path):
	data = []
	label = []
	tokenizer = RegexpTokenizer(r'[A-Za-z]+')
	with open(path, 'r') as file:
		file.readline()
		for line in file:
			line = line.split(',"', 1)[1].split('",', 1)
			label.append(line[0].split(' '))
			data.append(' '.join(tokenizer.tokenize(line[1])))
	#print(data[0])
	return data, label

def load_test_data(path):
	data = []
	tokenizer = RegexpTokenizer(r'[A-Za-z]+')
	with open(path, 'r') as file:
		file.readline()
		for line in file:
			data.append(' '.join(tokenizer.tokenize(line.split(',', 1)[1])))
	return data

def f1score(y_true, y_pred):
	y_pred = K.round(y_pred - THRES + 0.5)
	tp = K.sum(y_true * y_pred, axis = -1)
	precision = tp / (K.sum(y_pred, axis = -1) + K.epsilon())
	recall = tp / (K.sum(y_true, axis = -1) + K.epsilon())
	return K.mean(2 * ((precision * recall) / (precision + recall + K.epsilon())))

def rnn_model(embedding_layer):

	model = Sequential()
	model.add(embedding_layer)
	
	model.add(GRU(128, activation = 'tanh', dropout = 0.2))
	model.add(Dense(256, activation = 'elu'))
	model.add(Dropout(0.2))
	model.add(Dense(128, activation = 'elu'))
	model.add(Dropout(0.2))
	model.add(Dense(64, activation = 'elu'))
	model.add(Dropout(0.5))
	model.add(Dense(38, activation='sigmoid'))

	adam = Adam(lr = 0.001, decay = 1e-6, clipvalue = 0.5)
	model.compile(loss='categorical_crossentropy', optimizer = adam, metrics = [f1score])
	model.summary()
	return model

def validation_set(x, y, r):
	index = np.arange(x.shape[0])
	np.random.shuffle(index)
	x, y = x[index], y[index]
	
	n = int(r * x.shape[0])

	x_train, y_train = x[n:], y[n:]
	x_valid, y_valid = x[:n], y[:n]

	return x_train, y_train, x_valid, y_valid

if __name__ == '__main__':
	#x_train, y_train = load_train_data('train_data.csv')
	x_test = load_test_data(sys.argv[1])
	
	#x_tokenizer = Tokenizer()
	#x_tokenizer.fit_on_texts(x_train + x_test)
	#pickle.dump(x_tokenizer, open('tokenizer', 'wb'))
	
	x_tokenizer = pickle.load(open('tokenizer', 'rb'))
	stop = list(itertools.chain(*x_tokenizer.texts_to_sequences(nltk.corpus.stopwords.words('english'))))
	
	#x_train = x_tokenizer.texts_to_sequences(x_train)
	#x_train = pad_sequences(x_train, maxlen = TEXT_LEN)
	
	x_test = x_tokenizer.texts_to_sequences(x_test)
	x_test = pad_sequences(x_test, maxlen = TEXT_LEN)

	for x in stop:
		#x_train[x_train == x] = 0
		x_test[x_test == x] = 0
	
	y_mlb = pickle.load(open('MultiLabelBinarizer', 'rb'))
	#y_mlb.fit(y_train)
	#y_train = y_mlb.transform(y_train)

	# Use GloVe 
	
	word_index = x_tokenizer.word_index
	'''
	embeddings_index = {}
	f = open('glove.6B.{}d.txt'.format(EMBEDDING_DIM))
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()

	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		if i <= len(word_index):
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
	'''
	embedding_layer = Embedding(
			len(word_index) + 1,
			EMBEDDING_DIM,
			#weights=[embedding_matrix],
			input_length = TEXT_LEN,
			#mask_zero = True,
			trainable = False)
	
	#x_train, y_train, x_valid, y_valid = validation_set(x_train, y_train, 0.1)
	# Train
	model = rnn_model(embedding_layer)
	earlystopping = EarlyStopping(monitor = 'val_f1score', patience = 10, verbose = 1, mode = 'max')
	checkpoint = ModelCheckpoint(
			filepath='rnn.hdf5',
			verbose=1,
			save_best_only=True,
			save_weights_only=True,
			monitor='val_f1score',
			mode='max')
	#model.fit(x_train, y_train, epochs = EPOCHS, validation_data = (x_valid, y_valid), callbacks = [earlystopping, checkpoint], batch_size = BATCH_SIZE)
	#model.fit(x_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)
	model.load_weights('rnn.hdf5')
	
	
	predict = model.predict(x_test)
	predict = np.round(predict - THRES + 0.5)
	predict = y_mlb.inverse_transform(predict)
	
	out = open(sys.argv[2], 'w')
	print('"id","tags"', file = out)
	for i, x in enumerate(predict):
		print('"{}","{}"'.format(i, ' '.join(x)), file = out)
	exit()
