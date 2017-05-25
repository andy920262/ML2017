from keras.preprocessing.text import Tokenizer
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import nltk
import numpy as np
import pickle
import itertools
import sys

TEXT_LEN = 310
THRES = 0.4
EPOCHS = 20
BATCH_SIZE = 64

def load_train_data(path):
	data = []
	label = []
	tokenizer = RegexpTokenizer(r'[A-Za-z]+')
	lemmatizer = WordNetLemmatizer()
	with open(path, 'r') as file:
		file.readline()
		for line in file:
			line = line.split(',"', 1)[1].split('",', 1)
			label.append(line[0].split(' '))
			data.append(' '.join([lemmatizer.lemmatize(x) for x in tokenizer.tokenize(line[1])]))
	#print(data[0])
	return data, label

def load_test_data(path):
	data = []
	tokenizer = RegexpTokenizer(r'[A-Za-z]+')
	lemmatizer = WordNetLemmatizer()
	with open(path, 'r') as file:
		file.readline()
		for line in file:
			data.append(' '.join([lemmatizer.lemmatize(x) for x in tokenizer.tokenize(line.split(',', 1)[1])]))
	return data

def f1score(y_true, y_pred):
	y_pred = K.round(y_pred - THRES + 0.5)
	tp = K.sum(y_true * y_pred, axis = -1)
	precision = tp / (K.sum(y_pred, axis = -1) + K.epsilon())
	recall = tp / (K.sum(y_true, axis = -1) + K.epsilon())
	return K.mean(2 * ((precision * recall) / (precision + recall + K.epsilon())))

def build_model():

	model = Sequential()
	
	model.add(Dense(512, activation = 'elu', input_shape = (TEXT_LEN,)))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation = 'elu'))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation = 'elu'))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation = 'elu'))
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

def train(argv, x_train, y_train, x_test):
	
	#x_train, y_train, x_valid, y_valid = validation_set(x_train, y_train, 0.1)
	
	# Train
	model = build_model()
	#model.fit(x_train, y_train, epochs = EPOCHS, validation_data = (x_valid, y_valid), batch_size = BATCH_SIZE)
	#model.fit(x_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)
	#model.save_weights('model{}.hdf5'.format(argv))
	model.load_weights('model{}.hdf5'.format(argv))
	
	
	predict = model.predict(x_test)
	predict = np.round(predict - THRES + 0.5)
	return predict

if __name__ == '__main__':
	nltk.download('stopwords')
	nltk.download('wordnet')
	#x_train, y_train = load_train_data('train_data.csv')
	x_test = load_test_data(sys.argv[1])
	
	#x_tokenizer = Tokenizer()
	#x_tokenizer.fit_on_texts(x_train + x_test)
	#pickle.dump(x_tokenizer, open('tokenizer', 'wb'))
	
	x_tokenizer = pickle.load(open('tokenizer', 'rb'))
	stop = list(itertools.chain(*x_tokenizer.texts_to_sequences(nltk.corpus.stopwords.words('english'))))
	
	#x_train = x_tokenizer.texts_to_matrix(x_train, mode = 'tfidf')
	x_test = x_tokenizer.texts_to_matrix(x_test, mode = 'tfidf')

	TEXT_LEN = x_test.shape[1]
	for x in stop:
		#x_train[:,x] = 0
		x_test[:,x] = 0
	
	y_mlb = pickle.load(open('MultiLabelBinarizer', 'rb'))
	#y_mlb.fit(y_train)
	#y_train = y_mlb.transform(y_train)
	
	predict = []
	for i in range(20):
		predict.append(train(i, 0, 0, x_test))	
		#predict.append(train(i, x_train, y_train, x_test))	
	predict = np.sum(predict, axis = 0)
	predict[predict <  8] = 0
	predict[predict >= 8] = 1
	predict = y_mlb.inverse_transform(predict)

	out = open(sys.argv[2], 'w')
	print('"id","tags"', file = out)
	for i, x in enumerate(predict):
		print('"{}","{}"'.format(i, ' '.join(x)), file = out)
	exit()
