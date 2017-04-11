import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json

train_path = 'train.csv'

SIZE = 48
DIM = SIZE * SIZE

def load_data(path, data_type):
	x = []
	y = []
	with open(path, 'r') as file:
		file.readline()
		for line in file:
			s = [int(i) for i in line.replace(',', ' ').split(' ')]
			x.append(s[1 : ])
			y.append(s[0])
	if data_type == 'train':
		return (np.array(x).reshape(len(x), SIZE, SIZE, 1), to_categorical(np.array(y), 7))
	if data_type == 'test':
		return np.array(x).reshape(len(x), SIZE, SIZE, 1)

if __name__ == "__main__":
	train_x, train_y = load_data(train_path, 'train')
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape = (SIZE, SIZE, 1)))
	model.add(MaxPooling2D(2, 2))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (2, 2)))
	model.add(MaxPooling2D(2, 2))
	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dense(units = 128, activation = 'relu'))
	model.add(Dense(units = 128, activation = 'relu'))
	model.add(Dense(units = 7, activation = 'softmax'))
	model.summary()
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	model.fit(train_x, train_y, batch_size = 100, epochs = 10)
	#model.save_weights('model_weight')
	x = load_data('test.csv', 'test')
	print(model.evaluate(train_x, train_y, batch_size = 100))
	ans = model.predict_classes(x, batch_size = 100)
	out = open('ans.csv', 'w')
	print('id,label', file = out)
	for i in range(len(ans)):
		print(i, ',', ans[i], sep = '', file = out)

	exit()
