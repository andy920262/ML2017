import numpy as np
from keras.utils import to_categorical
SIZE = 48

def normalization(x):
	return x / 255

def histogram_equalization(data):
	for i in range(data.shape[0]):
		image = data[i]
		histogram, bins = np.histogram(image.flatten(), 256, normed=True)
		cdf = histogram.cumsum()
		cdf = 255 * cdf / cdf[-1]
		ret = np.interp(image.flatten(), bins[:-1], cdf)
		data[i] = ret.reshape(image.shape)
	return data

def load_data(path, data_type):
	x = []
	y = []
	with open(path, 'r') as file:
		file.readline()
		for line in file:
			if path == 'fer2013.csv':
				s = [int(i) for i in line.replace(',', ' ').split(' ')[:-1]]
			else:
				s = [int(i) for i in line.replace(',', ' ').split(' ')]
			x.append(s[1 : ])
			y.append(s[0])
	 
	x = normalization(histogram_equalization(np.array(x).reshape(len(x), SIZE, SIZE, 1)))
	if data_type == 'train':
		return x, to_categorical(np.array(y), 7)
	if data_type == 'test':
		return x
	if data_type == 'origin':
		return x, np.array(y)

