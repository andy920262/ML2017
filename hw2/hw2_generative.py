import sys
import numpy as np
import math


raw_data_path = sys.argv[1]
test_data_path = sys.argv[2]
x_train_path = sys.argv[3]
y_train_path = sys.argv[4]
x_test_path = sys.argv[5]
output_path = sys.argv[6]

'''
x_train_path = 'X_train'
y_train_path = 'Y_train'
x_test_path = 'X_test'
output_path = 'ans.csv'
'''

PI = math.pi

def get_normp(data):
	return [(x.mean(), x.std()) for x in data.transpose()]

def normalization(data, normp):
	for i in range(len(data[0])):
		data[:, i] = (data[:,i] - normp[i][0]) / normp[i][1]
	return data

def load_train_data(x_path, y_path):
	train_data = []
	train_label = []
	with open(x_path, 'r') as file:
		file.readline()
		for line in file:
			train_data.append([-1 if x == '?' else int(x) for x in line.split(',')])
	with open(y_path, 'r') as file:
		for line in file:
			train_label.append(int(line))
			#train_label.append(-1 if line == '0' else 1)
	return (np.array(train_data), np.array(train_label))

def load_test_data(path):
	test_data = []
	with open(path, 'r') as file:
		file.readline()
		for line in file:
			test_data.append([-1 if x == '?' else int(x) for x in line.split(',')])
	return np.array(test_data)

def gauss_dist(sigma, mu, det, inv, x):
	d = len(x)
	return (1 / np.sqrt((2 * PI)**d * det)) * np.exp(-(1 / 2) * np.dot(np.dot((x - mu), inv), (x - mu)))

def accuracy(sigma, mu0, mu1, p0, p1, det, inv, x, y):
	s = 0
	for i in range(len(x)):
		g1 = gauss_dist(sigma, mu1, det, inv, x[i]) * p1
		g0 = gauss_dist(sigma, mu0, det, inv, x[i]) * p0
		s += (g1 > g0) == y[i]
	return s / len(x)

def generative_model(train_data, train_label):
	class0 = []
	class1 = []
	for i in range(len(train_label)):
		if train_label[i] == 0:
			class0.append(train_data[i])
		else:
			class1.append(train_data[i])
	p0 = len(class0) / (len(class1) + len(class0))
	p1 = len(class1) / (len(class1) + len(class0))
	class0 = np.array(class0)
	class1 = np.array(class1)
	mu0 = np.mean(class0, axis = 0)
	mu1 = np.mean(class1, axis = 0)
	#cov0 = np.dot((class0 - mu0).transpose(), (class0 - mu0))
	#cov1 = np.dot((class1 - mu1).transpose(), (class1 - mu1))
	sigma = p0 * np.cov(class0.T) + p1 * np.cov(class1.T)
	det = abs(np.linalg.det(sigma) + 1e-200)
	if det < 1e-100:
		inv = np.linalg.pinv(sigma)
	else:
		inv = np.linalg.inv(sigma)

	return (sigma, mu0, mu1, p0, p1, det, inv)

def validation(train_data, train_label, val_set):
	np.random.seed(7122);np.random.shuffle(train_data)
	np.random.seed(7122);np.random.shuffle(train_label)
	s = 0
	for x in range(val_set):
		u = int(len(train_data) / val_set * x)
		v = int(len(train_data) / val_set * (x + 1))
		x = np.delete(train_data, range(u, v), 0)
		y = np.delete(train_label, range(u, v), 0)
		tx = train_data[u : v]
		ty = train_label[u : v]
		sigma, mu0, mu1, p0, p1, det, inv = generative_model(x, y)
		s += accuracy(sigma, mu0, mu1, p0, p1, det, inv, tx, ty)
	print('valid accuracy:', s / val_set, file = sys.stderr)

def sample(data):

	data = np.hstack((data, (data[:, : 6])**2))
	data = np.hstack((data, (data[:, : 6])**3))
	data = np.hstack((data, (data[:, : 6])**4))
	data = np.hstack((data, (data[:, : 6])**5))
	data = np.hstack((data, (data[:, : 6])**6))
	data = np.hstack((data, (data[:, : 6])**7))
	data = np.hstack((data, (data[:, : 6])**8))
	data = np.hstack((data, np.log(data[:, : 6] + 1e-200)))
	data = np.hstack((data, np.tan(data[:, : 6])))

	return data

if __name__ == "__main__":

	train_data, train_label = load_train_data(x_train_path, y_train_path)
	train_data = sample(train_data)

	normp = get_normp(train_data)
	train_data = normalization(train_data, normp)

	validation(train_data, train_label, 5)
	#exit()
	sigma, mu0, mu1, p0, p1, det, inv = generative_model(train_data, train_label)
	print('training accuracy:', accuracy(sigma, mu0, mu1, p0, p1, det, inv, train_data, train_label), file = sys.stderr)

	test_data = load_test_data(x_test_path)
	test_data = sample(test_data)

	test_data = normalization(test_data, normp)

	output = open(output_path, 'w')
	print('id,label', file = output)
	for x in range(len(test_data)):
		r = (gauss_dist(sigma, mu1, det, inv, test_data[x]) * p1 >  gauss_dist(sigma, mu0, det, inv, test_data[x]) * p0)
		print(x + 1, ',', int(r), file = output, sep = '')