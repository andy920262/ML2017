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
EPA = 1
LAM = 1e-3
ITER = 300

def get_normp(data):
	return [(x.mean(), x.std()) for x in data.transpose()]

def normalization(data, normp):
	for i in range(len(data[0])):
		data[:, i] = (data[:,i] - normp[i][0]) / normp[i][1]
	return data

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def accuracy(w, b, x, y):
	s = 0
	for i in range(len(x)):
		r = round(sigmoid(np.sum(w * x[i]) + b))
		s += r == y[i]
	return s / len(x)

def loss_func(w, b, x, y):
	sig = sigmoid(np.dot(x, w) + b)
	return -np.sum(y * np.log(sig + 1e-200) + (1 - y) * np.log(1 - sig + 1e-200)) + LAM * np.sum(w**2)

def grad(w, b, x, y):
	sig = sigmoid(np.dot(x, w) + b)
	s = y - sig
	return (-np.sum(s * x.transpose(), axis = 1) + 2 * LAM * w, -np.sum(s))

def grad_descent(x, y):
	current_w = np.ones(len(x[0]))
	current_b = 1
	gw = 1e-200
	gb = 1e-200
	for i in range(ITER):
		grad_w, grad_b = grad(current_w, current_b, x, y)
		gw += grad_w**2
		gb += grad_b**2
		current_w += -(EPA / gw)**0.5 * grad_w
		current_b += -(EPA / gb)**0.5 * grad_b
		#if i % 10 == 0:
			#print(i, ':', accuracy(current_w, current_b, x, y), file = sys.stderr)
	return (current_w, current_b)

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
		w ,b =  grad_descent(x, y)
		s += accuracy(w, b, tx, ty)
		#print(accuracy(w, b, tx, ty))
	print('valid accuracy:', s / val_set, file = sys.stderr)

def sample(data):
	np.delete(data, 1, 1)
	data = np.hstack((data, (data[:, : 5])**2))
	data = np.hstack((data, (data[:, : 5])**3))
	data = np.hstack((data, (data[:, : 5])**4))
	data = np.hstack((data, (data[:, : 5])**5))
	data = np.hstack((data, (data[:, : 5])**6))
	data = np.hstack((data, (data[:, : 5])**7))
	data = np.hstack((data, (data[:, : 5])**8))
	data = np.hstack((data, (data[:, : 5])**9))
	data = np.hstack((data, (data[:, : 5])**10))
	data = np.hstack((data, np.log(data[:, : 5] + 1e-200)))
	data = np.hstack((data, np.tan(data[:, : 5])))
	data = np.hstack((data, np.exp(data[:, : 5] * 1e-5)))
	data = np.hstack((data, np.arctan(data[:, : 5])))

	return data

if __name__ == "__main__":

	train_data, train_label = load_train_data(x_train_path, y_train_path)
	train_data = sample(train_data)

	normp = get_normp(train_data)
	train_data = normalization(train_data, normp)

	validation(train_data, train_label, 5)
	w, b = grad_descent(train_data, train_label)
	print('training accuracy:', accuracy(w, b, train_data, train_label), file = sys.stderr)

	test_data = load_test_data(x_test_path)
	test_data = sample(test_data)

	test_data = normalization(test_data, normp)

	output = open(output_path, 'w')
	print('id,label', file = output)
	for x in range(len(test_data)):
		r = round(sigmoid(np.sum(w * test_data[x]) + b))
		print(x + 1, ',', int(r), file = output, sep = '')