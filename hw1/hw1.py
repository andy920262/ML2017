import sys
import random
import numpy as np

ETA = 1
LAM = 1e-3
MAX_DATA = 480
MOD_DEG = 2
MAX_FTR = 9 * 6 * MOD_DEG
PM25 = 9
normp = []

def parse_line(line, u, v):
	data = [0.0 if line[x][0] == 'N' else float(line[x]) for x in range(u, v)]
	return data

def read_training_data(path):	
	with open(path, 'r', encoding = 'big5') as file:
		file.readline()
		training_data = [[],[]]
		for month in range(12):
			tmp = [[] for x in range(18)]
			for day in range(20):
				for feature in range(18):
					tmp[feature] += parse_line(file.readline().split(','), 3 + 0, 3 + 24)
			for i in range(MAX_DATA - 10):
				training_data[0].append([x**(d + 1) for d in range(MOD_DEG) for x in tmp[8][i : i + 9] + tmp[9][i : i + 9] + tmp[5][i : i + 9] + tmp[6][i : i + 9] + tmp[7][i : i + 9] + tmp[12][i : i + 9]])
				training_data[1].append(tmp[9][i + 9])
		training_data[0] = np.array(training_data[0])
		training_data[1] = np.array(training_data[1])
		#print(np.corrcoef(training_data[0].transpose())[-1])
	return training_data

def get_normp(data):
	return [(x.mean(), x.std()) for x in data.transpose()]


def loss_func(weight, bias, training_data):
	s = training_data[1] - bias - np.dot(training_data[0], weight)
	ret = np.sum(s**2) + LAM * np.sum(weight**2)
	return (ret / len(training_data[1]))**0.5

def gradient(weight, bias, training_data):
	s = training_data[1] - bias - np.dot(training_data[0], weight)
	dw = -2 * np.sum(s * training_data[0].transpose(), axis = 1) + 2 * LAM * weight
	db = -2 * np.sum(s)
	return (dw, db)

def testing(path, weight, bias):
	with open(path, 'r') as file:
		testing_data = []
		print('id,value')
		for i in range(240):
			tmp = []
			for j in range(18):
				tmp.append(parse_line(file.readline().split(','), 2 + 0, 2 + 9))
			testing_data.append([x**(d + 1) for d in range(MOD_DEG) for x in tmp[8] + tmp[9] + tmp[5] + tmp[6] + tmp[7] + tmp[12]])
		testing_data = np.array(testing_data)
		#print(testing_data)
		for i in range(len(testing_data[0])):
			testing_data[:, i] = (testing_data[:, i] - normp[i][0]) / normp[i][1]
		for i in range(len(testing_data)):
			val = np.dot(testing_data[i], weight) + bias
			print('id_', i, ',', val, sep='')

def gradient_descent(data):
	current_w = np.ones(MAX_FTR, dtype = np.float64)
	current_b = 1.0
	Gw = np.zeros(MAX_FTR, dtype = np.float64)
	Gb = 0.0

	for i in range(2000):
		dw, db = gradient(current_w, current_b, data)
		
		#AdaGrad
		Gw += dw**2
		Gb += db**2
		current_w -= (ETA / Gw**0.5) * dw
		current_b -= (ETA / Gb**0.5) * db

		#Gradient Descent
		#current_w -= ETA * dw
		#current_b -= ETA * db
		if i % 100 == 0:
			loss =  loss_func(current_w, current_b, data)
			print(i, ':loss:' , loss, ',dw:', dw.sum(), ',b:', current_b, file = sys.stderr, sep = '')
		#if loss < 6:
		#	break

	print(current_w)
	return (current_w, current_b)
	
training_data = read_training_data(sys.argv[1])
#exit()
normp = get_normp(training_data[0])
for i in range(len(training_data[0][0])):
	training_data[0][:, i] = (training_data[0][:,i] - normp[i][0]) / normp[i][1]
w, b = gradient_descent(training_data)
testing(sys.argv[2], w, b)
