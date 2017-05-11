import numpy as np
import sys

data_path = sys.argv[1]
output_path = sys.argv[2]

thres = [
	0.86951 ,0.85137 ,0.85285, 0.83094, 0.82012, 0.82775, 0.82284 ,0.81704, 0.81696, 0.82130,
	0.82517, 0.82639, 0.82530, 0.83368, 0.83934, 0.83647, 0.84331, 0.84659, 0.85249, 0.85873,
	0.85985, 0.86484, 0.87150, 0.87117, 0.87985, 0.88471, 0.88911, 0.89492, 0.89668, 0.90442,
	0.90642, 0.91335, 0.91326, 0.91743, 0.92275, 0.92655, 0.92959, 0.93010, 0.93766, 0.93958,
	0.94268, 0.94466, 0.94909, 0.95083, 0.95534, 0.95655, 0.95900, 0.96152, 0.96331, 0.96675,
	0.96821, 0.96971, 0.97246, 0.97411, 0.97540, 0.97683, 0.97827, 0.98018, 0.98112, 0.98278]

def predict(x):
	x = x - x.mean(axis = 0) + np.random.random(x.shape) * 0.5
	e_val, e_vec = np.linalg.eigh(np.cov(x.T))
	e_val, e_vec = e_val[::-1], e_vec[::-1]
	prefix_s = np.cumsum(e_val) / np.sum(e_val)
	for i in range(60):
		if prefix_s[i] > thres[i]:
			return np.log(i + 1)
	return np.log(60)

'''
def hand():
	from PIL import Image
	data = []
	for i in range(1, 482):
		data.append(Image.open('hand/hand.seq{}.png'.format(i)))
	data = [np.asarray(x.resize((64, 60))) for x in data]
	data = np.array(data).reshape(481, -1)
	print('Dim = {}'.format(np.round(np.exp(predict(data)))))
'''

if __name__ == '__main__':
	data = np.load(data_path)
	output = open(output_path, 'w')
	print('SetId,LogDim', file = output)
	for i in range(200):
		dim = predict(data[str(i)])
		print('{},{}'.format(i, dim), file = output)
	exit()
