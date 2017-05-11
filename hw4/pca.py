import sys
import os
import string
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


'''
def pca_skl(data, dim):
	from sklearn.decomposition import PCA
	return PCA(n_components = dim).fit(data).transform(data)
'''


def eig_vec(data):
	data = data - data.mean(axis = 0)
	cov = np.cov(data.T)
	e_val, e_vec = np.linalg.eigh(cov)
	idx = np.argsort(e_val)[::-1]
	return e_vec[:, idx]

def load_data():
	
	img_dir = 'faceExpressionDatabase'
	data = []
	for i in string.ascii_uppercase[:13]:
		img = []
		for j in range(75):
			img.append(mpimg.imread(os.path.join(img_dir, '{}{:02d}.bmp'.format(i, j))))
		data.append(img)

	return np.array(data)

def plot_img(x, y, data, path):
	fig = plt.figure()
	for i in range(x * y):
		ax = fig.add_subplot(x, y, i + 1)
		ax.imshow(data[i], cmap = 'gray')
		plt.xticks(np.array([]))
		plt.yticks(np.array([]))
	plt.show()
	fig.savefig(path)

if __name__ == '__main__':

	data = load_data()[: 10, : 10, :, :]
	data = data.reshape(100, 64 * 64)

	# Average face
	avg = np.average(data, axis = 0)
	
	# Eigen face
	e_face = eig_vec(data).T

	# Reconstruct
	recover = np.zeros((100, 64 * 64)) + data.mean(axis = 0)
	w = ((data - data.mean(axis = 0)) @ e_face.T).T
	for i in range(100):
		k = i + 1
		recover += np.outer(w[i], e_face[i])
		err = np.sqrt(np.sum((recover - data)**2) / (100 * 64 * 64)) / 256
		print('k = {}, error:{}'.format(k, err))
		if k == 5:
			recover_k5 = recover.copy()
		if err <= 0.01:
			break
	# Draw img
	plot_img(1, 1, avg.reshape(1, 64, 64), 'avg_face.png')
	plot_img(3, 3, e_face.reshape(4096, 64, 64), 'eigen_face.png')
	plot_img(10, 10, data.reshape(100, 64, 64), 'origin_face.png')
	plot_img(10, 10, recover_k5.reshape(100, 64, 64), 'k5_face.png')
	plot_img(10, 10, recover.reshape(100, 64, 64), 'recover_face.png')
	exit()
