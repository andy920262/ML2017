import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from keras.utils import *
import numpy as np

def normalize(x):
	# utility function to normalize a tensor by its L2 norm
	return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func):
	ETA = 1e-2
	filter_images = input_img_data
	Gw = np.zeros(filter_images.shape)
	for epochs in range(num_step):
		filt, grad = iterate([filter_images, False])
		#Gw += grad**2
		filter_images = filter_images + ETA * grad
		#filter_images += (ETA / Gw**0.5) * grad
		#filter_images[np.where(grad < 0)] -= 1
		#filter_images[np.where(grad > 0)] += 1
		if (epochs % 10 == 0):
			print('epochs:{}, loss:{}'.format(epochs, filt))
	#print(np.min(filter_images), np.max(filter_images))
	filter_images[np.where(filter_images < 0)] = 0
	filter_images[np.where(filter_images > 1)] = 1
	#for i in filter_images.ravel():
	#	print(i)
	return filter_images.reshape(48, 48) * 255, filt

if __name__ == '__main__':
	emotion_classifier = load_model('best_model')
	layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers)
	input_img = emotion_classifier.input

	name_ls = ["conv2d_4"]
	collect_layers = [ layer_dict[name].output for name in name_ls ]

	nb_filter = 64
	for cnt, c in enumerate(collect_layers):
		filter_imgs = [[] for i in range(nb_filter)]
		for filter_idx in range(nb_filter):
			input_img_data = np.random.random((1, 48, 48, 1)) # random noise
			target = K.mean(c[:, :, :, filter_idx])
			grads = normalize(K.gradients(target, input_img)[0])
			iterate = K.function([input_img, K.learning_phase()], [target, grads])

			###
			"You need to implement it."
			filter_imgs[filter_idx] = grad_ascent(150, input_img_data, iterate)
			#print(filter_imgs[-1])
			###
		print(len(filter_imgs))
		fig = plt.figure(figsize=(14, 8))
		for i in range(nb_filter):
			ax = fig.add_subplot(nb_filter/16, 16, i+1)
			ax.imshow(filter_imgs[i][0], cmap='BuGn')
			plt.xticks(np.array([]))
			plt.yticks(np.array([]))
			#plt.xlabel('{:.3f}'.format(filter_imgs[i][1]))
			plt.tight_layout()
		fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], 150))
		fig.savefig('e{}'.format(100))
