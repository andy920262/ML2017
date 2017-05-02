import matplotlib
matplotlib.use('Agg')
import argparse
from keras.models import load_model
import keras.backend as K
from keras.utils import *
import numpy as np
import matplotlib.pyplot as plt
from data_process import load_data

if __name__ == '__main__':
	model_path = ('best_model')
	emotion_classifier = load_model(model_path)

	private_pixels = load_data('test.csv', 'test')
	private_pixels = [x.reshape(1, 48, 48, 1) for x in private_pixels]

	input_img = emotion_classifier.input
	img_ids = [7122]

	for idx in img_ids:
		val_proba = emotion_classifier.predict(private_pixels[idx])
		pred = val_proba.argmax(axis=-1)
		target = K.mean(emotion_classifier.output[:, pred])
		grads = K.gradients(target, input_img)[0]
		fn = K.function([input_img, K.learning_phase()], [grads])
		#print(fn([private_pixels[idx], True]))
		heatmap = fn([private_pixels[idx], False])
		heatmap = np.array(heatmap).reshape(48, 48)
		'''
		Implement your heatmap processing here!
		hint: Do some normalization or smoothening on grads
		'''

		thres = np.mean(np.abs(heatmap))
		see = private_pixels[idx].reshape(48, 48)
		#see[np.where(np.abs(heatmap) <= thres)] = np.mean(see)

		plt.figure()
		plt.imshow(heatmap, cmap=plt.cm.jet)
		plt.colorbar()
		plt.tight_layout()
		fig = plt.gcf()
		plt.draw()
		fig.savefig('heatmap{}.png'.format(idx))

		plt.figure()
		plt.imshow(see,cmap='gray')
		plt.colorbar()
		plt.tight_layout()
		fig = plt.gcf()
		plt.draw()
		fig.savefig('see{}.png'.format(idx))
