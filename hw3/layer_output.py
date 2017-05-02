import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from keras.utils import *
import numpy as np
from data_process import load_data

if __name__ == '__main__':
	emotion_classifier = load_model('best_model')
	layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
	
	input_img = emotion_classifier.input
	name_ls = ['conv2d_4']
	collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]
	#collect_layers = [ emotion_classifier.layer ]

	private_pixels = load_data('test.csv', 'test')
	private_pixels = [x.reshape(1, 48, 48, 1) for x in private_pixels]
	
	choose_id = 7122
	photo = private_pixels[choose_id]

	for cnt, fn in enumerate(collect_layers):
		im = fn([photo, 0]) #get the output of that layer
		fig = plt.figure(figsize=(14, 8))
		nb_filter = im[0].shape[3]
		for i in range(nb_filter):
			ax = fig.add_subplot(nb_filter/16, 16, i+1)
			ax.imshow(im[0][0, :, :, i], cmap='BuGn')
			plt.xticks(np.array([]))
			plt.yticks(np.array([]))
			plt.tight_layout()
		fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
		fig.savefig('layer{}.png'.format(cnt))
