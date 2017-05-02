import numpy as np
import sys
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from data_process import load_data
from model import History, build_model 

train_path = sys.argv[1]
model_path = 'model'

SIZE = 48

if __name__ == "__main__":
	train_x, train_y = load_data(train_path, 'train')
	
	datagen = ImageDataGenerator(
			#rescale = 1 / 255,
			#zca_whitening = True,
			rotation_range = 3,
			width_shift_range = 0.1,
			height_shift_range = 0.1,
			zoom_range = 0.1,
			horizontal_flip = True)
	datagen.fit(train_x)

	#valid_x, valid_y = load_data('valid', 'train')

	#model = load_model('model')
	#model.save_weights('weight')
	model = build_model()
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	#model.load_weights('best_w')
	history = History()
	model.fit_generator(
			datagen.flow(train_x, train_y, batch_size = 128),
			steps_per_epoch = len(train_x) / 128 * 8,
			#steps_per_epoch = 1,
			epochs = 50,
			#validation_data = (valid_x, valid_y),
			callbacks = [history])
	history.save('history')

	model.save(model_path)
	#model.save_weights('weight')
	#print('\nvalid:', model.evaluate(valid_x, valid_y)[1])

	exit()
