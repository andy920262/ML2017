from data_process import load_data 
from model import build_model, History
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

v = int(28709 * 0.2)
r = int(28709 * 0.8 * 0.2) + v

if __name__ == '__main__':
	data_x, data_y = load_data('train.csv', 'train')
	valid_x, valid_y = data_x[  : v], data_y[  : v]
	data1_x, data1_y = data_x[v : r], data_y[v : r]
	data2_x, data2_y = data_x[r :  ], data_y[r :  ]


	datagen = ImageDataGenerator(
			#rescale = 1 / 255,
			#zca_whitening = True,
			rotation_range = 3,
			width_shift_range = 0.1,
			height_shift_range = 0.1,
			zoom_range = 0.1,
			horizontal_flip = True)
	datagen.fit(data1_x)

	history1 = History()

	model = build_model()
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	model.fit_generator(
			datagen.flow(data1_x, data1_y, batch_size = 128),
			steps_per_epoch = len(data1_x) / 128 * 4,
			#steps_per_epoch = 1,
			epochs = 50,
			validation_data = (valid_x, valid_y),
			callbacks = [history1])
	#model.fit(data1_x, data1_y, batch_size = 128, epochs = 100, validation_data = (valid_x, valid_y), callbacks = [history1])
	model.save('self_model')
	history1.save('history1')

	datagen.fit(data2_x)

	history2 = History()
	
	data2_y = to_categorical(model.predict_classes(data2_x, batch_size = 128), 7)
	model2 = build_model()
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	model.fit_generator(
			datagen.flow(data2_x, data2_y, batch_size = 128),
			steps_per_epoch = len(data2_x) / 128 * 4,
			#steps_per_epoch = 1,
			epochs = 50,
			validation_data = (valid_x, valid_y),
			callbacks = [history2])
	#model.fit(data2_x, data2_y, batch_size = 128, epochs = 100, validation_data = (valid_x, valid_y), callbacks = [history2])
	history2.save('history2')

	exit()
