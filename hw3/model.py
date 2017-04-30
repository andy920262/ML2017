import numpy as np
import sys
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.callbacks import Callback

SIZE = 48

class History(Callback):
	def on_train_begin(self,logs={}):
		self.tr_losses=[]
		self.val_losses=[]
		self.tr_accs=[]
		self.val_accs=[]

	def on_epoch_end(self,epoch,logs={}):
		self.tr_losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))
		self.tr_accs.append(logs.get('acc'))
		self.val_accs.append(logs.get('val_acc'))

	def save(self, path):
		with open(path, 'a') as f:
			print('train_acc, train_loss, val_acc, val_loss', file = f)
			for i in range(len(self.tr_accs)):
				print(i + 1, self.tr_accs[i], self.tr_losses[i], self.val_accs[i], self.val_losses[i], sep = ', ', file = f)

def build_model():
	model = Sequential()
	
	#48
	model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (SIZE, SIZE, 1)))
	model.add(BatchNormalization())
	#model.add(MaxPooling2D(2, 2))
	model.add(Dropout(0.3))

	model.add(Conv2D(32, (3, 3), padding = 'same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(2, 2))
	model.add(Dropout(0.3))

	#24
	model.add(Conv2D(64, (3, 3), padding = 'same'))
	model.add(BatchNormalization())
	#model.add(MaxPooling2D(2, 2))
	model.add(Dropout(0.5))
	
	model.add(Conv2D(64, (3, 3), padding = 'same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(2, 2))
	model.add(Dropout(0.5))
	
	#12
	model.add(Conv2D(128, (3, 3), padding = 'same'))
	model.add(BatchNormalization())
	#model.add(MaxPooling2D(2, 2))
	model.add(Dropout(0.5))
	
	model.add(Conv2D(128, (3, 3), padding = 'same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(2, 2))
	model.add(Dropout(0.5))
	
	#6
	model.add(Conv2D(256, (3, 3)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(2, 2))
	model.add(Dropout(0.5))
	
	#4i
	#model.add(Conv2D(256, (3, 3)))
	#model.add(BatchNormalization())
	#model.add(MaxPooling2D(2, 2))
	#model.add(Dropout(0.5))
	
	model.add(Flatten())

	model.add(Dense(units = 1024, activation = 'relu'))
	model.add(BatchNormalization())
	#model.add(Dropout(0.5))

	model.add(Dense(units = 1024, activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(units = 7, activation = 'softmax'))
	model.summary()
	return model

def build_dnn_model():
	model = Sequential()
	

	model.add(Dense(units = 128, activation = 'relu', input_shape = (SIZE, SIZE, 1)))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(units = 128, activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	#model.add(Dense(units = 128, activation = 'relu'))
	#model.add(BatchNormalization())
	#model.add(Dropout(0.5))

	model.add(Flatten())
	
	model.add(Dense(units = 7, activation = 'softmax'))
	model.summary()
	return model


