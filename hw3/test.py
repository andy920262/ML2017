import numpy as np
import sys
from data_process import load_data
from keras.models import load_model

model_path = 'best_model'
test_path = sys.argv[1]
predict_path = sys.argv[2]

if __name__ == "__main__":

	model = load_model(model_path)

	#print('\nvalid:', model.evaluate(valid_x, valid_y)[1])

	test_x = load_data(test_path, 'test')
	ans = model.predict_classes(test_x, batch_size = 128)
	out = open(predict_path, 'w')
	print('id,label', file = out)
	for i in range(len(ans)):
		print(i, ',', ans[i], sep = '', file = out)

	exit()
