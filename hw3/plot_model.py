import sys
from keras.utils import plot_model
from keras.models import load_model
#from keras.utils import plot_model
#from keras.models import load_model

if __name__ == '__main__':
	model = load_model(sys.argv[1])
	plot_model(model, to_file = sys.argv[2])
	exit()
