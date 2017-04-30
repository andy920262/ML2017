import matplotlib
matplotlib.use('Agg')
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from data_process import load_data
import itertools

def plot_confusion_matrix(cm, classes,
			  title='Confusion matrix',
			  cmap=plt.cm.jet):
	"""
	This function prints and plots the confusion matrix.
	"""
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
			color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

if __name__ == '__main__':
	emotion_classifier = load_model('best_model')
	np.set_printoptions(precision=2)
	dev_feats, te_labels = load_data('valid', 'origin')
	predictions = emotion_classifier.predict_classes(dev_feats)
	print(te_labels.shape, predictions.shape)
	conf_mat = confusion_matrix(te_labels, predictions)

	fig = plt.figure()
	plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
	fig.savefig('confusion_matrix.png')
