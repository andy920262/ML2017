import word2vec
from sklearn import manifold
import matplotlib.pyplot as plt
import nltk
import numpy as np
from adjustText import adjust_text

def train():
	word2vec.word2phrase('all.txt', 'phrase.txt', verbose = True)
	word2vec.word2vec('phrase.txt', 'vec.bin', min_count = 50, size = 50, verbose = False)

def plot():
	k = 500
	model = word2vec.load('vec.bin')
	data = manifold.TSNE(2).fit_transform(model.vectors[:k]).T
	pos_tag = nltk.tag.pos_tag(model.vocab[:k])

	# Select vocab
	idx = []
	for i in range(k):
		if pos_tag[i][1] not in ['JJ', 'NNP', 'NN', 'NNS']:
			continue
		if len(model.vocab[i]) < 2:
			continue
		if any((c in '(“,.:;’!?”)') for c in pos_tag[i][0]):
			continue
		idx.append(i)

	# Draw
	fig = plt.figure(figsize = (16, 9))
	plt.scatter(data[0][idx], data[1][idx], c = np.random.rand(k, 3), s =15)
	texts = []
	for i in idx:
		texts.append(plt.text(data[0][i], data[1][i], model.vocab[i], size = 10))
	adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
	#plt.show()
	fig.savefig('word2vec.png')

if __name__ == '__main__':
	train()
	plot()
	exit()

