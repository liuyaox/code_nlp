
import numpy as np
import cPickle as pickle
from sklearn.decomposition import PCA


def get_word_embedding(word_embedding_file, header=False, seps=('\t', '\t')):
    """ Create dictionary mapping character or word to embedding using pretrained word embeddings """
    word_embedding = {}
    with open(word_embedding_file, 'r', encoding='utf-8') as fr:
        if header:
            fr.readline()                        # Drop line 1
        for line in fr:
            try:
                values = line.strip().split(seps[0])
                word = values[0]
                vector = values[1:] if seps[0] == seps[1] else values[1].split(seps[1])
                word_embedding[word] = np.asarray(vector, dtype='float32')
            except ValueError as e:
                pass
    return word_embedding

word_embedding_file = '/home/vikas/Desktop/glove.6B/glove.6B.300d.txt'
Glove = get_word_embedding(word_embedding_file, header=True, seps=(' ', ' '))


def dict_to_ndarray(dic, sortby=None):
	""" 把字典的keys和values转化为2个ndarray  sortby: 按key(=0)或value(=1)排序 """
	if sortby is None:
		items = dic.items()
	else:
		items = sorted(dic.items(), key=lambda x: x[sortby])
	keys, values = zip(*items)
	return np.asarray(keys), np.asarray(values)

X_train_names, X_train = dict_to_ndarray(Glove, sortby=0)


# PCA
def pca_reduce(X, n_components=100):
	pca = PCA(n_components=n_components)
	X_mean = X - np.mean(X)
	X_pca = pca.fit_transform(X_mean)
	U1 = pca.componets_
	return X_mean, X_pca, U1

# PPA
def ppa(X, n_components=200, d=7):
    X_mean, _, U1 = pca_reduce(X, n_components)		# Get Top Components
	X2 = []
	for i, x in enumerate(X_mean):
		for u in U1[:d]:							# Remove Projections on Top Components
			x = x - np.dot(u.transpose(), x) * u
		X2.append(x)
	return np.asarray(X2)

# PCA->PPA
def pca_ppa(X1, n_components=100, d=7):
	_, X2, _ = pca_reduce(X1, n_components)			# PCA
	X3 = ppa(X2, n_components, d=d)					# PPA
	return X3

# PPA->PCA
def ppa_pca(X0, ns_components=(200, 100), d=7):
	X1 = ppa(X0, ns_components[0], d=d)				# PPA
	_, X2, _ = pca_reduce(X1, ns_components[1])		# PCA
	return X2

# PPA->PCA->PPA
def ppa_pca_ppa(X0, ns_components=(200, 100), ds=(7, 7))
	X1 = ppa(X0, ns_components[0], d=ds[0])			# PPA
	_, X2, _ = pca_reduce(X1, ns_components[1])		# PCA
	X3 = ppa(X2, ns_components[1], d=ds[1])			# PPA
	return X3


X3 = ppa_pca_ppa(X_train, (300, 150), (7, 7))
embeddings_final = dict(zip(X_train_names, X3))

