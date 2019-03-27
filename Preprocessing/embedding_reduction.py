
import numpy as np
from sklearn.decomposition import PCA

from embedding_vocabulary import get_word_embedding, dict_to_2arrays


def pca_reduce(X, n_components=100):
    """ PCA """
    assert X.shape[1] >= n_components, "n_components shouldn't be greater than shape of X"
    pca = PCA(n_components=n_components)
    X_mean = X - np.mean(X)
    X_pca = pca.fit_transform(X_mean)
    U1 = pca.components_
    return X_mean, X_pca, U1

def ppa(X, d=7):
    """ PPA """
    X_mean, _, U1 = pca_reduce(X, X.shape[1])  # Get Components Ranked
    X2 = []
    for i, x in enumerate(X_mean):
        for u in U1[:d]:						   # Remove Projections on Top-d Components
            x = x - np.dot(u.transpose(), x) * u
        X2.append(x)
    return np.asarray(X2)

def pca_ppa(X1, n_components=100, d=7):
    """ PCA->PPA """
    _, X2, _ = pca_reduce(X1, n_components)	# PCA
    X3 = ppa(X2, d=d)							# PPA
    return X3

def ppa_pca(X0, n_components=100, d=7):
    """ PPA->PCA """
    X1 = ppa(X0, d=d)							# PPA
    _, X2, _ = pca_reduce(X1, n_components)	# PCA
    return X2

def ppa_pca_ppa(X0, n_components=100, ds=(7, 7)):
    """ PPA->PCA->PPA """
    X1 = ppa(X0, d=ds[0])						# PPA
    _, X2, _ = pca_reduce(X1, n_components)	# PCA
    X3 = ppa(X2, d=ds[1])						# PPA
    return X3


def embedding_reduce(word2vector, method='all', n_components=50, d=7, ds=(7, 7)):
    """ word2vector即为word embedding，对其降维 """
    arr_word, arr_vector = dict_to_2arrays(word2vector, sortby=0)
    if method == 'pca':
        _, arr_reduced, _ = pca_reduce(arr_vector, n_components)
    elif method == 'ppa':
        arr_reduced = ppa(arr_vector, d=d)
    elif method == 'pcappa':
        arr_reduced = pca_ppa(arr_vector, n_components, d=d)
    elif method == 'ppapca':
        arr_reduced = ppa_pca(arr_vector, n_components, d=d)
    elif method == 'ppapcappa':
        arr_reduced = ppa_pca_ppa(arr_vector, n_components, ds=ds)
    else:
        print('Invalid method! Valid methods are pca, ppa, pcappa, ppapca and ppapcappa.')
    return dict(zip(arr_word, arr_reduced))
    


if __name__ == '__main__':
    
    # 避免使用全量的Word Embedding，在使用前应该先经词汇表进行过滤，否则数据量巨大，降维操作无法执行或执行较慢
    word_embedding_file = 'word_embedding.txt'   # 格式：word\tval1,val2,val3,...  无header
    word_embedding = get_word_embedding(word_embedding_file, header=False, seps=('\t', ','))
    # 使用PPA-PCA-PPA算法对Embedding进行降维
    word_embedding_reduced = embedding_reduce(word_embedding, method='ppapcappa', n_components=50, ds=(7, 7))
