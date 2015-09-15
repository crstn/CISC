import numpy as np
from scipy import sparse

# See http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format
def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

# Testing:
# csr = load_sparse_csr("/Users/carsten/Downloads/Numpy/glurextents-sparse.npz")
# print csr.shape
# print csr.size
