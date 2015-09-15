import numpy as np
from scipy import sparse
import os
import sparsio

def make_sparse():
    os.chdir('/Users/carsten/Downloads/Numpy')
    dataset = np.load("glurextents.npy")
    out = sparse.csr_matrix(dataset)
    del(dataset)
    sparsio.save_sparse_csr("glurextents-sparse", out)

if __name__ == '__main__':
    make_sparse()
