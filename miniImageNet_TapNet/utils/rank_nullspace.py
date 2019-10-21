import numpy as np
from numpy.linalg import svd

import cupy as cp
from cupy.linalg import svd as svd_gpu
from cupy import core

def rank(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol*s[0])
    rank = int((s >= tol).sum())
    return rank

def nullspace(A, tol=1e-13):
    A=np.atleast_2d(A)
    u, s, vh = svd(A)
    if len(A.shape) == 2:
        nnz = (s >= tol).sum()
        ns = vh[nnz:].conj().T
    elif len(A.shape) == 3:
        nnz = (s >= tol).sum(axis=-1)
        nnz = max(nnz)
        ns = np.transpose(vh[:,nnz:,:].conj(), axes=[0,2,1])
    return ns

def nullspace_gpu(A, tol=1e-13):
    A = cp.atleast_2d(A)
    u, s, vh =svd_gpu(A)
    nnz = (s >=  tol).sum()
    ns = vh[nnz:].conj().T
    return ns 
