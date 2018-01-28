import numpy as np
import numpy.random as rn
import sktensor as skt

import pickle
from path import Path as path
from time import sleep


def sp_uttkrp(vals, subs, m, U):
    """Alternative implementation of the sparse version of the uttkrp.
    ----------
    subs : n-tuple of array-likes
        Subscripts of the nonzero entries in the tensor.
        Length of tuple n must be equal to dimension of tensor.
    vals : array-like
        Values of the nonzero entries in the tensor.
    U : list of array-likes
        Matrices for which the Khatri-Rao product is computed and
        which are multiplied with the tensor in mode `mode`.
    m : int
        Mode in which the Khatri-Rao product of `U` is multiplied
        with the tensor.
    Returns
    -------
    out : np.ndarray
        Matrix which is the result of the matrix product of the unfolding of
        the tensor and the Khatri-Rao product of `U`.
    """

    D, K = U[m].shape
    out = np.zeros_like(U[m])
    for k in xrange(K):
        tmp = vals.copy()
        for mode, matrix in enumerate(U):
            if mode == m:
                continue
            tmp *= matrix[subs[mode], k]
        out[:, k] += np.bincount(subs[m],
                                 weights=tmp,
                                 minlength=D)
    return out


def parafac(matrices, axis=None):
    """Computes the PARAFAC of a set of matrices.

    For a set of N matrices,

        U = {U1, U2, ..., UN}

    where Ui is size (Di X K),

    PARAFAC(U) is defined as the sum of column-wise outer-products:

        PARAFAC(U) = \sum_k U1(:, k) \circ \dots \circ UN(:, k)

    and results in a tensor of size D1 x ... x DN.

    Calls np.einsum repeatedly (instead of all at once) to avoid memory
    usage problems that occur when too many matrices are passed to einsum.

    Parameters
    ----------
    matrices : list of array-likes
        Matrices for which the PARAFAC is computed
    axis : int, optional
        The axis along which all matrices have the same dimensionality.
        Either 0 or 1.  If set to None, it will check which axis
        all matrices agree on. If matrices are square, it defaults to 1.
    Returns
    -------
    out : np.ndarray
        ndarray which is the result of the PARAFAC
    """
    assert len(matrices) > 1
    if axis is None:
        N, M = matrices[0].shape
        axis_0_all_equal = all([X.shape[0] == N for X in matrices[1:]])
        axis_1_all_equal = all([X.shape[1] == M for X in matrices[1:]])
        if axis_1_all_equal:
            axis = 1
        elif axis_0_all_equal:
            axis = 0
        else:
            raise ValueError('Matrices not aligned.')

    if len(matrices) == 2:
        s = 'za,zb->ab' if axis == 0 else 'az,bz->ab'
        return np.einsum(s, matrices[0], matrices[1])
    else:
        s = 'za,zb->zab' if axis == 0 else 'az,bz->abz'
        tmp = np.einsum(s, matrices[0], matrices[1])
        curr = 'ab'

        letters = list('cdefghijklmnopqrstuv')
        for matrix in matrices[2:-1]:
            ltr = letters.pop(0)
            if axis == 0:
                s = 'z%s,z%s->z%s%s' % (curr, ltr, curr, ltr)
            else:
                s = '%sz,%sz->%s%sz' % (curr, ltr, curr, ltr)
            tmp = np.einsum(s, tmp, matrix)
            curr += ltr

        ltr = letters.pop(0)
        if axis == 0:
            s = 'z%s,z%s->%s%s' % (curr, ltr, curr, ltr)
        else:
            s = '%sz,%sz->%s%s' % (curr, ltr, curr, ltr)
        return np.einsum(s, tmp, matrices[-1])


def serialize_bptf(model, out_dir, desc=None):
    out_dir = path(out_dir)
    assert out_dir.exists()

    out_path = out_dir.joinpath('%s.npz' %(desc))
    np.savez(out_path,
             E_DK_M=model.E_DK_M,
             G_DK_M=model.G_DK_M)
    print out_path