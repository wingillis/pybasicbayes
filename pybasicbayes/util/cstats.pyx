import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t

from cython.parallel cimport prange


cdef inline int32_t csample_discrete_normalized(double[::1] distn, double u):
    cdef int i
    cdef int N = distn.shape[0]
    cdef double tot = u

    for i in range(N):
        tot -= distn[i]
        if tot < 0:
            break

    return i


def sample_markov(int T, trans_matrix, init_state_distn):
    # Normalize in numpy before binding typed memoryviews: memoryviews do
    # not expose ndarray methods, so normalizing after binding (as the
    # previous revision did) raised AttributeError at runtime.
    trans_matrix = np.ascontiguousarray(trans_matrix, dtype=np.double)
    init_state_distn = np.ascontiguousarray(init_state_distn, dtype=np.double)

    cdef double[:, ::1] A = trans_matrix / trans_matrix.sum(1)[:, None]
    cdef double[::1] pi = init_state_distn / init_state_distn.sum()

    cdef int32_t[::1] out = np.empty(T, dtype=np.int32)
    cdef double[::1] randseq = np.random.random(T)

    cdef int t
    out[0] = csample_discrete_normalized(pi, randseq[0])
    for t in range(1, T):
        out[t] = csample_discrete_normalized(A[out[t - 1]], randseq[t])

    return np.asarray(out)


def sample_crp_tablecounts(concentration, customers, colweights=None):
    # Sample Chinese restaurant process table counts (used by HDP
    # transition and concentration resampling). All inputs are converted
    # to concrete dtypes up front so the typed kernels below are simple.
    concentration = float(concentration)
    _customers = np.ascontiguousarray(customers, dtype=np.int64)
    if colweights is not None:
        _colweights = np.ascontiguousarray(colweights, dtype=np.double)
    else:
        _colweights = np.ones(_customers.shape[1], dtype=np.double)

    cdef long[:, ::1] cust = _customers
    cdef double[::1] w = _colweights
    cdef long[:, ::1] m = np.zeros_like(_customers)
    cdef long tot = _customers.sum()

    cdef double[::1] randseq = np.random.random(tot)

    tmp = np.empty_like(_customers)
    tmp[0, 0] = 0
    tmp.flat[1:] = np.cumsum(
        np.ravel(_customers)[:_customers.size - 1], dtype=tmp.dtype)
    cdef long[:, ::1] starts = tmp

    cdef int i, j, k
    cdef double conc = concentration

    with nogil:
        for i in prange(cust.shape[0]):
            for j in range(cust.shape[1]):
                for k in range(cust[i, j]):
                    m[i, j] += randseq[starts[i, j] + k] \
                        < (conc * w[j]) / (k + conc * w[j])

    return np.asarray(m)
