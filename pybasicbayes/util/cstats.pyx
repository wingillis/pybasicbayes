import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t

from cython cimport floating, integral
from cython.parallel import prange, parallel

ctypedef fused floating_t:
    float
    double

ctypedef fused integral_t:
    int
    long

cdef inline int32_t csample_discrete_normalized(floating[::1] distn, floating u):
    cdef int i
    cdef int N = distn.shape[0]
    cdef floating tot = u

    for i in range(N):
        tot -= distn[i]
        if tot < 0:
            break

    return i

def sample_markov(
        int T,
        floating_t[:, ::1] trans_matrix,
        floating_t[::1] init_state_distn
        ):
    cdef int32_t[::1] out = np.empty(T, dtype=np.int32)
    cdef floating_t[:,::1] A = trans_matrix / trans_matrix.sum(1)[:, None]
    cdef floating_t[::1] pi = init_state_distn / init_state_distn.sum()

    cdef floating_t[::1] randseq = np.random.random(T)

    cdef int t
    out[0] = csample_discrete_normalized(pi,randseq[0])
    for t in range(1,T):
        out[t] = csample_discrete_normalized(A[out[t-1]],randseq[t])

    return np.asarray(out)

def sample_crp_tablecounts(
        floating_t concentration,
        integral_t[:, :] customers,
        floating_t[:] colweights=None
        ):
    cdef integral_t[:,::1] _customers = np.require(customers, requirements='C')
    cdef integral_t[:,::1] m = np.zeros_like(_customers)
    cdef floating_t[::1] _colweights = np.require(colweights, requirements='C') \
        if colweights is not None else np.ones(customers.shape[1])
    cdef int i, j, k
    cdef integral_t tot = np.sum(_customers)

    cdef floating_t[::1] randseq = np.random.random(tot)

    tmp = np.empty_like(_customers)
    tmp[0,0] = 0
    tmp.flat[1:] = np.cumsum(np.ravel(customers)[:_customers.size-1],dtype=tmp.dtype)
    cdef integral_t[:,::1] starts = tmp

    with nogil:
        for i in prange(_customers.shape[0]):
            for j in range(_customers.shape[1]):
                for k in range(_customers[i,j]):
                    m[i,j] += randseq[starts[i,j]+k] \
                        < (concentration * _colweights[j]) / (k+concentration*_colweights[j])

    return np.asarray(m)
