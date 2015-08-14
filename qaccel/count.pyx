#cython: boundscheck=False, wraparound=False


cdef int update_counts(int[:,:] counts, int[:] states):
    cdef int i
    for i in range(states.shape[0] - 1):
        counts[states[i], states[i+1]] += 1
    return states[states.shape[0]-1]


import numpy as np

def make_counts(chunkedtrajs, n_states):
    cdef int[:,:] counts = np.zeros((n_states, n_states), dtype=np.int32)
    for chunks in chunkedtrajs:
        for c in chunks:
            last = update_counts(counts, np.asarray(c, dtype=np.int32))

    return np.asarray(counts)
