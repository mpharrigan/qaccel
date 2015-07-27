#cython: boundscheck=False, wraparound=False

cdef int update_counts(int[:,:] counts, int[:] states, int prev):
    counts[prev, states[0]] += 1
    cdef int i
    for i in range(counts.shape[0] - 1):
        counts[states[i], states[i+1]] += 1
    return states[states.shape[0]-1]

cdef int update_counts_first(int[:,:] counts, int[:] states):
    cdef int i
    for i in range(counts.shape[0] - 1):
        counts[states[i], states[i+1]] += 1
    return states[states.shape[0]-1]


import numpy as np

def make_counts(chunkedtrajs, n_states, dref):
    cdef int[:,:] counts = np.zeros((n_states, n_states), dtype=np.int32)
    cdef int last;
    for chunks in chunkedtrajs:
        fc, *ocs = chunks
        last = update_counts_first(counts, dref(fc))
        for c in chunks:
            last = update_counts(counts, dref(c), last)
