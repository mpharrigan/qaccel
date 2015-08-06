#cython: boundscheck=False, wraparound=False

cdef int update_counts(int[:,:] counts, int[:] states, int prev):
    counts[prev, states[0]] += 1
    cdef int i
    for i in range(states.shape[0] - 1):
        counts[states[i], states[i+1]] += 1
    return states[states.shape[0]-1]

cdef int update_counts_first(int[:,:] counts, int[:] states):
    cdef int i
    for i in range(states.shape[0] - 1):
        counts[states[i], states[i+1]] += 1
    return states[states.shape[0]-1]


import numpy as np

def make_counts(chunkedtrajs, n_states, dref):
    cdef int[:,:] counts = np.zeros((n_states, n_states), dtype=np.int32)
    cdef int last
    for chunks in chunkedtrajs:
        fc = np.asarray(dref(chunks[0]), dtype=np.int32)
        fc.flags['WRITEABLE'] = True # see [1]
        last = update_counts_first(counts, fc)
        for c in chunks[1:]:
            c2 = np.asarray(dref(c), dtype=np.int32)
            c2.flags['WRITEABLE'] = True # see [1]
            last = update_counts(counts, c2, last)

    return np.asarray(counts)

# [1] https://mail.python.org/pipermail/cython-devel/2013-February/003384.html
