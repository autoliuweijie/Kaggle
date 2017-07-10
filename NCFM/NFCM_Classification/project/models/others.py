'''
    some useful function
    @author: Liu Weijie
'''
import numpy as np


def unison_shuffled_copies(a, b):
    '''
    shuffle a and b in copy
    '''
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
