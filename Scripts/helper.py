"""
These functions are not a part of the actual project
but are used to extract useful information that helps
with rest of the project code

"""

from torch._C import FileCheck
from Constants import IMG_FOLDER
from Utils.util import UtilityFunctions
import os 

def max_slices ():

    """
    Find maximum number of slices in the
    patient volumes
    """

    filenames = os.listdir (IMG_FOLDER)
    max_slice_no = 0
    
    for filename in filenames:
        _, slice_no = UtilityFunctions.extract_patient_slice (filename)
        slice_no = int (slice_no)
        if slice_no > max_slice_no:
            max_slice_no = slice_no

    print ("Max Slice no = ", max_slice_no)


def digit_sum (N):

    N_sum = 0
    while N//10 != 0:
        N_sum += N%10
        N = N // 10
    
    N_sum += N
    return N_sum


def solution(N):
    # write your code in Python 3.6
    
    N_sum = digit_sum(N)
    print (N_sum)
    for i in range (N+1, 50000):
        if digit_sum(i) == N_sum:
            return i


import numpy as np


def augmented_euclidean_dist(a, b):
    # Write your code here.
    # Remember to return the right object.


    if (a==b).all():
        return 0.0

    drop_idx_a = a != -999
    drop_idx_b = b != -999

    mask = drop_idx_a & drop_idx_b

    a_updated, b_updated = a[mask], b[mask]

    if len(a_updated) < 2:
        return np.Infinity
    
    return np.linalg.norm (a_updated - b_updated)


if __name__ == "__main__":

    a = np.array([1, 2, 3.5, 4.24])
    b = np.array([2, 1, -999, -999])

    print (augmented_euclidean_dist(a,b))



