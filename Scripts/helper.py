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


if __name__ == "__main__":
    max_slices()



