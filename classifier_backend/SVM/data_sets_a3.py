################################################################
# data_sets_a3.py 
#
# Methods for data loading and processing
#
# Author: R. Zanibbi
# Author: E. Lima
################################################################
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np

################################################################
# Class labels / attributes for plots
################################################################

A1_CLASS = { 0: 'Blue', 1: 'Orange' }
BNRS_CLASS = { 0: 'Bolts', 1: 'Nuts', 2: 'Rings', 3: 'Scrap' }

A1_CLASS_FORMAT = { 0: ('o','blue'), 1: ('o','orange') }
BNRS_CLASS_FORMAT = { 0: ('+', 'blue'), 1: ('*', 'red'), 2: ('o', 'yellow'), 3: ('x', 'black') }

################################################################
# Data set functions / test functions
################################################################

def split_labels( data_matrix ):
    # Assumes labels are in the final column
    return ( data_matrix[:,0:-1], data_matrix[:,-1] )

def split_rows_on_class_labels( sample_array, debug=False ):
    index_positions = np.argsort( sample_array[:, -1] )
    sorted_array = sample_array[ index_positions ]

    # Collect feature vectors and corresponding labels AFTER sort 
    X = sorted_array[:,:-1]
    y = sorted_array[:,-1]

    # Produce *list* of data arrays w. one array per class (( represented by list index ))
    # * np.diff( data[:,2] ) -- computes differences between rows in the label column
    # * np.flatnonzero -- returns vector of INDICES for non-zero elements in vector of differences
    #     NOTE: +1 corrects index offsets (N item list -> N-1 differences)
    label_diffs = np.diff( sorted_array[:,-1] )
    split_indices = np.flatnonzero( label_diffs ) + 1

    sorted_no_labels = sorted_array[:,:-1]
    split_data = np.vsplit( sorted_no_labels, split_indices )

    # Debugging
    if debug:
        dcheck( 'split shape', [ np.shape(a) for a in split_data ] )
        dnpcheck( 'Top elements from list of arrays in split', [ a[:3,:] for a in split_data ] )

    # Return split data, along with data matrix and corresponding labels in y
    return ( split_data, X, y )

################################################################
# Loading data sets
################################################################

def load_a1_data( timer ):
   # Load a1 data
    ob_data = np.load('./SVM/data/data_a1.npy')
    ( a1_data, a1_y ) = split_labels( ob_data )
    a1_tuple = ( a1_data, a1_data, a1_y, a1_y )

    timer.qcheck("A1 (Orange/Blue) data loaded")
    return ( a1_tuple, ob_data )

def load_a2_data( timer ):
    # For 'metal parts' data, decrease class labels values by 1
    a2_data = np.loadtxt('data/data_bnrs.csv', delimiter=',') - np.array([0,0,1])
    ( metal_data, metal_y ) = split_labels( a2_data )
    a2_tuple = ( metal_data, metal_data, metal_y, metal_y )

    timer.qcheck("A2 (Metal Parts) data loaded")
    return ( a2_tuple, a2_data )

def load_small_mnist( timer ):
    ###############################################################################
    # (this comment from sklearn example: 
    #  https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html) 
    #
    # Digits dataset
    # --------------
    #
    # The digits dataset consists of 8x8
    # pixel images of digits. The ``images`` attribute of the dataset stores
    # 8x8 arrays of grayscale values for each image. We will use these arrays to
    # visualize the first 4 images. The ``target`` attribute of the dataset stores
    # the digit each image represents and this is included in the title of the 4
    # plots below.
    #
    # Note: if we were working from image files (e.g., 'png' files), we would load
    # them using :func:`matplotlib.pyplot.imread`.
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    timer.qcheck("Small MNIST load")

    # Split data into 50% train and 50% test subsets
    # Without shuffling (i.e., randomizing) the data 
    # (i.e., split using provided order of samples)
    data_tuple = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )
    timer.qcheck("Small MNIST train/test split")
    return ( digits, data_tuple )


def load_a3_data( timer ):
    
    ( a1_tuple, ob_data ) = load_a1_data(timer)
    ( a2_tuple, a2_data ) = load_a2_data(timer)
    mnist_tuple = load_small_mnist(timer) 
    print(timer)
    
    return ( mnist_tuple, a1_tuple, a2_tuple, ob_data, a2_data )
