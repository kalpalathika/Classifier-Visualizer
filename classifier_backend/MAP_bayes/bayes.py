################################################################
# Machine Learning
# Programming Assignment - Bayesian Classifier
#
# bayes.py - functions for Bayesian classifier
#
# Author: R. Zanibbi
# Author: E. Lima
################################################################

import math
import numpy as np

# from debug import *
# from debug import *
# from results_visualization import *
import sys
max_int = sys.maxsize
min_int = -sys.maxsize - 1


################################################################
# Cost matrices
################################################################

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def uniform_cost_matrix(num_classes):
    cost_matrix = np.ones((num_classes, num_classes))
    np.fill_diagonal(cost_matrix, 0)
    print("Uniform cost matrix:\n", cost_matrix)
    return cost_matrix


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def bnrs_unequal_costs( num_classes ):
    cost_matrix = np.ones((num_classes, num_classes))
    new_array = np.array([[-0.20, 0.07, 0.07, 0.07],
                          [0.07, -0.15, 0.07, 0.07],
                          [0.07, 0.07, -0.05, 0.07],
                          [0.03, 0.03, 0.03, 0.03]])

    cost_matrix = np.full_like(cost_matrix, new_array)
    return cost_matrix 
################################################################
# Bayesian parameters 
################################################################

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def priors( split_data ):
    est_priors = []
    total_rows = 0
    for group in split_data:
        total_rows += len(group)
    for i in range(len(split_data)):
        est_priors.append(len(split_data[i])/total_rows)
    return est_priors

def bayesian_parameters( CLASS_DICT, split_data, title='' ):
    # Compute class priors, means, and covariances matrices WITH their inverses (as pairs)
    class_priors = priors(split_data)
    class_mean_vectors = list( map( mean_vector, split_data ) )
    class_cov_matrices = list( map( covariances, split_data ) )

    # Show parameters if title passed
    # if title != '':
    #     print('>>> ' + title)
    #     show_for_classes(CLASS_DICT, "[ Priors ]", class_priors )

    #     show_for_classes(CLASS_DICT, "[ Mean Vectors ]", class_mean_vectors)
    #     show_for_classes(CLASS_DICT, '[ Covariances and Inverse Covariances]', class_cov_matrices )
    #     print('')

    return (class_priors, class_mean_vectors, class_cov_matrices)


################################################################
# Gaussians (for class-conditional density estimates) 
################################################################

def mean_vector( data_matrix ):
    # Axis 0 is along columns (default)
    return np.mean( data_matrix, axis=0)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def covariances( data_matrix ):
    # HEADS-UP: The product of the matrix by its inverse may not be identical to the identity matrix
    #           due to finite precision. Can use np.allclose() to test for 'closeness'
    #           to ideal identity matrix (e.g., np.eye(2) for 2D identity matrix)
    d = data_matrix.shape[1]
    # Calculate the covariance matrix
    covariance_matrix = np.cov(data_matrix,rowvar=False)
    try:
        inverse_covariance_matrix = np.linalg.inv(covariance_matrix)
    except np.linalg.LinAlgError:
        inverse_covariance_matrix = None
    # Returns a pair: ( covariance_matrix, inverse_covariance_matrix )
    return (covariance_matrix, inverse_covariance_matrix)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def mean_density( cov_matrix ):
    det = np.linalg.det(cov_matrix)
    d = cov_matrix.shape[0]
    mean_density = 1.0 / (np.sqrt(det) * (2 * np.pi) ** (d / 2))
    return mean_density 

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def sq_mhlnbs_dist( data_matrix, mean_vector, cov_inverse ):
    # Square of distance from the mean in *standard deviations* 
    # (e.g., a sqared mahalanobis distance of 9 implies a point is sqrt(9) = 3 standard
    # deviations from the mean.

    # Numpy 'broadcasting' insures that the mean vector is subtracted row-wise
    diff = data_matrix - mean_vector
    mahalanobis_sq = np.sum(np.dot(diff, cov_inverse) * diff, axis=1)
    return mahalanobis_sq   
 
def gaussian( mean_density, distances ):
    # NOTE: distances is a column vector of squared mahalanobis distances

    # Use numpy matrix op to apply exp to all elements of a vector
    scale_factor = np.exp( -0.5 * distances )

    # Returns Gaussian values as the value at the mean scaled by the distance
    return mean_density * scale_factor


def map_classifier(priors, mean_vectors, covariance_pairs):
    covariances = np.array([cov_pair[0] for cov_pair in covariance_pairs])
    inv_covariances = np.array([cov_pair[1] for cov_pair in covariance_pairs])
    num_classes = len(priors)
    peak_scores = priors * np.array([mean_density(c) for c in covariances])

    def classifier(data_matrix):
        num_samples = data_matrix.shape[0]
        distances = np.zeros((num_samples, num_classes))
        class_scores = np.zeros((num_samples, num_classes + 1))

        for i in range(num_classes):
            distances[:, i] = sq_mhlnbs_dist(data_matrix, mean_vectors[i], inv_covariances[i])
            class_scores[:, i] = gaussian(peak_scores[i], distances[:, i])

        class_scores[:, -1] = np.argmax(class_scores[:, :-1], axis=1)
        return class_scores

    return classifier

def bayes_classifier(cost_matrix, priors, mean_vectors, covariance_pairs):
    covariances = np.array([cov_pair[0] for cov_pair in covariance_pairs])
    inv_covariances = np.array([cov_pair[1] for cov_pair in covariance_pairs])
    num_classes = len(priors)
    peak_scores = priors * np.array([mean_density(c) for c in covariances])

    def classifier(data_matrix):
        num_samples = data_matrix.shape[0]
        distances = np.zeros((num_samples, num_classes))
        class_posteriors = np.zeros((num_samples, num_classes))
        class_costs_output = np.zeros((num_samples, num_classes + 1))

        for i in range(num_classes):
            distances[:, i] = sq_mhlnbs_dist(data_matrix, mean_vectors[i], inv_covariances[i])
            class_posteriors[:, i] = gaussian(peak_scores[i], distances[:, i])

        for i in range(num_samples):
            costs = np.sum(class_posteriors[i] * cost_matrix, axis=1)
            min_cost_index = np.argmin(costs)
            class_costs_output[i, :-1] = costs
            class_costs_output[i, -1] = min_cost_index

        return class_costs_output

    return classifier



