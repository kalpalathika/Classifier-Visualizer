################################################################
# Machine Learning
# Assignment 1 Starting Code
#
# Author: R. Zanibbi
# Author: E. Lima
################################################################

import multiprocessing as mp
import argparse
import numpy as np
# import statistics as stats
import matplotlib.pyplot as plt
import io



################################################################
# Metrics and visualization
################################################################


def conf_matrix(class_matrix, data_matrix, title=None, print_results=False):

    # Initialize confusion matrix with 0's
    confusions = np.zeros((2, 2), dtype=int)

    # Generate output/target pairs, convert to list of integer pairs
    # * '-1' indexing indicates the final column
    # * list(zip( ... )) combines the output and target class label columns into a list of pairs
    out_target_pairs = [
        (int(out), int(target))
        for (out, target) in list(zip(class_matrix[:, -1], data_matrix[:, -1]))
    ]

    # Use output/target pairs to compile confusion matrix counts
    for (out, target) in out_target_pairs:
        confusions[out][target] += 1

    # Compute recognition rate
    inputs_correct = confusions[0][0] + confusions[1][1]
    inputs_total = np.sum(confusions)
    recognition_rate = inputs_correct / inputs_total * 100

    if print_results:
        if title:
            print("\n>>>  " + title)
        print(
            "\n    Recognition rate (correct / inputs):\n    ", recognition_rate, "%\n"
        )
        print("    Confusion Matrix:")
        print("              0: Blue-True  1: Org-True")
        print("---------------------------------------")
        print("0: Blue-Pred |{0:12d} {1:12d}".format(confusions[0][0], confusions[0][1]))
        print("1: Org-Pred  |{0:12d} {1:12d}".format(confusions[1][0], confusions[1][1]))

    return (recognition_rate, confusions)


def draw_results(data_matrix, class_fn, title, file_name):

    # Fix axes ranges so that X and Y directions are identical (avoids 'stretching' in one direction or the other)
    # Use numpy amin function on first two columns of the training data matrix to identify range
    pad = 0.25
    plt.switch_backend('Agg')

    min_tick = np.amin(data_matrix[:, 0:2]) - pad
    max_tick = np.amax(data_matrix[:, 0:2]) + pad
    plt.xlim(min_tick, max_tick)
    plt.ylim(min_tick, max_tick)

    ##################################
    # Grid dots to show class regions
    ##################################

    axis_tick_count = 75
    x = np.linspace(min_tick, max_tick, axis_tick_count, endpoint=True)
    y = np.linspace(min_tick, max_tick, axis_tick_count, endpoint=True)
    (xx, yy) = np.meshgrid(x, y)
    grid_points = np.concatenate(
        (xx.reshape(xx.size, 1), yy.reshape(yy.size, 1)), axis=1
    )

    class_out =  class_fn(grid_points)

    # Separate rows for blue (0) and orange (1) outputs, plot separately with color
    blue_points = grid_points[np.where(class_out[:, 1] < 1.0)]
    orange_points = grid_points[np.where(class_out[:, 1] > 0.0)]

    plt.scatter(
        blue_points[:, 0],
        blue_points[:, 1],
        marker=".",
        s=1,
        facecolors="blue",
        edgecolors="blue",
        alpha=0.4,
    )
    plt.scatter(
        orange_points[:, 0],
        orange_points[:, 1],
        marker=".",
        s=1,
        facecolors="orange",
        edgecolors="orange",
        alpha=0.4,
    )

    ##################################
    # Decision boundary (black line)
    ##################################
    
    # MISSING -- add code to draw class boundaries
    # ONE method for ALL classifier types
    plt.contour(xx, yy, class_out[:, 1].reshape(xx.shape), levels=[0.5])


    ##################################
    # Show training samples
    ##################################

    # Separate rows for blue (0) and orange (1) target inputs, plot separately with color
    blue_targets = data_matrix[np.where(data_matrix[:, 2] < 1.0)]
    orange_targets = data_matrix[np.where(data_matrix[:, 2] > 0.0)]

    plt.scatter(
        blue_targets[:, 0],
        blue_targets[:, 1],
        marker="o",
        facecolors="none",
        edgecolors="blue",
    )
    plt.scatter(
        orange_targets[:, 0],
        orange_targets[:, 1],
        marker="o",
        facecolors="none",
        edgecolors="darkorange",
    )
    ##################################
    # Add title and write file
    ##################################
    # image_stream = io.BytesIO()

    # Set title and save plot if file name is passed (extension determines type)
    # plt.figure()
    plt.title(title)
    # plt.savefig(file_name)
    # print("\nWrote image file: " + file_name)
    # plt.close()
    plot_stream = io.BytesIO()
    plt.savefig(plot_stream, format='png')
    plt.close()

    plot_stream.seek(0)

    return plot_stream.getvalue()
    



################################################################
# Interactive Testing
################################################################
def test_points(data_matrix, beta_hat):

    print("\n>> Interactive testing for (x_1,x_2) inputs")
    stop = False
    while True:
        x = input("\nEnter x_1 ('stop' to end): ")

        if x == "stop":
            break
        else:
            x = float(x)

        y = float(input("Enter x_2: "))
        k = int(input("Enter k: "))

        lc = linear_classifier(beta_hat)
        knn = knn_classifier(k, data_matrix)

        print("   least squares: " + str(lc(np.array([x, y]).reshape(1, 2))))
        print("             knn: " + str(knn(np.array([x, y]).reshape(1, 2))))


################################################################
# Classifiers
################################################################
def least_squares_classifier(input_matrix):
    X = input_matrix[:, 0:2]
    ones_column = np.ones((X.shape[0], 1))
        
    # Concatenate the ones_column with the data_matrix
    X = np.hstack((ones_column, X))
    XT = X.transpose()
    XTX = np.dot(XT, X)
    XTX_inv = np.linalg.inv(XTX)
    XTX_inv_XT = np.dot(XTX_inv, XT)
    y = input_matrix[:, -1]
    beta_hat = np.dot(XTX_inv_XT, y)
    return beta_hat


def linear_classifier(weight_vector):
    # Constructs a linear classifier
    def classifier(input_matrix):
        # Take the dot product of each sample ( data_matrix row ) with the weight_vector (col vector)
        # -- as defined in Hastie equation (2.2)
        data_matrix = input_matrix[:, 0:2]
        ones_column = np.ones((data_matrix.shape[0], 1))
        
        # Concatenate the ones_column with the data_matrix
        data_matrix = np.hstack((ones_column, data_matrix))
        # REVISE: Always choose class 0
        scores = np.dot(data_matrix, weight_vector)
        new_column = np.where(scores > 0.5, 1, 0)
        scores_classes = np.column_stack((scores, new_column))
        
        # Return N x 2 result array
        return scores_classes

    return classifier

def euclidean_distance(x1, x2,y1,y2):    
    return (x1-x2)**2 + (y1-y2)**2

def knn_classifier(k, data_matrix):
    # Constructs a knn classifier for the passed value of k and training data matrix
    def classifier(input_matrix):
        (input_count, _) = input_matrix.shape
        (data_count, _) = data_matrix.shape
        scores = np.array([])

        for i in range(input_count):
            distances = []
            for j in range(data_count):
                distance = euclidean_distance(input_matrix[i][0], data_matrix[j][0], input_matrix[i][1], data_matrix[j][1])
                distances.append([data_matrix[j][2],distance])
            k_nearest_indices = sorted(distances, key=lambda x: x[1])[:k]
            k_nearest_indices = np.array(k_nearest_indices)
            avg = np.mean(k_nearest_indices[:,0])
            scores = np.append(scores, avg)
        result_class = np.where(scores > 0.5, 1, 0)
        scores_classes = np.column_stack((scores, result_class))

        # Return N x 2 result array
        return scores_classes

    return classifier

def get_data_matrix():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("data_file", help="numpy data matrix file containing samples")
    # args = parser.parse_args()

    # Load data
    data_matrix = np.load("./linear_knn/data.npy")
    print("\nLoaded data matrix of size " + str(data_matrix.shape))
    return data_matrix


################################################################
# Main function
################################################################


def main():
    data_matrix = get_data_matrix()
    # Process arguments using 'argparse'
    (confm, rr) = conf_matrix(
        data_matrix, data_matrix, "Data vs. Itself Sanity Check", print_results=True
    )

    # Construct linear classifier
    lc = linear_classifier(least_squares_classifier(data_matrix))
    lsc_out = lc(data_matrix)

    # Compute results on training set
    conf_matrix(lsc_out, data_matrix, "Least Squares", print_results=True)
    draw_results(data_matrix, lc, "Least Squares Linear Classifier", "ls.pdf")

    # Nearest neighbor
    for k in [1, 15]:
        knn = knn_classifier(k, data_matrix)
        knn_out = knn(data_matrix)
        conf_matrix(knn_out, data_matrix, "knn: k=" + str(k), print_results=True)
        draw_results(
            data_matrix,
            knn,
            "k-NN Classifier (k=" + str(k) + ")",
            "knn-" + str(k) + ".pdf",
        )

    # Interactive testing
    test_points(data_matrix, np.array([1,1,1]))


if __name__ == "__main__":
    main()
