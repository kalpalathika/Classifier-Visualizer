################################################################
# a3_q1.py 
#
# (SVM) Assignment 3 Question 1 (ML Fall 2022)
################################################################
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from debug import *

from data_sets_a3 import *
from classifiers import *
from SVM import results_visualization
# timer = DebugTimer("Initialization")

 
def show_score( data, data_tuple, kernel_type='linear', C_value=0.1, gamma_value=0.1, show_setting=False ):
    # Note: gamma value ignored for linear kernel
    clf = svm.SVC(kernel=kernel_type, C=C_value, gamma=gamma_value)
    train_test_clf(clf, "A1 Train for " + str(clf), data_tuple)

    # NOTE: Setting show=True will make the plot interactive; initial program only writes files to disk
    # * Left click to rotate the 3D plot
    # * Right click and move mouse up/down to zoom
    results_visualization.draw_contours(data, clf.decision_function, title="A1 Scores for " + str(clf), class_formats=A1_CLASS_FORMAT, show=show_setting)

# def main():
    # timer.qcheck("Imports complete")

    # ( a1_data_tuple, a1_data ) = load_a1_data(timer)

    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(">>>> TRAIN / RUN SVMs ")
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    # # RUNNNG individual classifiers
    # show_score( a1_data, a1_data_tuple )
    # show_score( a1_data, a1_data_tuple, kernel_type='rbf', C_value=0.1, gamma_value=0.1 )
    # show_score( a1_data, a1_data_tuple, kernel_type='rbf', C_value=10, gamma_value=10 )

    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(">>>> GRID SEARCH ")
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    # # Small grid search
    # C_values = [ 0.1,  1.0, 10 ]
    # kernels = [ 'linear', 'rbf' ]
    # gamma_values = [ 0.1, 1.0, 10 ]

    # for kernel_value in kernels:
    #     for C_value in C_values:
    #         if kernel_value == 'linear':
    #             clf = svm.SVC(kernel='linear', C=C_value)
    #             train_test_clf(clf, "A1 (Orange/Blue) data", a1_data_tuple)
    #             draw_results( a1_data, clf.decision_function, 'Q1 ' + str(clf), A1_CLASS_FORMAT)
    #             print("---- SUPPORT VECTORS:" + str(clf.n_support_))
    #         else:
    #             for gamma_value in gamma_values:
    #                 clf = svm.SVC(kernel=kernel_value, C=C_value, gamma=gamma_value)
    #                 train_test_clf( clf, "A1 (Orange/Blue) data", a1_data_tuple)
    #                 draw_results( a1_data, clf.decision_function, 'Q1 ' + str(clf), A1_CLASS_FORMAT)
    #                 print("----- SUPPORT VECTORS:" + str(clf.n_support_))

    # timer.qcheck("Q1 program complete")
    # print(timer)


# if __name__ == "__main__":
#     main()
