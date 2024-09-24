from linear_knn import a1_solution
from flask import Response
import numpy as np
import base64
from MAP_bayes import bayes
from MAP_bayes import a2_main
from MAP_bayes import results_visualization
from sklearn import svm, metrics
from SVM import debug, data_sets_a3, classifiers as svm_classifier, a3_q1
from SVM.results_visualization import draw_results as svm_draw_results, draw_contours as svm_draw_contours, A1_CLASS_FORMAT as SVM_A1_CLASS_FORMAT, conf_matrix as svm_conf_matrix
timer = debug.DebugTimer("Initialization")

A1_CLASS = { 0: 'Blue', 1: 'Orange' }
A1_CLASS_FORMAT = { 0: ('o','blue'), 1: ('o','orange') }



def knn_helper(knn_number= 1):
    data_matrix = a1_solution.get_data_matrix()
    knn = a1_solution.knn_classifier(knn_number, data_matrix)
    knn_out = knn(data_matrix)
    (recognition_rate, confusions)= a1_solution.conf_matrix(knn_out, data_matrix)
    # (recognition_rate, confusions) = a1_solution.conf_matrix(knn, data_matrix)
    confusions_list = confusions.tolist()

    plot_blob =  a1_solution.draw_results(
            data_matrix,
            knn,
            "k-NN Classifier (k=" + str(1) + ")",
            "knn-" + str(1) + ".pdf",
        )
    plot_blob_base64 = base64.b64encode(plot_blob).decode('utf-8')
    return {"prediction_blob": plot_blob_base64, "recognition_rate": recognition_rate, "confusion_matrix": confusions_list}

def load_MAP_bayes_data():
    ################################
    # Class dicts, formatting
    # & load data
    ##################################
 
    # For 'metal parts' data, decrease class labels values by 1
    data = np.load('MAP_bayes/data/data.npy')
    ( a1_split_data, a1_X, a1_y ) = a2_main.split_rows_on_class_labels( data )

    (a1_class_priors, a1_class_mean_vectors, a1_class_cov_matrices ) = \
            bayes.bayesian_parameters( A1_CLASS, a1_split_data, title='A1 Parameters')
    return (a1_class_priors, a1_class_mean_vectors, a1_class_cov_matrices, data, a1_X, a1_y)

def MAP_non_uniform_helper():
    (a1_class_priors, a1_class_mean_vectors, a1_class_cov_matrices, data, a1_X, a1_y) = load_MAP_bayes_data()
    a1_non_uniform = bayes.map_classifier( a1_class_priors, a1_class_mean_vectors, a1_class_cov_matrices )
    map_non_uniform_blob = results_visualization.draw_results(data, a1_non_uniform, "A1 MAP", "A1_MAP.pdf", A1_CLASS_FORMAT, axis_tick_count=1000)
    plot_blob_base64 = base64.b64encode(map_non_uniform_blob).decode('utf-8')
    (recognition_rate, confusions) = results_visualization.conf_matrix(A1_CLASS, a1_non_uniform( a1_X ), data, title='A1 MAP')
    confusions_list = confusions.tolist()
    return {"prediction_blob": plot_blob_base64, "recognition_rate": recognition_rate, "confusion_matrix": confusions_list}


def MAP_uniform_helper():
    (a1_class_priors, a1_class_mean_vectors, a1_class_cov_matrices, data, a1_X, a1_y) = load_MAP_bayes_data()
    a1_map = bayes.bayes_classifier( bayes.uniform_cost_matrix( len(A1_CLASS) ), a1_class_priors, a1_class_mean_vectors, a1_class_cov_matrices )
    map_uniform_blob = results_visualization.draw_results(data, a1_map, "A1 BAYES", "A1_Bayes.pdf", A1_CLASS_FORMAT, axis_tick_count=1000)
    plot_blob_base64 = base64.b64encode(map_uniform_blob).decode('utf-8')
    (recognition_rate, confusions) = results_visualization.conf_matrix(A1_CLASS, a1_map( a1_X ), data, title='A1 Uniform Cost Bayesian')
    confusions_list = confusions.tolist()
    return {"prediction_blob": plot_blob_base64, "recognition_rate": recognition_rate, "confusion_matrix": confusions_list}


def svm_helper():
    timer.qcheck("Imports complete")

    ( a1_data_tuple, a1_data ) = data_sets_a3.load_a1_data(timer)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(">>>> TRAIN / RUN SVMs ")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    # RUNNNG individual classifiers
    a3_q1.show_score( a1_data, a1_data_tuple )
    a3_q1.show_score( a1_data, a1_data_tuple, kernel_type='rbf', C_value=0.1, gamma_value=0.1 )
    a3_q1.show_score( a1_data, a1_data_tuple, kernel_type='rbf', C_value=10, gamma_value=10 )

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(">>>> GRID SEARCH ")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    # Small grid search
    C_values = [ 0.1,  1.0, 10 ]
    kernels = [ 'linear', 'rbf' ]
    gamma_values = [ 0.1, 1.0, 10 ]

    # for kernel_value in kernels:
    #     for C_value in C_values:
    #         if kernel_value == 'linear':
    #             clf = svm.SVC(kernel='linear', C=C_value)
    #             svm_classifier.train_test_clf(clf, "A1 (Orange/Blue) data", a1_data_tuple)
    #             svm_draw_results( a1_data, clf.decision_function, 'Q1 ' + str(clf), SVM_A1_CLASS_FORMAT)
    #             print("---- SUPPORT VECTORS:" + str(clf.n_support_))
    #         else:
    #             for gamma_value in gamma_values:
    #                 clf = svm.SVC(kernel=kernel_value, C=C_value, gamma=gamma_value)
    #                 svm_classifier.train_test_clf( clf, "A1 (Orange/Blue) data", a1_data_tuple)
    #                 svm_draw_results( a1_data, clf.decision_function, 'Q1 ' + str(clf), SVM_A1_CLASS_FORMAT)
    #                 print("----- SUPPORT VECTORS:" + str(clf.n_support_))

    clf = svm.SVC(kernel='linear', C=1)
    svm_classifier.train_test_clf( clf, "A1 (Orange/Blue) data", a1_data_tuple)
    svm_blob = svm_draw_results( a1_data, clf.decision_function, 'Q1 ' + str(clf), SVM_A1_CLASS_FORMAT)

    (recognition_rate, confusions) = svm_conf_matrix(A1_CLASS, clf.predict(a1_data_tuple[0]), a1_data, title='SVM')
    confusions_list = confusions.tolist()

    plot_blob_base64 = base64.b64encode(svm_blob).decode('utf-8')

    return {"prediction_blob": plot_blob_base64, "recognition_rate": recognition_rate, "confusion_matrix": confusions_list}



    # timer.qcheck("Q1 program complete")
    # print(timer)




 


