################################################################
# classifiers.py
#
# A3: scikit-learn SVM, Random Forests and grid searches
#     for hyper-parameter tuning
#
# Author: R. Zanibbi
# Author: E. Lima
################################################################

# Performance parameter
# Change to 'None' if you experience problems (will run slower)
nJobs=-1



# Standard python libraries
import pandas as pd
import numpy as np
from operator import itemgetter

# Sklearn libraries
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from SVM import debug
# from debug import DebugTimer, ncheck, dncheck, dcheck, dnpcheck

##################################
# Show classification test metrics
##################################

def report_metrics( clf, test_data_name, y_test, predicted ):
    
    print(
        f"\nMetrics for {test_data_name}\nRunning {clf}:\n"
        f"{metrics.classification_report(y_test, predicted, digits=3)}\n"
    )



##################################
# Train/test single classifier
##################################

def train_test_clf( classifier, data_name, data_tuple, pause=False ):
    ( X_train, X_test, y_train, y_test ) = data_tuple
    nTrain = X_train.shape[0]
    nTest = X_test.shape[0] 

    print("\n>>> Running ",classifier,"on",data_name)

    timer_clf = debug.DebugTimer( str(classifier) + " on " + data_name )
    
    classifier.fit( X_train, y_train)
    timer_clf.qcheck("Training time (" + str(nTrain) + " samples)")

    y_predicted = classifier.predict( X_test )
    timer_clf.qcheck("Test time (" + str(nTest) + " samples)")

    report_metrics( classifier, data_name, y_test, y_predicted )
    timer_clf.qcheck("Metrics")

    print( timer_clf )
    
    if pause:
        input("Press any key to continue...")


##################################
# Grid Search
##################################

# Show entries
def print_dataframe(cv_results):
    tuples = sorted(list( zip( cv_results["rank_test_accuracy"], cv_results["mean_test_accuracy"], cv_results["std_test_accuracy"],
                 cv_results["mean_score_time"], cv_results["params"] )), key=itemgetter(0))

    for ( rank_position, mean_accuracy, std_accuracy, mean_score_time, params ) in tuples:
        print(
                f"{rank_position:3d}, "
                f"accuracy: {mean_accuracy:0.3f} (±{std_accuracy:0.05f}), "
                f"avg score time: {mean_score_time:0.3f}, "
                f"for {params}"
        )
    print("")

# Used to select parameters for final training on full training dataset
# cv_results: results from sklearn cross validation function
def refit_accuracy( cv_results ):
    # Create pandas data frame
    pd_cv_results = pd.DataFrame(cv_results)
    settings = len(pd_cv_results.index)
    
    print("All grid-search results: ", settings, "models")
    print_dataframe(pd_cv_results)
    
    # Select maximum mean accuracy, then the (first) with smallest standard deviation
    max_mean_accuracy = pd_cv_results[ pd_cv_results['rank_test_accuracy'] == 1 ]
    min_std_max_accuracy = max_mean_accuracy[ "std_test_accuracy" ].idxmin()

    return min_std_max_accuracy

def grid_search( classifier, data_name, parameter_values, data_tuple,
                    scores=["accuracy"], refit_fn=refit_accuracy, jobs=nJobs, pause=False):
    (X_train, X_test, y_train, y_test) = data_tuple
    nTrain = X_train.shape[0]
    nTest = X_test.shape[0] 

    print("\n>>> Running Grid Search for ", classifier, "on", data_name)

    grid_search = GridSearchCV( classifier, parameter_values, scoring=scores, refit=refit_fn, n_jobs=jobs )
    timer_grid = debug.DebugTimer( str(classifier) + " on " + data_name )

    grid_search.fit( X_train, y_train )
    timer_grid.qcheck("Grid search w. 5-fold cross-validation + refit on best params")

    #dncheck('Grid search output fields', grid_search.cv_results_.keys())

    y_pred = grid_search.predict(X_test)
    timer_grid.qcheck("Best model test time (" + str(nTest) + " samples)")

    print('')
    debug.ncheck("Selected parameters ('model')", grid_search.best_params_)
    report_metrics( grid_search, data_name, y_test, y_pred )
    timer_grid.qcheck("Metrics")

    print(timer_grid)

    if pause:
        input("Press any key to continue...")

    # Return grid_search object (holds retrained model)
    return grid_search




