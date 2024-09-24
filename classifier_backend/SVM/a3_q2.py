################################################################
# a3_q2.py
#
# (Random Forest) A3 Question 2
################################################################

from debug import *
timer = DebugTimer("Initialization")

from data_sets_a3 import *
from classifiers import *
from results_visualization import draw_results, draw_contours

RANDOM_SEED=0

def show_score( data, data_tuple, clf, show_setting=False ):
    # NOTE: Setting show=True will make the plot interactive; initial program only writes files to disk
    # * Left click to rotate the 3D plot
    # * Right click and move mouse up/down to zoom
    for i in BNRS_CLASS:
        draw_contours(data, clf.predict_proba, "Metal Parts Scores for " +
                      BNRS_CLASS[i] + ":\n" + str(list(clf.best_params_.values())),
                      BNRS_CLASS_FORMAT, show=show_setting, classes=[i],
                      file_name = BNRS_CLASS[i] + "_" + str(clf.best_params_.values()).replace(' ','_')+".pdf")

def main():
    timer.qcheck("Imports complete")

    ( a2_data_tuple, a2_data ) = load_a2_data(timer)
    ( X_train, X_test, y_train, y_text ) = a2_data_tuple


    # Grid search using sklearn built-in grid_search function
    nTrees = [ 2, 4, 8, 16, 32 ]
    maxDepth = [ 1, 2, 4, 8, 16 ]
    nFeatures = [ 1 ]
    
    parameter_values = [
            {"n_estimators": [1], "max_depth": [None], "max_features": [None] },
            {"n_estimators": nTrees, "max_depth": maxDepth, "max_features": nFeatures }
    ]


    # See classifiers.py for details on this grid_search function (wraps sklearn version)
    clf = RandomForestClassifier( random_state=RANDOM_SEED )
    grid_out = grid_search(clf, "Metal Parts", parameter_values, a2_data_tuple) 


    # Single decision tree
    sdt = RandomForestClassifier( random_state=RANDOM_SEED, n_estimators=1, max_depth=None, max_features=None )
    sdt.fit( X_train, y_train )

    # Plots
    draw_results( a2_data, grid_out.predict_proba, 'Q2 RF' + str([maxDepth,nFeatures,nTrees]), BNRS_CLASS_FORMAT)
    draw_results( a2_data, sdt.predict_proba, 'Q2 DTree ' + str([RANDOM_SEED, 1, None, None]), BNRS_CLASS_FORMAT)

    # Change 'showsetting' to True to interact with the score plots.
    show_score(a2_data, a2_data_tuple, grid_out, show_setting=False)
    draw_contours( a2_data, sdt.predict_proba, "Metal Parts Scores for Single DTree:\n",
                      BNRS_CLASS_FORMAT, file_name="DTree_Metal.pdf", show=True)


    timer.qcheck("Q2 program complete")
    print(timer)

if __name__ == "__main__":
    main()

