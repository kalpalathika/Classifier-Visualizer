################################################################
# Machine Learning
#
# A2 Visualization and Outputs
#
# Author: R. Zanibbi
# Author: E. Lima
################################################################

import numpy as np
import matplotlib.pyplot as plt
import io
import matplotlib as mpl
# mpl.use('macosx')
# you might want to try mpl.use('Qt5Agg') or mpl.use('TkAgg')


from SVM import debug
from data_sets_a3 import *

def show_for_classes(class_dict, message, prop_list, pause=False):
    print( '\n' + message )
    for i in range(len(prop_list)):
        print( '  ', class_dict[i], ':', prop_list[i] )
    
    if pause:
        input("Press any key to continue...\n")


def pshow_for_classes(message, prop_list):
    show_for_classes(message, prop_list, True)



def conf_matrix(class_dict, class_matrix, data_matrix, title=None, print_results=True, cost_matrix=np.array([])):
    # Updated from A1 to accomodate > 2 classes, and to show losses if cost matrix is provided
    class_count = len(class_dict.keys())
    class_labels = list(class_dict.values())

    # Initialize confusion matrix with 0's
    confusions = np.zeros((class_count, class_count), dtype=int)

    # Generate output/target pairs, convert to list of integer pairs
    # * '-1' indexing indicates the final column
    # * list(zip( ... )) combines the output and target class label columns into a list of pairs
    out_target_pairs = [
        (int(out), int(target))
        for (out, target) in list(zip(class_matrix, data_matrix[:, -1]))
    ]

    # Use output/target pairs to compile confusion matrix counts
    for (out, target) in out_target_pairs:
        confusions[out][target] += 1

    # Compute recognition rate
    inputs_correct = sum( [ confusions[i][i] for i in range(class_count) ] )
    inputs_total = np.sum(confusions)
    recognition_rate = inputs_correct / inputs_total 

    if print_results:
        width=10
        if title:
            print("\n>>>  " + title)
        print(
                "\n    Recognition rate (correct / inputs):\n      {:4.2%}".format(recognition_rate) + "\n"
        )

        # ADDITION: Loss measure
        if cost_matrix.size > 0:
            # Total loss/profit is sum of element-wise product of confusion and cost matrix
            loss = np.sum( confusions * cost_matrix )
            print("    Total Loss for Test Samples:\n      {:.2f}".format(loss) + "\n")

        print("    Confusion Matrix:\n")
        
        # HACK
        print( ("Out/GT" + (" " * (width-6)) + ("{:>" + str(width) + "}")*(class_count)).format( *class_labels ) )
        print( "-" * ( width * (class_count+1) ) )
        
        for i in range(class_count):
            # HACK
            row_label = ('{:1d}: {:'+ str(width-3) + '}').format(i, class_dict[i]) 
            row_entries =  (("{:" + str(width) + "d}")*class_count).format( *(confusions[i]) )
            print( row_label + row_entries )
        

    return (recognition_rate, confusions)


def classify_grid( data_matrix, title, class_fn, axis_tick_count=100, last_col_class=True, binary_threshold=0 ):
    # Applies classification function over a grid of data

    # Fix axes ranges so that X and Y directions are identical (avoids 'stretching' in one direction or the other)
    # Use numpy amin function on first two columns of the training data matrix to identify range
    pad = 0.25
    min_tick = np.amin(data_matrix[:, 0:2]) - pad
    max_tick = np.amax(data_matrix[:, 0:2]) + pad

    # Default axis_tick_count: 100 points in each direction, giving 10,000 samples
    x = np.linspace(min_tick, max_tick, axis_tick_count, endpoint=True)
    y = np.linspace(min_tick, max_tick, axis_tick_count, endpoint=True)
    (xx, yy) = np.meshgrid(x, y)
    grid_points = np.concatenate(
        (xx.reshape(xx.size, 1), yy.reshape(yy.size, 1)), axis=1
    )

    # Classify points, organize results for visualization
    # DEBUG: Z needs to have grid layout per meshgrid arrays
    class_out = class_fn(grid_points)

    # A3: Reshape differently if classes are attached in the final column.
    # Keep A2 behavior as default
    Z = None
    if not last_col_class:
        if class_out.ndim > 1:
            # A3 new case: ASSUMES max score selected for class 
            Z = np.argmax( class_out, axis=1 ).reshape(xx.shape)
        else:
            # Binary classification -- single score
            #Z = class_out.reshape(xx.shape)
            Z = ((class_out > binary_threshold ).astype(int)).reshape(xx.shape)
    else:
        Z = ( class_out[:,-1] ).reshape(xx.shape)

    return (xx, yy, Z, min_tick, max_tick, class_out)


# Increased tick count to 1000, remove file name argument
def draw_results(data_matrix, class_fn, title, class_formats, axis_tick_count=500 ):
    print("*** Drawing decision boundary")
    timer = debug.DebugTimer("Drawing Decision Boundary")

    # Generate file_name from title
    file_name = title.replace(' ','_') + ".pdf"
    # Addition to handle sklearn output that does not contain class selection
    last_col_class = False
    if data_matrix.ndim > len(class_formats):
        last_col_class = True

    # Draw 2D sample, class region, and decision boundary plots

    # Run classifier over grid around test samples
    (xx, yy, Z, min_tick, max_tick, _) = classify_grid(data_matrix, title, class_fn, axis_tick_count, last_col_class )
    timer.qcheck("Grid classification (ticks: " + str(axis_tick_count) + ")")
    plt.switch_backend('Agg')

    # Plot filled contour with transparent color
    plt.xlim(min_tick, max_tick)
    plt.ylim(min_tick, max_tick)
    plt.contourf(xx, yy, Z, levels=len(class_formats)-1, 
                 colors=[ class_formats[key][1] for key in class_formats ], alpha=0.2)
    plt.contour(xx, yy, Z,  levels=len(class_formats)-1, colors='black', alpha=1.0, linewidths=1.0, )

    # Draw training samples
    ( split_data, X, y ) = split_rows_on_class_labels( data_matrix )
    for i in range( len(split_data) ):
        p = split_data[i]
        plt.scatter(
            p[:,0],
            p[:,1],
            marker=class_formats[i][0],
            facecolors=class_formats[i][1],
            edgecolors='black'
        )

    # Add title and write file (extension determines type)
    # plt.title(title)
    # plt.savefig(file_name)
    # plt.close()
    # timer.qcheck("Plot created & saved to: " + file_name)
    plt.title(title)
    plot_stream = io.BytesIO()
    plt.savefig(plot_stream, format='png')
    plt.close()

    plot_stream.seek(0)

    return plot_stream.getvalue()

    # print(timer)


def draw_contours(data_matrix, class_fn, title, class_formats, 
                  axis_tick_count=500, classes=[], show=False, file_name=None ):
    # Generate file_name from title
    if not file_name:
        file_name = title.replace(' ','_') + ".pdf"

    # Addition to handle sklearn output that does not contain class selection
    last_col_class = False
    if data_matrix.ndim > len(class_formats):
        last_col_class = True
    # Draw class scores as 3D contours over samples in 2D plane
    plt.switch_backend('Agg')
 
    # Run classifier over grid around test samples
    (xx, yy, Z, min_tick, max_tick, class_out) = classify_grid(data_matrix, title, class_fn, axis_tick_count, last_col_class)

    # Draw class score contours
    f, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # DEBUG: Addition for binary case with single score (sklearn will not generate 1 vs. all entries for binary)
    if class_out.ndim < 2:
        # ASSUMPTION: treat output score as score for class 1
        surf = ax.plot_surface(xx, yy, class_out.reshape(xx.shape), color=class_formats[1][1], alpha=0.4, linewidth=2)
    elif len(classes) < 1:
        # Plot surface for all classes, one at a time
        for i in range(len(class_formats)):
            surf = ax.plot_surface(xx, yy, (class_out[:,i]).reshape(xx.shape), color=class_formats[i][1], alpha=0.4, linewidth=2)
    else:
        # Specific classes selected
        for i in classes:
            surf = ax.plot_surface(xx, yy, (class_out[:,i]).reshape(xx.shape), color=class_formats[i][1], alpha=0.4, linewidth=2)
    
    # Draw training samples
    ( split_data, X, y ) = split_rows_on_class_labels( data_matrix )
    for i in range( len(split_data) ):
        p = split_data[i]
        ax.scatter(
            p[:,0],
            p[:,1],
            np.zeros( len(p) ),
            marker=class_formats[i][0],
            facecolors=class_formats[i][1],
            edgecolors='black'
        )

    # Add title and write file/interact with plot if title is given as empty string
    plt.title(title)
    plt.savefig(file_name)
    if show:
        plt.show()
    plt.close()

