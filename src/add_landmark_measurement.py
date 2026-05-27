import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_landmark_measurement(graph, initial_estimate, result):
    # Determine the correct rotation (bearing) and distance from X(4) to L(2) 
    rotation = math.degrees(math.atan2( 2**0.5, (2-2**0.5)))
    distance = math.sqrt( 2 + (2-2**0.5)**2)
    graph.add(gtsam.BearingRangeFactor2D(X(4), L(2), gtsam.Rot2.fromDegrees(rotation), distance, MEASUREMENT_NOISE))


    # Creating LM parameters (`gtsam.LevenbergMarquardtParams`). We'll use the defaults.
    params = gtsam.LevenbergMarquardtParams()
    # Creating the optimizer instance, providing the graph, initial estimate, and parameters.
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    # Running the optimization
    result = optimizer.optimize()
    # Print the optimized result
    print("\nFinal Result:\n{}".format(result))

    # Calculating marginal covariances for all variables.
    marginals = gtsam.Marginals(graph, result)

    # Print the covariance matrix for each variable
    print("X1 covariance:\n{}\n".format(marginals.marginalCovariance(X(1))))
    print("X2 covariance:\n{}\n".format(marginals.marginalCovariance(X(2))))
    print("X3 covariance:\n{}\n".format(marginals.marginalCovariance(X(3))))
    print("X4 covariance:\n{}\n".format(marginals.marginalCovariance(X(4))))
    print("L1 covariance:\n{}\n".format(marginals.marginalCovariance(L(1))))
    print("L2 covariance:\n{}\n".format(marginals.marginalCovariance(L(2))))


    return graph