
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    # TODO: Add the odometry factor between X(4) and X(5) to the graph (BetweenFactorPose2) 
                # ????? SHouldn't it be between X(3) and X(4) ????
    
    # Between X(3) and X(4): Turn 45 degrees and move forward 2m (so move 2m diagonally), then turn another 45 degrees
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), gtsam.Pose2(2.0/math.sqrt(2), 2.0/math.sqrt(2), math.radians(90)), ODOMETRY_NOISE))

    # TODO: Based on the odometry, find the initial estimate for the pose of X(5) and add it to the graph
                # ????? Again shouldn't it be between X(4) ????

    # Position: move 2m [x] + move 2m [x] + (turn 45 degrees and move 2m [xy]), turn 45 degrees
    initial_estimate.insert(X(4), gtsam.Pose2(4.0+2.0/math.sqrt(2), 0.0+2.0/math.sqrt(2), 0.0+math.radians(90)))

    
    return graph, initial_estimate