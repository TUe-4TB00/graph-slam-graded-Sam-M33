import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X
import pickle

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer 

    # Creating LM parameters (`gtsam.LevenbergMarquardtParams`). We'll use the defaults.
    params = gtsam.LevenbergMarquardtParams()
    # Creating the optimizer instance, providing the graph, initial estimate, and parameters.
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # TODO: Perform the optimization and print the result

    # Running the optimization
    result = optimizer.optimize()
    # Print the optimized result
    print("\nFinal Result:\n{}".format(result))

    return result


def optimize_temp(graph, initial_estimate):
    # TODO: Initialize the optimizer 

    # Creating LM parameters (`gtsam.LevenbergMarquardtParams`). We'll use the defaults.
    params = gtsam.LevenbergMarquardtParams()
    # Creating the optimizer instance, providing the graph, initial estimate, and parameters.
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # TODO: Perform the optimization and print the result

    # Running the optimization
    result = optimizer.optimize()
    # Print the optimized result
    # print("\nFinal Result:\n{}".format(result))

    return result



def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    
    pickle.dump(graph, open('data_graph.pkl', 'wb'))
    pickle.dump(initial_estimate, open('data_init_est.pkl', 'wb'))


    chosen_pose = None
    chosen_landmark = None
    marginals_list = []
    sum_of_marginals = 0

    for best_landmark in range(1,3):
        for best_pose in pose_options:
            pose_5 = pose_options[best_pose]

            graph_temp = pickle.load(open('data_graph.pkl', 'rb')) 
            initial_estimate_temp = pickle.load(open('data_init_est.pkl', 'rb')) 

            graph_temp, initial_estimate_temp = add_pose(graph_temp, initial_estimate_temp, pose_5)
            result = optimize_temp(graph_temp, initial_estimate_temp)
            graph_temp = add_landmark_measurement(graph_temp, result, pose_5, best_landmark)
            result = optimize_temp(graph_temp, initial_estimate_temp)

            marginals_pose = gtsam.Marginals(graph_temp, result)
            
            sum_of_marginals = marginals_pose.marginalCovariance(L(best_landmark)).sum() 

            marginals_list.append(sum_of_marginals)
            
            if min(marginals_list) == sum_of_marginals:
                chosen_pose = best_pose
                chosen_landmark = best_landmark
            

    # # TODO: Calculate marginal covariances for the relevant variables and visualize the updated factor graph with covariances
    
    graph_temp = pickle.load(open('data_graph.pkl', 'rb')) 
    initial_estimate_temp = pickle.load(open('data_init_est.pkl', 'rb')) 

    graph_temp, initial_estimate_temp = add_pose(graph_temp, initial_estimate_temp, pose_options[chosen_pose])
    result = optimize(graph_temp, initial_estimate_temp)
    graph_temp = add_landmark_measurement(graph_temp, result, pose_5, chosen_landmark)
    result = optimize(graph_temp, initial_estimate_temp)
    marginals = gtsam.Marginals(graph_temp, result)
    sum_of_marginals = marginals.marginalCovariance(L(1)).sum() + marginals.marginalCovariance(L(2)).sum()

    return chosen_pose, chosen_landmark, sum_of_marginals

    
    

def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.


    pickle.dump(graph, open('data_graph2.pkl', 'wb'))
    pickle.dump(initial_estimate, open('data_init_est2.pkl', 'wb'))

    true_positions = {
        X(1): np.array([0.0, 0.0]),
        X(2): np.array([2.0, 0.0]),
        X(3): np.array([4.0, 0.0]),
    }

    chosen_pose = None
    chosen_landmark = None
    chosen_error = 0
    error_list = []

    for best_landmark in range(1,3):
        for best_pose in pose_options:
            pose_5 = pose_options[best_pose]

            graph_temp = pickle.load(open('data_graph2.pkl', 'rb')) 
            initial_estimate_temp = pickle.load(open('data_init_est2.pkl', 'rb')) 

            graph_temp, initial_estimate_temp = add_pose(graph_temp, initial_estimate_temp, pose_5)
            result = optimize_temp(graph_temp, initial_estimate_temp)
            graph_temp = add_landmark_measurement(graph_temp, result, pose_5, best_landmark)
            result = optimize_temp(graph_temp, initial_estimate_temp)

            list_of_errors = []
            for key, true_xy in true_positions.items():
                estimated_pose = result.atPose2(key)
                estimated_xy = np.array([estimated_pose.x(), estimated_pose.y()])
                error = np.linalg.norm( estimated_xy - true_xy)
                list_of_errors.append(error)

            sum_of_errors = sum(list_of_errors)

            error_list.append(sum_of_errors)
            
            if min(error_list) == sum_of_errors:
                chosen_pose = best_pose
                chosen_landmark = best_landmark
                chosen_error = sum_of_errors
                
    sum_of_errors = chosen_error
    best_pose = chosen_pose
    best_landmark = chosen_landmark


    return best_pose, best_landmark, sum_of_errors 