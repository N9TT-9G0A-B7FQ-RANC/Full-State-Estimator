import sys
sys.path.append('./')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from config import *
from tqdm import tqdm
import gc
from nn_architecture import Hybrid_resnet
from utils import to_numpy, data_preprocessing, open_data, open_json, create_folder_if_not_exists
from pykalman_ import UnscentedKalmanFilter
import argparse

def get_transition_and_observation_f(model):
    """
    Returns transition and observation functions for specified dynamic models.

    Args:
        model (str): Name of the dynamic model ('lorenz' or 'vanderpol').

    Returns:
        tuple: Tuple containing the transition and observation functions.

    The function returns a tuple of transition and observation functions based on the specified dynamic model.
    The available models are 'lorenz' and 'vanderpol'.
    
    For the Lorenz model:
    - Transition function simulates the Lorenz attractor dynamics.
    - Observation function extracts the x-component of the state.

    For the Vanderpol model:
    - Transition function simulates the dynamics of the Vanderpol oscillator.
    - Observation function extracts the x-component of the state.

    Each function takes the current state and noise as input and returns the next state or observation.
   """

    def vanderpol_transition_function(state, control, noise):
        mu = 2
        dt = 0.02
        x, y = state[0], state[1]
        xdt = mu * (x - 1/3 * x**3 - y)
        ydt = 1/mu * x
        return np.asarray([xdt , ydt]) * dt + state + noise

    def vanderpol_observation_function(state, control, noise):
        C = np.array([[1, 0], [0, 0]])
        return np.dot(C, state) + noise
        
    def duffing_transition_function(state, control, noise):
        gamma = 0.002
        beta = 1
        alpha = 1
        dt = 0.02
        x0, x1  = state[0], state[1]
        f = control[0]
        dx0__dt = x1
        dx1__dt = gamma * x1 - alpha * x0 - beta * x0**3 + f
        return np.asarray([dx0__dt , dx1__dt]) * dt + state + noise

    def duffing_observation_function(state, control, noise):
        C = np.array([[1, 0], [0, 0]])
        return np.dot(C, state) + noise

    if model == 'vanderpol':
        return vanderpol_transition_function, vanderpol_observation_function
    if model == 'duffing':
        return duffing_transition_function, duffing_observation_function
   
def plot_states_over_time(
        index,
        prediction,
        unscented_predictions,
        groundtruth,
        noisy_groundtruth, 
        control,
        states,
        nb_trajectories,
        training_parameters,
        eval_trajectory_duration,
        delta_t,
        max_lag,
        path,
        model
    ):

    for trajectory_idx in range(nb_trajectories):
        for state_idx, state in enumerate(states):

            # Specify the width and height ratio
            width = 10  # Width in inches
            ratio = 0.75  # Desired ratio (height/width)

            # Calculate the height based on the ratio
            height = width * ratio

            # Create a figure with the specified width and height
            fig = plt.figure(figsize=(width, height), dpi=100)

            for idx, training_idx in enumerate(index):

                idx = training_idx
   
                sequence_duration = float(training_parameters['past_temporal_horizon'].values[training_idx])
                nb_lag = int(sequence_duration/subsampling_dt) + 1
                state_prediction = prediction[idx][trajectory_idx, :, state_idx]
                unscented_state_prediction = unscented_predictions[idx][trajectory_idx, :, state_idx]
                state_groundtruth = groundtruth[idx][trajectory_idx, :, state_idx]
                state_noisy_groundtruth = noisy_groundtruth[idx][trajectory_idx, :, state_idx]
                time = np.arange(0, eval_trajectory_duration, delta_t)[max_lag+1:]

                if state in in_variables:
                    plt.plot(
                        time,
                        state_noisy_groundtruth,
                        'grey',
                        label=f"Noisy ground truth"
                    )

                plt.plot(
                    time,
                    state_groundtruth,
                    'red',
                    label=f"Ground truth"
                )

                plt.plot(
                    time,
                    unscented_state_prediction,
                    'green',
                    label=f"UKF Prediction"
                )

                plt.plot(
                    time,
                    state_prediction,
                    'black',
                    label=f"Prediction"
                )

                if model == 'vanderpol':
                    if state == 'x':
                        plt.ylim(-6.5, 6.5)
                    if state == 'y':
                        plt.ylim(-2., 2)
    
                if state in in_variables:
                    plt.title(f'$x_{state_idx+1}$ (Observed)')
                else:
                    plt.title(f'$x_{state_idx+1}$ (Unobserved)')
                if state_idx == 0:
                    plt.legend(loc='lower left', fontsize = 12)    
                plt.ylabel(f'$x_{state_idx+1}$', fontsize = 16)
                plt.xlabel('$t$', fontsize = 16)
                plt.gcf().set_size_inches(width, height)
                plt.savefig(f"{path}/state_{state}_{training_idx}_{trajectory_idx}.pdf",  dpi=200)
                plt.close()
    
def get_observation(model, X, U, input_size, output_size, delay, sequence_duration, dt):
    """
    Generates observations using a state observer model.

    Args:
        model (torch.nn.Module): The state observer model.
        X (torch.Tensor): Input data representing the state of the system.
        U (torch.Tensor): Input data representing the control inputs.
        input_size (int): Size of the input data.
        output_size (int): Size of the output data (observation).
        delay (float): Time delay in the system.
        sequence_duration (float): Duration of each observation sequence.
        dt (float): Time step.

    Returns:
        torch.Tensor: Tensor containing the generated observations.

    This function generates observations by applying a state observer model to input sequences (X, U).
    It uses a sliding window approach with a specified time delay and sequence duration.
    The resulting observations are stored in a tensor and returned.
    """
    sequence_size = int(sequence_duration / dt) + 1
    space_between_element = int(delay / dt)
    in_idx = torch.arange(0, sequence_size, space_between_element).long().to(device)
    results = torch.zeros((X.shape[0], X.shape[1] - sequence_size, output_size))
    for i in range(0, nb_step - sequence_size):
        indexed_X = torch.index_select(X, 1, i + in_idx).clone()
        indexed_U = torch.index_select(U, 1, i + in_idx).clone()
        pred = model.forward(indexed_X, indexed_U, None, batchsize = X.shape[0], mode = 'state_observer')
        results[:, i] = pred.detach().cpu()
    return results

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="")

    # Add command-line options
    parser.add_argument("--training_name", type=str, help="Training name")
    parser.add_argument("--training_parameters", type=str, help="Training parameters")
    parser.add_argument("--start_index", type=int, help="Start training index")
    parser.add_argument("--end_index", type=int, help="End training idx")

    # Parse the command-line arguments
    args = parser.parse_args()
    # Access the values of the options
    training_name = args.training_name
    start = args.start_index
    end = args.end_index

    training_folder = f"./results/training_{training_name}/"
    training_parameters = pd.read_csv(f"{training_folder}/config.csv")
    plot = True

    groundtruth = []
    noisy_groundtruth = []
    predictions = []
    unscented_predictions = []
    control = []

    nb_training = len(training_parameters)
    
    for training_idx in tqdm(range(start, end, 1)):

        training_config = open_json(f"{training_folder}/training_config_{training_idx}.json")
            
        subsampling_dt = float(training_parameters['subsampling_dt'].values[training_idx])
        nb_hidden_layer = int(training_parameters['nb_layers'].values[training_idx])
        nb_neurones_per_layer = int(training_parameters['nb_neurones_per_layers'].values[training_idx])
        activation = str(training_parameters['activation'].values[training_idx])
        batchsize = int(training_parameters['batchsize'].values[training_idx])
        sequence_duration = float(training_parameters['past_temporal_horizon'].values[training_idx])
        delay = float(training_parameters['past_delay'].values[training_idx])
        futur_delay = float(training_parameters['future_temporal_horizon'].values[training_idx])
        state_configuration = str(training_parameters['state_configuration'].values[training_idx])
        smoothing = bool(training_parameters['smoothing'].values[training_idx])
        training_data = str(training_parameters['data'].values[training_idx])
        state_configuration = str(training_parameters['state_configuration'].values[training_idx])
        max_sequence_duration = int(np.max(training_parameters['past_temporal_horizon']))
        data_dt = float(training_parameters['data_dt'].values[training_idx])

        transition_function, observation_function = get_transition_and_observation_f(state_configuration)

        data_path = f'./simulation/{training_data}'

        if training_idx == 0:
            data_list = open_data(data_path, 'simulation', nb_trajectories)

        sequence_size = int(sequence_duration/delay) + 1
        nb_evaluation_step = int(eval_trajectory_duration / futur_delay)
        nb_residual_blocks = int(futur_delay / subsampling_dt)

        train_trajectories_idx = training_config['training_idx']
        val_trajectories_idx = training_config['validation_idx']
        test_trajectories_idx = training_config['test_idx']
        state_variables = training_config['states_variables']
        in_variables = training_config['in_states_variables']
        out_variables = training_config['out_states_variables']
        control_variables = training_config['control_variables']

        observed_variables_idx = []
        non_observed_variables_idx = []
        for idx in range(len(state_variables)):
            for variable in in_variables:
                if variable == state_variables[idx] or f'{variable}_' == state_variables[idx]:
                    observed_variables_idx.append(idx)
                    break
        for idx in range(len(state_variables)):
                if idx not in observed_variables_idx:
                    non_observed_variables_idx.append(idx)

        train_data_list, val_data_list, test_data_list = [], [], []
        for idx in train_trajectories_idx:
            train_data_list.append(data_list[idx].copy())
        for idx in val_trajectories_idx:
            val_data_list.append(data_list[idx].copy())
        for idx in test_trajectories_idx:
            test_data_list.append(data_list[idx].copy())

        nb_in_state = len(in_variables)
        nb_out_state = len(out_variables)
        nb_state = len(state_variables)
        nb_control = len(control_variables)

        X_noisy_in_list, U_noisy_in_list, X_noisy_out_list = data_preprocessing(
                data_list = test_data_list.copy(),
                data_dt = data_dt,
                subsampling_dt = subsampling_dt,
                state_variables = state_variables,
                out_variables = state_variables,
                control_variables = control_variables,
                differentiate = False,
                smoothing = smoothing,
            )

        X_in_list, U_in_list, X_out_list = data_preprocessing(
                data_list = test_data_list.copy(),
                data_dt = data_dt,
                subsampling_dt = subsampling_dt,
                state_variables = [f'{state}_' for state in state_variables],
                out_variables = [f'{state}_' for state in state_variables],
                control_variables = control_variables,
                differentiate = False,
                smoothing = smoothing,
            )

        nb_evaluation_step = int(eval_trajectory_duration / subsampling_dt)
        input_size = len(state_variables) + len(control_variables)
        output_size = len(state_variables)
        
        model = Hybrid_resnet(
                nb_integration_steps = nb_residual_blocks,
                control_variables = control_variables,
                observed_states = in_variables,
                nb_hidden_layer = nb_hidden_layer,
                nb_neurones_per_hidden_layer = nb_neurones_per_layer,
                activation = activation,
                delay = delay,
                sequence_duration = sequence_duration,
                dt = subsampling_dt,
                prior_model = state_configuration,
            )

        model.load_state_dict(torch.load(f'{training_folder}/best_model_{training_idx}.pt'))
        model = model.to(device)

        model.set_z_scale(False)
        model.to(device)
        model.eval()

        nb_test_trajectories = len(X_noisy_in_list)
        max_lag = int(max_sequence_duration / subsampling_dt) + 1
        nb_lag = int(sequence_duration / subsampling_dt) + 1
        nb_step = int(eval_trajectory_duration / subsampling_dt) - max_lag + nb_lag - 1
        
        X_in_tensor = torch.zeros((nb_test_trajectories, nb_step, nb_state)).to(device)
        U_in_tensor = torch.zeros((nb_test_trajectories, nb_step, nb_control)).to(device)
        X_out_tensor = torch.zeros((nb_test_trajectories, nb_step, nb_state)).to(device)
        
        X_noisy_in_tensor = torch.zeros((nb_test_trajectories, nb_step, nb_state)).to(device)
        U_noisy_in_tensor = torch.zeros((nb_test_trajectories, nb_step, nb_control)).to(device)
        X_noisy_out_tensor = torch.zeros((nb_test_trajectories, nb_step, nb_state)).to(device)

        for idx, (X_noisy_in, U_noisy_in, X_in, U_in) in enumerate(zip(X_noisy_in_list, U_noisy_in_list, X_in_list, U_in_list)):
            X_in_tensor[idx] = torch.tensor(X_in[max_lag - nb_lag:])
            U_in_tensor[idx] = torch.tensor(U_in[max_lag - nb_lag:])
            X_noisy_in_tensor[idx] = torch.tensor(X_noisy_in[max_lag - nb_lag:])
            U_noisy_in_tensor[idx] = torch.tensor(U_noisy_in[max_lag - nb_lag:])

        X_pred = get_observation(
            model, 
            X_noisy_in_tensor[:, :, observed_variables_idx], 
            U_noisy_in_tensor, 
            input_size, 
            output_size, 
            delay, 
            sequence_duration, 
            subsampling_dt
        )

        if training_idx == start:
            X_unscented_pred = np.zeros((nb_test_trajectories, int(eval_trajectory_duration / subsampling_dt) - 1, nb_state))
            for idx, (X_noisy_in, U_noisy_in) in enumerate(zip(X_noisy_in_list, U_noisy_in_list)): 
            
                # Initialize UKF
                if state_configuration == 'vanderpol':
                    if 'vanderpol_a' in training_name:
                        transition_covariance = np.diag([0.0001] * 2)
                        observation_covariance = np.diag([0.05] * 2)
                        initial_state_covariance = np.diag([0.05] * 2)
                    if 'vanderpol_b' in training_name:
                        transition_covariance = np.diag([0.001] * 2)
                        observation_covariance = np.diag([1] * 2)
                        initial_state_covariance = np.diag([1] * 2)
                    if 'vanderpol_c' in training_name:
                        transition_covariance = np.diag([0.001] * 2)
                        observation_covariance = np.diag([1.5] * 2)
                        initial_state_covariance = np.diag([1.5] * 2)

                    random_state = np.random.RandomState(1)
                    initial_state_mean = [
                        X_noisy_in[0, 0], 
                        (np.random.rand() - 0.5) * 10,
                    ]

                if state_configuration == 'duffing':
                    if 'duffing_a' in training_name:
                        transition_covariance = np.diag([0.0001] * 2)
                        observation_covariance = np.diag([1] * 2)
                        initial_state_covariance = np.diag([1] * 2)
                    if 'duffing_b' in training_name:
                        transition_covariance = np.diag([0.0001] * 2)
                        observation_covariance = np.diag([1] * 2)
                        initial_state_covariance = np.diag([1] * 2)
                    if 'duffing_c' in training_name:
                        transition_covariance = np.diag([0.0001] * 2)
                        observation_covariance = np.diag([1.5] * 2)
                        initial_state_covariance = np.diag([1.5] * 2)

                    random_state = np.random.RandomState(1)
                    initial_state_mean = [
                        X_noisy_in[0, 0], 
                        (np.random.rand() - 0.5) * 1,
                    ]

                # Sample from model
                kf = UnscentedKalmanFilter(
                    transition_function, observation_function,
                    transition_covariance, observation_covariance,
                    initial_state_mean, initial_state_covariance,
                    random_state = random_state
                )
                X_unscented_pred[idx] = kf.filter(X_noisy_in, U_noisy_in)[0]

        unscented_predictions.append(X_unscented_pred[:, max_lag:])
        groundtruth.append(to_numpy(X_in_tensor[:, nb_lag:]))
        noisy_groundtruth.append(to_numpy(X_noisy_in_tensor[:, nb_lag:]))
        predictions.append(to_numpy(X_pred))
        control.append(to_numpy(U_noisy_in_tensor[:, nb_lag:]))

        del X_in, U_in, model
        gc.collect()
        torch.cuda.empty_cache()

    rmse = {}
    for idx in range(len(state_variables)):
        rmse[state_variables[idx]] = []
        rmse[f'ukf_{[state_variables[idx]]}'] = []
    rmse['rmse'] = []
    rmse['ukf_rmse'] = []

    # Compute RMSE
    for training_idx, observation_pred, observation_truth in zip(range(end - start), predictions, groundtruth):
        state_error = []
        for trajectory_idx in range(nb_test_trajectories):
            state_error.append(np.sqrt(np.mean((observation_pred[trajectory_idx] - observation_truth[trajectory_idx])**2, axis=0)))
        state_error = np.asarray(state_error)
        state_error = np.mean(state_error, axis=0)
        for idx in range(len(state_variables)):
            rmse[state_variables[idx]].append(state_error[idx])
        rmse['rmse'].append(np.mean(state_error))
    
    # Compute RMSE for UKF
    for training_idx, observation_pred, observation_truth in zip(range(end - start), unscented_predictions, groundtruth):
        state_error = (observation_pred - observation_truth)**2
        state_error = state_error.reshape(state_error.shape[0] * state_error.shape[1], state_error.shape[2])
        state_error = np.sqrt(np.mean(state_error, axis=0))
        for idx in range(len(state_variables)):
            rmse[f'ukf_{[state_variables[idx]]}'].append(state_error[idx])
        rmse['ukf_rmse'].append(np.mean(state_error))

    results = pd.DataFrame(rmse)
    results['past_temporal_horizon'] = training_parameters['past_temporal_horizon']
    results['future_temporal_horizon'] = training_parameters['future_temporal_horizon']
    results = results.sort_values(by=['rmse'])
    results.to_csv(f'{training_folder}/state_error.csv')

    sorted_idx = np.argsort(rmse['rmse'])
    
    if plot:

        create_folder_if_not_exists(f"{training_folder}/plot/")

        plot_states_over_time(
            list(results.index[:3]),
            predictions,
            unscented_predictions,
            groundtruth,
            noisy_groundtruth,
            control,
            state_variables,
            nb_test_trajectories,
            training_parameters,
            eval_trajectory_duration,
            subsampling_dt,
            max_lag,
            path = f"{training_folder}/plot/",
            model = state_configuration)