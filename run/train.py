import sys
sys.path.append('./')

import numpy as np
import pandas as pd
import torch
import os
from utils import to_numpy, SimpleDataLoader, save_to_json, data_preprocessing, open_data, split_data
from config import *
from tqdm import tqdm
import gc
from nn_architecture import Hybrid_resnet
import argparse
    
def train_loop(
        model,
        optimizer, 
        dataLoader,
        standardize,
        mean,
        std,
    ):

    mse_loss = []

    model.train()

    for X_in, U_in, X_out, U_out in dataLoader:

        X_out_pred, reconst, X_estimated = model(
            X_in,
            U_in,
            U_out,
            batchsize = X_in.shape[0]
        )

        if standardize :
            X_out = (X_out - mean) / std
       
        se = (X_out_pred - X_out)**2
        mse = torch.mean(se)
        loss = mse 

        # Compute training loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mse_loss.append(to_numpy(se))

    mse_loss = np.asarray(mse_loss)
    mse_loss = mse_loss.reshape(mse_loss.shape[0] * mse_loss.shape[1] * mse_loss.shape[2], mse_loss.shape[3])
    rmse_loss = np.mean(np.sqrt(mse_loss), axis = 0)

    return rmse_loss

def validation_loop(
        model, 
        valDataLoaderOneStep,
        standardize,
        mean,
        std,
    ):
    
    model.eval()
    loss = []

    # One step evaluation
    for X_in, U_in, X_out, U_out in valDataLoaderOneStep:

        X_out_pred, X_observed, X_estimated = model(
            X_in,
            U_in,
            U_out,
            batchsize = X_in.shape[0]
        )

        if standardize :
            X_out = (X_out - mean) / std
 
        se = (X_out_pred - X_out)**2
        loss.append(to_numpy(se))

    loss = np.asarray(loss)
    loss = loss.reshape(loss.shape[0] * loss.shape[1] * loss.shape[2], loss.shape[3])
    loss = np.mean(np.sqrt(loss), axis = 0)

    return loss

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
    training_parameters = args.training_parameters
    start_idx = args.start_index
    end_idx = args.end_index

    # training_name = 'vanderpol_b'
    # training_parameters = 'vanderpol_b'
    # start_idx = 0
    # end_idx = 16

    training_folder = f"./results/training_{training_name}/"
    parameters_path = f"./training_parameters/{training_parameters}.csv"

    if not os.path.exists(training_folder):
        os.makedirs(training_folder)
    parameters = pd.read_csv(parameters_path)
    parameters.to_csv(f'{training_folder}/config.csv')
    
    assert end_idx <= len(parameters)

    for training_idx in range(start_idx, end_idx):
        
        # Open training parameters
        subsampling_dt = float(parameters['subsampling_dt'].values[training_idx])
        nb_hidden_layer = int(parameters['nb_layers'].values[training_idx])
        nb_neurones_per_layer = int(parameters['nb_neurones_per_layers'].values[training_idx])
        activation = str(parameters['activation'].values[training_idx])
        batchsize = int(parameters['batchsize'].values[training_idx])
        past_sequence_duration = float(parameters['past_temporal_horizon'].values[training_idx])
        future_sequence_duration = float(parameters['future_temporal_horizon'].values[training_idx])
        past_delay = float(parameters['past_delay'].values[training_idx])
        future_delay = float(parameters['future_delay'].values[training_idx])
        state_configuration = str(parameters['state_configuration'].values[training_idx])
        smoothing = bool(parameters['smoothing'].values[training_idx])

        training_data = str(parameters['data'].values[training_idx])

        shuffle = True
        standardize = True
        learning_rate = 1e-3
  
        data_dt = float(parameters['data_dt'].values[training_idx])

        np.random.seed(seed)
        data_path = f'./simulation/{training_data}'
      
        data_list = open_data(data_path, 'simulation', nb_trajectories)

        (train_data_list, train_trajectories_idx, 
         val_data_list, val_trajectories_idx, 
         test_data_list, test_trajectories_idx) = split_data(data_list, nb_trajectories, shuffle, train_set_pct, val_set_pct)

        sequence_size = int(past_sequence_duration/past_delay) + 1
        nb_evaluation_step = int(eval_trajectory_duration / subsampling_dt)
        nb_residual_blocks = int(future_sequence_duration / future_delay)

        in_variables = system_configuration[state_configuration]['observed_state']
        out_variables = system_configuration[state_configuration]['observed_state']
        control_variables = system_configuration[state_configuration]['control']
        state_variables = system_configuration[state_configuration]['state']
 
        model = Hybrid_resnet(
                nb_integration_steps = nb_residual_blocks,
                control_variables = control_variables,
                observed_states = in_variables,
                nb_hidden_layer = nb_hidden_layer,
                nb_neurones_per_hidden_layer = nb_neurones_per_layer,
                activation = activation,
                delay = past_delay,
                sequence_duration = past_sequence_duration,
                dt = subsampling_dt,
                prior_model = state_configuration,
            )
             
        train_in_state, train_in_control, train_out_state = data_preprocessing(
            data_list = train_data_list.copy(),
            data_dt = data_dt,
            subsampling_dt = subsampling_dt,
            state_variables = in_variables,
            out_variables = out_variables,
            control_variables = control_variables,
            differentiate = False,
            smoothing = smoothing,
            smoothing_parameters = smoothing_parameters
        )

        val_in_state, val_in_control, val_out_state = data_preprocessing(
            data_list = val_data_list.copy(),
            data_dt = data_dt,
            subsampling_dt = subsampling_dt,
            state_variables = in_variables,
            out_variables = out_variables,
            control_variables = control_variables,
            differentiate = False,
            smoothing = smoothing,
            smoothing_parameters = smoothing_parameters
        )
        
        # Instantiate dataloader
        trainDataLoader = SimpleDataLoader(
            train_in_states = train_in_state.copy(),
            train_in_controls = train_in_control.copy(),
            train_out_states = train_out_state.copy(),
            batchsize = batchsize,
            past_sequence_duration = past_sequence_duration,
            future_sequence_duration = future_sequence_duration,
            past_delay = past_delay,
            future_delay = future_delay,
            dt = subsampling_dt,
            shuffle = shuffle,
            device = device
        )

        valDataLoader = SimpleDataLoader(
            train_in_states = val_in_state.copy(),
            train_in_controls = val_in_control.copy(),
            train_out_states = val_out_state.copy(),  
            batchsize = batchsize,
            past_sequence_duration = past_sequence_duration,
            future_sequence_duration = future_sequence_duration,
            past_delay = past_delay,
            future_delay = future_delay,
            dt = subsampling_dt,
            shuffle = shuffle,
            device = device
        )

        std = np.asarray(train_out_state).reshape(np.asarray(train_out_state).shape[0] * np.asarray(train_out_state).shape[1], len(out_variables)).std(axis=0)
        mean = np.asarray(train_out_state).reshape(np.asarray(train_out_state).shape[0] * np.asarray(train_out_state).shape[1], len(out_variables)).mean(axis=0)
        model.set_mean(mean.copy())
        model.set_std(std.copy())

        std = torch.tensor([std]).float().requires_grad_(False).to(device)
        mean = torch.tensor([mean]).float().requires_grad_(False).to(device)

        training_config = {
            'in_states_variables' : list(in_variables),
            'out_states_variables': list(out_variables),
            'states_variables': list(state_variables),
            'control_variables':list(control_variables),
            'training_idx' : [int(idx) for idx in train_trajectories_idx],
            'validation_idx': [int(idx) for idx in val_trajectories_idx],
            'test_idx' : [int(idx) for idx in test_trajectories_idx],
        }

        save_to_json(
            training_config,
            f'{training_folder}/training_config_{training_idx}.json'
        )

        model.to(device)
        model.set_z_scale(standardize)

        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        best_loss = np.inf

        training_results = {}
        for state in out_variables:
            training_results[f"training_rmse_{state}"] = []
            training_results[f"one_step_validation_rmse_{state}"] = []
            training_results[f"multi_step_validation_rmse_{state}"] = []
        training_results[f"mean_position_error"] = []

        save_to_json(
            training_results,
            f'{training_folder}/training_results_{training_idx}.json'
            )

        for epochs in range(nb_epochs):
            
            train_loss = train_loop(
                model,
                optimizer,
                trainDataLoader,
                standardize,
                mean,
                std,
            )

            print(f'Epoch : {epochs}, train loss : {np.mean(train_loss)}')
            print()

            if epochs % validation_frequency == 0:

                val_loss = validation_loop(
                    model,
                    valDataLoader,
                    standardize = standardize,
                    mean = mean,
                    std = std,
                )

                print(f'Epoch : {epochs} | train loss : {np.mean(train_loss):.4f} | val loss : {np.mean(val_loss):.4f}')
                print()

                for idx, state in enumerate(out_variables):
                    training_results[f"training_rmse_{state}"].append(float(train_loss[idx]))
                    training_results[f"one_step_validation_rmse_{state}"].append(float(val_loss[idx]))

                # if np.mean(val_loss) < best_loss:
                #     best_loss = np.mean(val_loss)
                #     torch.save(model.state_dict(), f'{training_folder}/best_model_{training_idx}.pt')

        save_to_json(
            training_results,
            f'{training_folder}/training_results_{training_idx}.json'
            )
        
        del trainDataLoader, valDataLoader
        del model
        del train_in_state, train_in_control, train_out_state,
        del train_data_list, val_data_list, test_data_list
        del val_in_state, val_in_control, val_out_state
        del data_list

        gc.collect()
        torch.cuda.empty_cache()