duffing_state_variables = ['x', 'y']
duffing_observed_state_variables = ['x']
duffing_control_variables = ['f']


vanderpol_state_variables = ['x', 'y']
vanderpol_observed_state_variables = ['x']
vanderpol_control_variables = []

system_configuration = {
    'vanderpol' : {
        'state' : vanderpol_state_variables,
        'observed_state' : vanderpol_observed_state_variables,
        'control' : vanderpol_control_variables
    },
    'duffing': {
        'state' : duffing_state_variables,
        'observed_state' : duffing_observed_state_variables,
        'control' : duffing_control_variables
    },
}

max_sequence_duration = 5.

nb_trajectories = 200

seed = 42
device = 'cuda:0'
train_set_pct = 0.7
val_set_pct = 0.2
test_set_pct = 0.1
eval_trajectory_duration = 20.
validation_frequency = 2
nb_epochs = 301