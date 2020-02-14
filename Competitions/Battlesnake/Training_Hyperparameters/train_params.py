## DO NOT DELETE ANYTHING FROM THIS FILE: ONLY ADD MORE TRIALS TO DATAFRAME
import pandas as pd

# This is a list of the trials that have been completed and data stored
# Trial_0, Trial_1, Trial_2, Trial_3, Trial_4, Trial_5, Trial_6



trial_0 = {
"food_reward": 5,
"step_reward": 1,
"conv_layer_params": [[(16, (4, 4), 2), (32, (3 ,3), 1)]],
"fc_layer_params": [[50, 50]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 100000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.97,
"init_replay_buffer": 20000,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 4000000
}

# Same as trial 0, except food reward decreased
trial_1 = {
"food_reward": 3,
"step_reward": 1,
"conv_layer_params": [[(16, (4, 4), 2), (32, (3 ,3), 1)]],
"fc_layer_params": [[50, 50]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 100000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.97,
"init_replay_buffer": 20000,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 4000000
}

# Same as trial 0, except more NN neurons
trial_2 = {
"food_reward": 5,
"step_reward": 1,
"conv_layer_params": [[(16, (4, 4), 2), (32, (3 ,3), 1)]],
"fc_layer_params": [[75, 75]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 100000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.97,
"init_replay_buffer": 20000,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 4000000
}

# Same as trial 0, except diff conv_layer_params
trial_3 = {
"food_reward": 5,
"step_reward": 1,
"conv_layer_params": [[(16, (3, 3), 1), (32, (2 ,2), 1)]],
"fc_layer_params": [[50, 50]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 100000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.97,
"init_replay_buffer": 20000,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 4000000
}

# Same as trial_0 except discount factor reduced
trial_4 = {
"food_reward": 5,
"step_reward": 1,
"conv_layer_params": [[(16, (4, 4), 2), (32, (3 ,3), 1)]],
"fc_layer_params": [[50, 50]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 100000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.90,
"init_replay_buffer": 20000,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 4000000
}

# Same as trial_0, more NN params than trial_2
trial_5 =  {
"food_reward": 5,
"step_reward": 1,
"conv_layer_params": [[(16, (4, 4), 2), (32, (3 ,3), 1)]],
"fc_layer_params": [[100, 100]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 100000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.97,
"init_replay_buffer": 20000,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 4000000
}

# Same as trial_3 except more conv windows used
trial_6 = {
"food_reward": 5,
"step_reward": 1,
"conv_layer_params": [[(32, (3, 3), 1), (64, (2 ,2), 1)]],
"fc_layer_params": [[50, 50]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 100000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.97,
"init_replay_buffer": 20000,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 4000000
}

# Uses best things found so far: smaller conv windows, more NN params, smaller discount factor
trial_7 = {
"food_reward": 5,
"step_reward": 1,
"conv_layer_params": [[(16, (4, 4), 2), (32, (3 ,3), 1)]],
"fc_layer_params": [[100, 100]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 100000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.90,
"init_replay_buffer": 20000,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 4000000
}

trials = [trial_0, trial_1, trial_2, trial_3, trial_4, trial_5, trial_6, trial_7]
dfs = [pd.DataFrame(trial) for trial in trials]
df = pd.concat(dfs).reset_index()
df.to_pickle('train_params.pkl')