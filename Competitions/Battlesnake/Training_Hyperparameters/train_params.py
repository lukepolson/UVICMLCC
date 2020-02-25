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

# Same as trial 0, except food reward decreased |DONE|
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

# Same as trial 0, except more NN neurons |DONE|
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

# Same as trial 0, except diff conv_layer_params |DONE|
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

# Same as trial_0 except discount factor reduced |DONE|
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

# Same as trial_0, more NN params than trial_2 |DONE|
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

# Same as trial_3 except more conv windows used |DONE|
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

# Uses best things found so far: smaller conv windows, more NN params, smaller discount factor |DONE|
trial_7 = {
"food_reward": 5,
"step_reward": 1,
"conv_layer_params": [[(16, (3, 3), 1), (32, (2 ,2), 1)]],
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

# Same as trial_7 but three layers of neurons |DONE|
trial_8 = {
"food_reward": 5,
"step_reward": 1,
"conv_layer_params": [[(16, (3, 3), 1), (32, (2 ,2), 1)]],
"fc_layer_params": [[100, 100, 100]],
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

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# --------------- These trials are now testing the snake that has health awareness ------
# --------------- The snake no longer no longer has any reward for eating food and its max
# --------------- health has been increased to 50.
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# |DONE|
trial_9 = {
"food_reward": 0,
"step_reward": 1,
"conv_layer_params": [[(16, (3, 3), 1), (32, (2 ,2), 1)]],
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
"n_iterations": 1000000
}

#-----------------------------------
# ----All trials below use 100 health snake and -25 reward for death  ---
#-----------------------------------


# Base trial for new snake with 100 health |DONE|

trial_10 = {
"food_reward": 0,
"step_reward": 1,
"conv_layer_params": [[(16, (3, 3), 1), (32, (2 ,2), 1)]],
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
"n_iterations": 2000000
}

# Try rewarding for food
trial_11 = {
"food_reward": 5,
"step_reward": 1,
"conv_layer_params": [[(16, (3, 3), 1), (32, (2 ,2), 1)]],
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
"n_iterations": 250000
}

# NEW PARAM ADDED: DEATH_REWARD- start at -100. Also decrease discount factor
trial_12 = {
"food_reward": 0,
"step_reward": 1,
"death_reward": -100,
"conv_layer_params": [[(16, (3, 3), 1), (32, (2 ,2), 1)]],
"fc_layer_params": [[100, 100]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 100000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.85,
"init_replay_buffer": 20000,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 250000
}


# NEW PARAM ADDED: FOOD_SPAWN_MODE: Food only spawns when snake has just eaten
# NEW PARAM ADDED: FOOD_REWARD_MODE: Rewards for eating food, higher rewards when health is low. 0 is regular, 1 is new mode.

# Food reward mode 0
trial_13 = {
"food_reward": 3,
"step_reward": 1,
"death_reward": -100,
"food_spawn_mode": 1,
"food_reward_mode": 0,
"conv_layer_params": [[(16, (3, 3), 1), (32, (2 ,2), 1)]],
"fc_layer_params": [[100, 100]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 250000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.85,
"init_replay_buffer": 20000,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 2500000
}

## ADDED KILL STEP REWARD
# Food reward mode 1
trial_14 = {
"food_reward": 3,
"step_reward": 1,
"death_reward": -100,
"kill_step_reward": -5,
"food_spawn_mode": 1,
"food_reward_mode": 1,
"conv_layer_params": [[(16, (3, 3), 1), (32, (2 ,2), 1)]],
"fc_layer_params": [[100, 100]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 100000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.85,
"init_replay_buffer": 200,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 500000
}

# Changed food reward policy
trial_15 = {
"food_reward": 3,
"step_reward": 1,
"death_reward": -100,
"kill_step_reward": -5,
"food_spawn_mode": 1,
"food_reward_mode": 1,
"conv_layer_params": [[(16, (3, 3), 1), (32, (2 ,2), 1)]],
"fc_layer_params": [[100, 100]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 100000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.85,
"init_replay_buffer": 200,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 500000
}

# Skip the CNN all together and use three fc_layers. Otherwise same as trial 15
# NEW BASE
trial_16 = {
"food_reward": 3,
"step_reward": 1,
"death_reward": -100,
"kill_step_reward": -5,
"food_spawn_mode": 1,
"food_reward_mode": 1,
"conv_layer_params": None,
"fc_layer_params": [[100, 100, 100]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 100000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.85,
"init_replay_buffer": 200,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 500000
}

# Try 4 hidden layers
trial_17 = {
"food_reward": 3,
"step_reward": 1,
"death_reward": -100,
"kill_step_reward": -5,
"food_spawn_mode": 1,
"food_reward_mode": 1,
"conv_layer_params": None,
"fc_layer_params": [[100, 100, 100, 100]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 100000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.85,
"init_replay_buffer": 200,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 500000
}

# Try 3 hidden layers with more neurons each
trial_18 = {
"food_reward": 3,
"step_reward": 1,
"death_reward": -100,
"kill_step_reward": -5,
"food_spawn_mode": 1,
"food_reward_mode": 1,
"conv_layer_params": None,
"fc_layer_params": [[150, 150, 150]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 100000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.85,
"init_replay_buffer": 200,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 500000
}

# Increase discount factor
trial_19 = {
"food_reward": 3,
"step_reward": 1,
"death_reward": -100,
"kill_step_reward": -5,
"food_spawn_mode": 1,
"food_reward_mode": 1,
"conv_layer_params": None,
"fc_layer_params": [[100, 100, 100]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 100000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.9,
"init_replay_buffer": 200,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 500000
}

# Reduce discount factor
trial_20 = {
"food_reward": 3,
"step_reward": 1,
"death_reward": -100,
"kill_step_reward": -5,
"food_spawn_mode": 1,
"food_reward_mode": 1,
"conv_layer_params": None,
"fc_layer_params": [[100, 100, 100]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 100000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.8,
"init_replay_buffer": 200,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 500000
}

# Increase food reward by a lot
trial_21 = {
"food_reward": 50,
"step_reward": 1,
"death_reward": -100,
"kill_step_reward": -5,
"food_spawn_mode": 1,
"food_reward_mode": 1,
"conv_layer_params": None,
"fc_layer_params": [[100, 100, 100]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 50000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.8,
"init_replay_buffer": 20000,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 2000000
}

# Increase food reward by a lot
trial_22 = {
"food_reward": 500,
"step_reward": 1,
"death_reward": -100,
"kill_step_reward": -5,
"food_spawn_mode": 1,
"food_reward_mode": 1,
"conv_layer_params": None,
"fc_layer_params": [[100, 100, 100]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 50000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.95,
"init_replay_buffer": 20000,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 500000
}

# Increase discount factor
trial_23 = {
"food_reward": 500,
"step_reward": 1,
"death_reward": -100,
"kill_step_reward": -5,
"food_spawn_mode": 1,
"food_reward_mode": 1,
"conv_layer_params": None,
"fc_layer_params": [[100, 100, 100]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 50000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.99,
"init_replay_buffer": 20000,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 10000000
}

# Same as before now with more neurons per layer
trial_24 = {
"food_reward": 500,
"step_reward": 1,
"death_reward": -100,
"kill_step_reward": -5,
"food_spawn_mode": 1,
"food_reward_mode": 1,
"conv_layer_params": None,
"fc_layer_params": [[150, 150, 150]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 50000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.99,
"init_replay_buffer": 20000,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 10000000
}

# NEW PARAM: Health cut off- point at which one is rewarded for eating. In past trials was 50: now set to 20
trial_25 = {
"food_reward": 500,
"health_cutoff": 20,
"step_reward": 1,
"death_reward": -100,
"kill_step_reward": -5,
"food_spawn_mode": 1,
"food_reward_mode": 1,
"conv_layer_params": None,
"fc_layer_params": [[100, 100, 100]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 50000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.99,
"init_replay_buffer": 20000,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 10000000
}

trial_26 = {
"food_reward": 500,
"health_cutoff": 20,
"step_reward": 1,
"death_reward": -100,
"kill_step_reward": -5,
"food_spawn_mode": 1,
"food_reward_mode": 1,
"conv_layer_params": None,
"fc_layer_params": [[100, 100, 100]],
"optimizer_learning_rate": 2.5e-4,
"optimizer_decay": 0.95,
"optimizer_momentum": 0,
"optimizer_epsilon": 0.00001,
"epsilon_decay_steps": 50000,
"epsilon_final": 0.01,
"target_update_period": 2000,
"discount_factor": 0.99,
"init_replay_buffer": 20000,
"dataset_sample_batch_size": 64,
"dataset_num_steps": 2,
"dataset_num_parallel_calls":3,
"n_iterations": 15000000
}


trials = [trial_0, trial_1, trial_2, trial_3, trial_4, trial_5, trial_6, trial_7, trial_8, trial_9,
         trial_10, trial_11, trial_12, trial_13, trial_14, trial_15, trial_16, trial_17, trial_18,
          trial_19, trial_20, trial_21, trial_22, trial_23, trial_24, trial_25]
dfs = [pd.DataFrame(trial) for trial in trials]
df = pd.concat(dfs).reset_index()
df.to_pickle('train_params.pkl')