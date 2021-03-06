## ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

import abc
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
import numpy as np
from numpy.random import exponential
import random
import pandas as pd



from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.environments.wrappers import ActionRepeat
from tf_agents.environments import batched_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tensorflow.keras.layers import Reshape
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.trajectories.trajectory import to_transition
from tf_agents.utils.common import function
from tf_agents.policies.policy_saver import PolicySaver

import logging

from collections import deque

from Snake_v6 import Snake
from Snake_v6 import SnakeEnv

class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")
            
## ------------------------------------------------------------------------------
## -------------------------Obtain all hyperparameters---------------------------
## ------------------------------------------------------------------------------

# sys.argv is given in bash: this line updates all local
# variables with the values from the specified row in the
# DataFrame
params_df = pd.read_pickle('../Training_Hyperparameters/train_params.pkl')
print(sys.argv[1])
II = int(sys.argv[1])
locals().update(dict(params_df.iloc[II]))

## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------
            
tf.random.set_seed(888)
env = SnakeEnv(BOARD_SIZE=int(board_size),
               MAX_HEALTH=100,
               FOOD_REWARD=food_reward,
               STEP_REWARD=step_reward,
               DEATH_REWARD=death_reward,
               KILL_STEP_REWARD = kill_step_reward,
               REWARD_TYPE = reward_type,
               FOOD_REWARD_TYPE = food_reward_type, 
               FOOD_SPAWN_MODE=food_spawn_mode,
               MAX_COUNTER=int(max_counter),
               HEALTH_CUTOFF=health_cutoff,
               DRF_MC_HIGH=DRF_MC_high,
               DRF_MC_LOW=DRF_MC_low,
               IMMORTAL=immortal,
               WALL_PENALTY=wall_penalty)
tf_env = tf_py_environment.TFPyEnvironment(env)

## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------

# Preprocessing layers
board_preprocessing = Sequential([
    keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32)),
    keras.layers.Flatten()
])
health_preprocessing =  keras.layers.Flatten()

# Layers params are specified by local variables ovtained from DataFrame
act_net = ActorDistributionNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=(board_preprocessing, health_preprocessing),
    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1),
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params,
    batch_squash=False)

## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------

# Create variable that counts the number of training steps
train_step = tf.Variable(0)
# Create optimizer 
optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=optimizer_learning_rate,
                                                decay=optimizer_decay, momentum=optimizer_momentum,
                                                epsilon=optimizer_epsilon, centered=True)

agent = reinforce_agent.ReinforceAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    actor_network=act_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step,
    gamma=discount_factor)

agent.initialize()
# Speed up as tensorflow function
agent.train = function(agent.train)

## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    # Determines the data spec type
    data_spec=agent.collect_data_spec,
    # The number of trajectories added at each step
    batch_size=tf_env.batch_size,
    # This can store n_iterations trajectories (note: requires a lot of RAM)
    max_length=n_iterations)

# Create the observer that adds trajectories to the replay buffer
replay_buffer_observer = replay_buffer.add_batch


def collect_episode(environment, policy, num_episodes):

    episode_counter = 0
    environment.reset()

    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)

    if traj.is_boundary():
        episode_counter += 1

## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------

train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

logging.getLogger().setLevel(logging.INFO)

## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------

collect_driver = DynamicStepDriver(
    tf_env, # Env to play with
    agent.collect_policy, # Collect policy of the agent
    observers=[replay_buffer_observer] + train_metrics, # pass to all observers
    num_steps=1) 
# Speed up as tensorflow function
collect_driver.run = function(collect_driver.run)

initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                        tf_env.action_spec())
init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(init_replay_buffer)],
    num_steps=init_replay_buffer) 
final_time_step, final_policy_state = init_driver.run()

## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------

dataset = replay_buffer.as_dataset(
    sample_batch_size=dataset_sample_batch_size,
    num_steps=dataset_num_steps,
    num_parallel_calls=dataset_num_parallel_calls).prefetch(dataset_num_parallel_calls)

## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------

if __name__ == "__main__":

    # a) For storing data
    training_info = [[], [], [], []]
    def add_metrics(arr, train_metrics):
        arr[0].append(train_metrics[0].result().numpy())
        arr[1].append(train_metrics[1].result().numpy())
        arr[2].append(train_metrics[2].result().numpy())
        arr[3].append(train_metrics[3].result().numpy())
    
    # b) For training agent
    def train_agent(n_iterations):
        time_step = None
        # Get initial policy state
        policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
        # Create iterator over dataset and loop
        iterator = iter(dataset)
        
        for iteration in range(n_iterations):
            # Pass current time step and policy state to get next time step and policy state
            time_step, policy_state = collect_driver.run(time_step, policy_state)
            # Sample a batch of trajectories from the dataset, pass to the train method
            trajectories, buffer_info = next(iterator)
            train_loss = agent.train(trajectories)
            if iteration % 10000 == 0:
                # This adds training data
                add_metrics(training_info, train_metrics)
                f = open("../DATA/Single/progress_{}.txt".format(II), "w")
                f.write("Current iteration: {} \n".format(iteration))
                f.close()

                
    train_agent(n_iterations=n_iterations)
    
    # c) For storing frames
    def get_vid_frames(policy, filename, num_episodes=100, fps=2):
        frames = []
        for _ in range(num_episodes):
            time_step = tf_env.reset()
            frames.append(np.abs(env.get_board()))
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = tf_env.step(action_step.action)
                frames.append(np.abs(env.get_board()))
        return frames

    # Store Data
    df = pd.DataFrame(np.array(training_info).T, columns=['N_Ep', 'Env_Steps', 'Avf_RM', 'Avg_EPLM'])
    df.to_csv('../DATA/Single/stats_{}.txt'.format(II), index=False, mode="a")
    
    # Store Frames
    frames = get_vid_frames(agent.policy, "trained-agent")
    with open('../DATA/Single/frames_{}.pkl'.format(II), 'wb') as f:
        pickle.dump(frames, f)
        
    # Store Model
    my_policy = agent.policy
    saver = PolicySaver(my_policy, batch_size=None)
    saver.save('../DATA/Single/policy_{}'.format(II))