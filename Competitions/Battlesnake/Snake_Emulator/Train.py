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
import numpy as np
from numpy.random import exponential
import random
import pandas as pd

# To get smooth ani|mations
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '\\Users\\lukep\\ffmpeg\\ffmpeg-20200206-343ccfc-win64-static\\bin\\'
import matplotlib.animation as animation
import matplotlib as mpl
import ffmpeg
mpl.rc('animation', html='jshtml')


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
from tf_agents.networks.q_network import QNetwork
from tensorflow.keras.layers import Reshape
from tf_agents.agents.dqn.dqn_agent import DqnAgent
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

from Snake import Snake
from Snake import SnakeEnv

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
params_df = pd.read_pickle('..\\Training_Hyperparameters\\train_params.pkl')
print(sys.argv[1])
II = int(sys.argv[1])
locals().update(dict(params_df.iloc[II]))

## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------
            
tf.random.set_seed(888)
env = SnakeEnv(FOOD_REWARD=food_reward, STEP_REWARD=step_reward, BOARD_SIZE=8, MAX_HEALTH=20)
tf_env = tf_py_environment.TFPyEnvironment(env)

## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------
## ------------------------------------------------------------------------------

preprocessing_layer = keras.layers.Lambda(
                          lambda obs: tf.cast(obs, np.float32) / 2.)

# Layers params are specified by local variables ovtained from DataFrame
q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
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
# Computes epsilon for epsilon greedy policy given the training step
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0, # initial ε
    decay_steps=epsilon_decay_steps, 
    end_learning_rate=epsilon_final) # final ε

agent = DqnAgent(tf_env.time_step_spec(),
                 tf_env.action_spec(),
                 q_network=q_net,
                 optimizer=optimizer,
                 target_update_period=target_update_period, 
                 td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                 gamma=discount_factor, # discount factor
                 train_step_counter=train_step,
                 epsilon_greedy=lambda: epsilon_fn(train_step))
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
    # This can store 4 million trajectories (note: requires a lot of RAM)
    max_length=4000000)

# Create the observer that adds trajectories to the replay buffer
replay_buffer_observer = replay_buffer.add_batch

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
            # Collects experience for 4 steps then broadcasts trajectories to replay buffer
            time_step, policy_state = collect_driver.run(time_step, policy_state)
            # Sample a batch of trajectories from the dataset, pass to the train method
            trajectories, buffer_info = next(iterator)
            train_loss = agent.train(trajectories)
            print("\r{} loss:{:.5f}".format(
                iteration, train_loss.loss.numpy()), end="")
            if iteration % 10000 == 0:
                add_metrics(training_info, train_metrics)
    train_agent(n_iterations=n_iterations)
    
    # c) For storing frames
    def get_vid_frames(policy, filename, num_episodes=20, fps=2):
        frames = []
        for _ in range(num_episodes):
            time_step = tf_env.reset()
            frames.append(np.abs(env.get_board())*100)
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = tf_env.step(action_step.action)
                frames.append(np.abs(env.get_board())*100)
        return frames

    # Store Data
    df = pd.DataFrame(np.array(training_info).T, columns=['N_Ep', 'Env_Steps', 'Avf_RM', 'Avg_EPLM'])
    df.to_csv('../DATA/stats_{}.txt'.format(II), index=False, mode="a")
    
    # Store Frames
    frames = get_vid_frames(agent.policy, "trained-agent")
    with open('../DATA/frames_{}.pkl'.format(II), 'wb') as f:
        pickle.dump(frames, f)
        
    # Store Model
    my_policy = agent.policy
    saver = PolicySaver(my_policy, batch_size=None)
    saver.save('..\\DATA\\policy_{}'.format(II))
    
    
    
    
    


