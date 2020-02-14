# Battlesnake 

This contains all the code for training the classic arcade game snake through reinforcement learning. Most of this code is based on the tf_agents library: check them out [here](https://github.com/tensorflow/agents)

This repository is organized into four seperate folders: Analysis, DATA, Snake_Emulator, and Training_Hyperparameters.

For an example of training, go to Analysis/initial_testing.py.ipynb

## Analysis

This folder contains notebook files that analyze training data obtained for different hyperparameter combinations. This training data can be obtained from the Stats files in the DATA folder.

## DATA

Stored here are

1. Frames
* Frames contain some examples of snake paths for the given models that were trained. They use the policy that was developed during training
2. Policies
* These are the final policies determined after training is completed. Policies essentially `take in a state s of the board and predict which action a to perform such that the future reward is maximized`. These can be loaded and used to make predictions.
3. Stats
* These are DataFrames which contain information such as average episode length and average reward throughout the training period. These DataFrames should be examined to see which models are the best.

The integer number at the end of each file is a key used to determine the set of hyperparameters used to train the model. For example, policy_2 was trained using the 2nd set of hyperparameters listed in the train_params.py file in the Training_Hyperparameters folder.

## Snake Emulator

This contains all the code of the snake game (the tf_agents snake environment) and the file Train.py used to actually train the snake from scratch.

## Training_Hyperparameters

This folder contains all the different hyperparameters that that the model will be trained using. *New lists of hyperparameters should only ever be added to the train_params.py.*
