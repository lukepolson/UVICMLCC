from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import time_step as ts1
from tf_agents.specs import array_spec
import numpy as np
import tensorflow as tf
from numpy.random import exponential
import random
from collections import deque



class Snake():
    # Initialize
    def __init__(self, BOARD_SIZE, MAX_HEALTH):  
        # Set Initial Position
        self.path_X = deque()
        self.path_X.append(np.random.randint(low=2, high=BOARD_SIZE-2))
        self.path_Y = deque()
        self.path_Y.append(np.random.randint(low=2, high=BOARD_SIZE-2))
        
        # Set Initial Health
        self.max_health = MAX_HEALTH
        self.health = MAX_HEALTH
        
        # Set if snake has just eaten
        self.just_eaten = False
    
    # Return all coordinates
    def get_path_coords(self):
        return self.path_X, self.path_Y
    
    def get_head_coords(self):
        return self.path_X[0], self.path_Y[0]
    
    def get_neck_coords(self):
        if len(self.path_X)>1:
            return self.path_X[1], self.path_Y[1]
        else:
            return self.path_X[0], self.path_Y[0]
    
    def get_tail_coords(self):
        return self.path_X[-1], self.path_Y[-1]
    
    # Return health
    def reduce_health(self):
        self.health-=1
    
    def get_health(self):
        return self.health
    
    def set_full_health(self):
        self.health = self.max_health
        
    def get_just_eaten(self):
        return self.just_eaten
    
    def set_just_eaten(self, boolean):
        self.just_eaten = boolean
    
    def move(self, action):
        i=0; j=0
        if (action==0):
            i = 1
        elif (action==1):
            j = 1
        elif (action==2):
            i = -1
        elif (action==3):
            j = -1
        else:
            raise ValueError('`action` should be 0 or 1 or 2 or 3.')
        self.path_X.appendleft(self.path_X[0]+i)
        self.path_Y.appendleft(self.path_Y[0]+j)
        if not self.just_eaten:
            self.path_X.pop()
            self.path_Y.pop()
  


class SnakeEnv(py_environment.PyEnvironment):
    
    def reset_board(self):
        
        # Create Snakes
        self.master_snake = Snake(self.BOARD_SIZE, self.M)
        head_x, head_y = self.master_snake.get_head_coords()
        self.MAX_COUNTER = random.choice(range(2,self.MC))
        self.COUNTER = 0
        
        
        # Create Board
        self._state = (np.array([([self.NB]*self.BOARD_SIZE) for i in range(self.BOARD_SIZE)], dtype=np.int32),
                       np.array([1, head_x/self.BOARD_SIZE, head_y/self.BOARD_SIZE], dtype=np.float32))
        self._episode_ended = False 
        master_init_x_coord, master_init_y_coord = self.master_snake.get_path_coords()
        
        #self._state[0][master_init_y_coord[0]][master_init_x_coord[0]] = self.HB
        
        # Special case for food spawn 0
        if (self.FOOD_SPAWN_MODE == 0):
            self.food_spawn_arr = np.ceil(exponential(5, size=100))
            self.current_food_spawn = 0 #current index of above array
            self.food_timer = 0
        # Special case for food spawn 1
        elif (self.FOOD_SPAWN_MODE == 1):
            self.foodspawn_assist()

        # Set if snake ate during previous turn, which is true for first spawn
        self.master_snake.set_just_eaten(True)     
    
    def __init__(self,
                 INIT_FOOD_REWARD,
                 STEP_REWARD,
                 KILL_STEP_REWARD,
                 DEATH_REWARD,
                 BOARD_SIZE,
                 MAX_HEALTH,
                 HEALTH_CUTOFF,
                 FOOD_SPAWN_MODE,
                 COUNTER_MODE,
                 MAX_COUNTER):
        
        self.COUNTER = 0 # used for initial snake growth
        self.FOOD_REWARD = INIT_FOOD_REWARD
        self.STEP_REWARD = STEP_REWARD
        self.DEATH_REWARD = DEATH_REWARD
        self.BOARD_SIZE = BOARD_SIZE
        self.KILL_STEP_REWARD = KILL_STEP_REWARD
        self.M = MAX_HEALTH
        self.HEALTH_CUTOFF = HEALTH_CUTOFF
        self.MC = MAX_COUNTER
        self.HB = np.array([1,1], dtype=np.int32) # Head block
        self.BB = np.array([1,0], dtype=np.int32) # Body Block
        self.FB = np.array([0,1], dtype=np.int32) # Food Block
        self.NB = np.array([0,0], dtype=np.int32) # Null Block
        if COUNTER_MODE == 0:
            self.MAX_COUNTER = 2
        elif COUNTER_MODE ==1:
            self.MAX_COUNTER = random.choice(range(2,self.MC))
        
        # For food spawn: 0 means exponential pdf spawn, 1 means only spawn if snake has just eaten
        self.FOOD_SPAWN_MODE = FOOD_SPAWN_MODE
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = (array_spec.BoundedArraySpec(
            shape=(BOARD_SIZE,BOARD_SIZE, 2), dtype=np.int32, minimum=0, maximum=1, name='observation'),
                                 array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.float32, minimum=0, maximum=1, name='health'))
        self.reset_board()

    def apply_actions(self, action):
        reward = 0
        reward += self.move_snake(action, 'master')
        return reward
    
    def print_food_reward(self):
        print(self.FOOD_REWARD)
    
    def food_reward_off(self):
        self.FOOD_REWARD = 0

    def move_snake(self, action, snake):
        
        reward = 0
        
        if snake=='master':
            head_x, head_y = self.master_snake.get_head_coords()
        elif snake=='enemy':
            head_x, head_y = self.enemy_snake.get_head_coords()
        
        kill_itself = False
        possible_random_choices = [0,1,2,3]
        
        # If action would kill the snake then choose a list of the other three actions
        
        # Action 0
        if (head_x+1 == self.BOARD_SIZE):
            if (action==0):
                kill_itself = True
            possible_random_choices.remove(0)
        elif np.array_equal(self._state[0][head_y][head_x+1],self.BB):
            if (action==0):
                kill_itself = True
            possible_random_choices.remove(0)
        
        # Action 1
        if (head_y+1 == self.BOARD_SIZE):
            if (action==1):
                kill_itself = True
            possible_random_choices.remove(1)
        elif np.array_equal(self._state[0][head_y+1][head_x],self.BB):
            if (action==1):
                kill_itself = True
            possible_random_choices.remove(1)
        
        # Action 2
        if (head_x-1 == -1):
            if (action==2):
                kill_itself = True
            possible_random_choices.remove(2)
        elif np.array_equal(self._state[0][head_y][head_x-1],self.BB):
            if (action==2):
                kill_itself = True
            possible_random_choices.remove(2)
            
        # Action 3
        if (head_y-1 == -1):
            if (action==3):
                kill_itself = True
            possible_random_choices.remove(3)
        elif np.array_equal(self._state[0][head_y-1][head_x],self.BB):
            if (action==3):
                kill_itself = True
            possible_random_choices.remove(3)
        
        if kill_itself:
            if snake=='master': reward+= self.KILL_STEP_REWARD
            # If nowhere to move then just die
            if (len(possible_random_choices) == 0):
                if snake=='master': self.master_snake.move(0)
                elif snake=='enemy': self.enemy_snake.move(0)
            # Else pick one of the random actions
            else:
                a = random.choice(possible_random_choices)
                if snake=='master': self.master_snake.move(a)
                elif snake=='enemy': self.enemy_snake.move(a)
        
        else:
            if snake=='master': self.master_snake.move(action)
            elif snake =='enemy': self.enemy_snake.move(action)
            
        return reward
        
           
    def action_spec(self):
        return self._action_spec
    
    def get_board(self):
        return np.array(self._state)

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.reset_board()
        return ts.restart(self._state)
                  
    def foodspawn_assist(self):
        coords = np.argwhere(((self._state[0] == self.NB).all(axis=2)))
        if (len(coords)>0):
            coord = random.choice(coords)
            self._state[0][coord[0]][coord[1]] = self.FB
        return None
    
    def spawn_food(self):
        # Spawn Food
        if (self.FOOD_SPAWN_MODE == 0):
            self.food_timer +=1
            if (self.food_spawn_arr[self.current_food_spawn]==self.food_timer):
                # Spawn Food
                self.foodspawn_assist()
                # Reset Timer
                self.food_timer = 0
                self.current_food_spawn = (self.current_food_spawn+1)%len(self.food_spawn_arr)
        elif (self.FOOD_SPAWN_MODE == 1):
            if (self.master_snake.get_just_eaten() and self.COUNTER>=self.MAX_COUNTER):
                self.foodspawn_assist()
    
    def check_if_hit(self, head_x, head_y):
        if (head_x == self.BOARD_SIZE or head_y == self.BOARD_SIZE \
            or head_x == -1 or head_y == -1 \
            or np.array_equal(self._state[0][head_y][head_x],self.BB)):
            return True
        else:
            return False
        
    
    def _step(self, action):
    
        reward = 0
        
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        
        # Reduce Health
        self.master_snake.reduce_health()
        self._state[1][0] = self.master_snake.get_health()/self.M
        
        # Get Previous tail coordinates
        master_tail_x_prev, master_tail_y_prev = self.master_snake.get_tail_coords()       
        # Apply actions to snake and enemy snake objects. Enemy snake uses external policy
        reward += self.apply_actions(action)
        
        # Get  New Head, NeckCoords
        master_head_x, master_head_y = self.master_snake.get_head_coords()
        master_neck_x, master_neck_y = self.master_snake.get_neck_coords()
        
        self.spawn_food()
        
        ## UPDATE THE BOARD -------------------
  
        # Set Neck and Tail
        self._state[0][master_neck_y][master_neck_x] = self.BB
        
        # If snake has not just eaten then set previous tail location to zero
        if not(self.master_snake.get_just_eaten()):
            self._state[0][master_tail_y_prev][master_tail_x_prev] = self.NB
            
        self.master_snake.set_just_eaten(False)
        
        ## UPDATE THE BOARD -------------------
        
        # HEAD BLOCK ------------------------------
        
        # Check to see if the new head coordinate is hitting anything on the updated board
        if (self.check_if_hit(master_head_x, master_head_y)):
            reward+=self.DEATH_REWARD
            self._episode_ended = True
        
        # If not then add the new head coord the the board
        else:
            # First check what block the head landed on
            block_master = self._state[0][master_head_y][master_head_x].copy()
            
            # Then update the block 
            self._state[0][master_head_y][master_head_x] = self.HB
            hx, hy = self.master_snake.get_head_coords()
            self._state[1][1], self._state[1][2] = hx/self.BOARD_SIZE, hy/self.BOARD_SIZE
            
            # For initial Snake growth
            if (self.COUNTER < self.MAX_COUNTER):
                self.COUNTER += 1
                self.master_snake.set_just_eaten(True)

            # Check if snake consumed food
            if(np.array_equal(block_master,self.FB)):
                if self.master_snake.get_health() < self.HEALTH_CUTOFF:
                    reward += self.FOOD_REWARD
                self.master_snake.set_full_health()
                self.master_snake.set_just_eaten(True)

            # If out of health then die
            if self.master_snake.get_health() == 0:
                reward+=self.DEATH_REWARD
                self._episode_ended = True
        
        # If Episode Ended
        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            reward += self.STEP_REWARD
            return ts.transition(self._state, reward=reward, discount=1.0) 