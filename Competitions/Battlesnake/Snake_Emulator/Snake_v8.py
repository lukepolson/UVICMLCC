from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import time_step as ts1
from tf_agents.specs import array_spec
import numpy as np
import tensorflow as tf
from numpy.random import exponential
import random
from collections import deque

# REWARD STUFF --------------------------------------------------
def DRF_v0(Lt, D_Neck, D_Head):
    if Lt>2:
        return np.log((Lt+D_Neck)/(Lt+D_Head)) / np.log(Lt)
    else:
        return 0
    
def DRF_v1(Lt, D_Neck, D_Head, health, max_health):
    if Lt>2:
        return (1-health/max_health) * np.log((Lt+D_Neck)/(Lt+D_Head)) / np.log(Lt)
    else:
        return 0

def DRF_v2(Lt, D_Neck, D_Head, health, max_health):
    if Lt>2:
        return (1-2*health/max_health) * np.log((Lt+D_Neck)/(Lt+D_Head)) / np.log(Lt)
    else:
        return 0
    
def DRF_v3(Lt, D_Neck, D_Head, health, max_health, health_cutoff, mc_high, mc_low):
    if Lt>2:
        if health<health_cutoff:
            return mc_low*np.log((Lt+D_Neck)/(Lt+D_Head)) / np.log(Lt)
        else:
            return -mc_high*np.log((Lt+D_Neck)/(Lt+D_Head)) / np.log(Lt)
    else:
        return 0

def DRF_v4(Lt, D_Neck, D_Head, health, max_health, health_cutoff=30):
    if Lt>2:
        if health<health_cutoff:
            return np.log((Lt+D_Neck)/(Lt+D_Head)) / np.log(Lt)
        else:
            return -10*np.log((Lt+D_Neck)/(Lt+D_Head)) / np.log(Lt)
    else:
        return 0 
    
def food_reward_v1(health, health_cutoff):
    if health<health_cutoff:
        return 1
    else:
        return 0

def food_reward_v2(health, max_health):
    return 1-2*health/max_health

def food_reward_v3(health, health_cutoff):
    if health<health_cutoff:
        return 1
    else:
        return -1

# END REWARD STUFF --------------------------------------------------


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
        self.init_growth = True
    
    def get_length(self):
        return len(self.path_X)
    
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
    
    def get_init_growth(self):
        return self.init_growth
        
    def set_init_growth(self, boolean):
        self.init_growth = boolean
    
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
        if not (self.just_eaten or self.init_growth):
            self.path_X.pop()
            self.path_Y.pop()
  


class SnakeEnv(py_environment.PyEnvironment):
    
    def reset_board(self):
        
        # Create Snakes
        self.master_snake = Snake(self.BOARD_SIZE, self.M)
        head_x, head_y = self.master_snake.get_head_coords()
        tail_x, tail_y = self.master_snake.get_tail_coords()
        self.COUNTER = 0
        self.food_coord = None
          
        # Create Board
        self._state = (np.array([([self.NB]*self.BOARD_SIZE) for i in range(self.BOARD_SIZE)], dtype=np.int32),
                       np.array([1, head_x/self.BOARD_SIZE, head_y/self.BOARD_SIZE,
                                 tail_x/self.BOARD_SIZE, tail_y/self.BOARD_SIZE], dtype=np.float32))
        self._episode_ended = False 
        master_init_x_coord, master_init_y_coord = self.master_snake.get_path_coords()
        
        self._state[0][master_init_y_coord[0]][master_init_x_coord[0]] = self.HB
        
        # Special case for food spawn 0
        if (self.FOOD_SPAWN_MODE == 0):
            self.food_spawn_arr = np.ceil(exponential(5, size=100))
            self.current_food_spawn = 0 #current index of above array
            self.food_timer = 0
        # Special case for food spawn 1
        elif (self.FOOD_SPAWN_MODE == 1):
            self.foodspawn_assist()

        # Set if snake ate during previous turn, which is true for first spawn
        # self.master_snake.set_just_eaten(True)     
        
    def __init__(self,
                 FOOD_REWARD,
                 STEP_REWARD,
                 KILL_STEP_REWARD,
                 DEATH_REWARD,
                 REWARD_TYPE,
                 WALL_PENALTY,
                 FOOD_REWARD_TYPE,
                 BOARD_SIZE,
                 MAX_HEALTH,
                 FOOD_SPAWN_MODE,
                 MAX_COUNTER,
                 HEALTH_CUTOFF,
                 DRF_MC_HIGH,
                 DRF_MC_LOW,
                 IMMORTAL,
                 SNAKE_PROTECT):
        
        self.COUNTER = 0 # used for initial snake growth
        self.FOOD_REWARD = FOOD_REWARD
        self.FOOD_REWARD_TYPE = FOOD_REWARD_TYPE
        self.STEP_REWARD = STEP_REWARD
        self.DEATH_REWARD = DEATH_REWARD
        self.REWARD_TYPE = REWARD_TYPE
        self.BOARD_SIZE = BOARD_SIZE
        self.KILL_STEP_REWARD = KILL_STEP_REWARD
        self.M = MAX_HEALTH
        self.MAX_COUNTER = MAX_COUNTER
        self.HEALTH_CUTOFF = HEALTH_CUTOFF
        self.DRF_MC_HIGH = DRF_MC_HIGH
        self.DRF_MC_LOW = DRF_MC_LOW
        self.IMMORTAL = IMMORTAL
        self.WALL_PENALTY = WALL_PENALTY
        self.SNAKE_PROTECT = SNAKE_PROTECT
        
        self.HB = np.array([1,1,0], dtype=np.int32) # Head block
        self.BB = np.array([1,0,0], dtype=np.int32) # Body Block
        self.TB = np.array([0,1,0], dtype=np.int32) # Body Block
        self.FB = np.array([0,0,1], dtype=np.int32) # Food Block
        self.NB = np.array([0,0,0], dtype=np.int32) # Null Block
        
        # For food spawn: 0 means exponential pdf spawn, 1 means only spawn if snake has just eaten
        self.FOOD_SPAWN_MODE = FOOD_SPAWN_MODE
        self.food_coord = None
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = (array_spec.BoundedArraySpec(
            shape=(BOARD_SIZE,BOARD_SIZE, 3), dtype=np.int32, minimum=0, maximum=1, name='observation'),
                                 array_spec.BoundedArraySpec(
            shape=(5,), dtype=np.float32, minimum=0, maximum=1, name='health'))
        self.action_space = self._action_spec
        self.observation_space = self._observation_spec
        self.reset_board()

    def apply_actions(self, action):
        reward = 0
        if self.SNAKE_PROTECT:
            reward += self.move_snake(action, 'master')
        else:
            self.master_snake.move(action)
        return reward

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
            self.food_coord = coord
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
            if (self.master_snake.get_just_eaten()):
                self.foodspawn_assist()
    
    def check_if_hit(self, head_x, head_y):
        if (head_x == self.BOARD_SIZE or head_y == self.BOARD_SIZE \
            or head_x == -1 or head_y == -1 \
            or np.array_equal(self._state[0][head_y][head_x],self.BB)):
            return True
        else:
            return False
        
    def check_wall(self, head_x, head_y):
        if (head_x == self.BOARD_SIZE-1 or head_y == self.BOARD_SIZE-1 \
            or head_x == 0 or head_y == 0):
            return self.WALL_PENALTY
        else:
            return 0
        
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
        
        # Get new Head, Neck Coords
        master_head_x, master_head_y = self.master_snake.get_head_coords()
        master_neck_x, master_neck_y = self.master_snake.get_neck_coords()
        master_tail_x, master_tail_y = self.master_snake.get_tail_coords()
        
        reward += self.check_wall(master_head_x, master_head_y)
        
        # Get head and neck distance to food
        D_head = np.sqrt((master_head_x - self.food_coord[1])**2 + (master_head_y - self.food_coord[0])**2)
        D_neck = np.sqrt((master_neck_x - self.food_coord[1])**2 + (master_neck_y - self.food_coord[0])**2)
        
        self.spawn_food()
        
        ## UPDATE THE BOARD -------------------
  
        # Set Neck and Tail
        self._state[0][master_neck_y][master_neck_x] = self.BB
        self._state[0][master_tail_y][master_tail_x] = self.TB
        
        # If snake has not just eaten or in initial growth then set previous tail location to zero
        if not(self.master_snake.get_just_eaten() or self.master_snake.get_init_growth()):
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
            if (self.COUNTER <= self.MAX_COUNTER):
                self.COUNTER += 1
                if (self.COUNTER == self.MAX_COUNTER):
                    self.master_snake.set_init_growth(False)

            # Check if snake consumed food
            if(np.array_equal(block_master,self.FB)):
                if self.FOOD_REWARD_TYPE==0:
                    reward += self.FOOD_REWARD
                elif self.FOOD_REWARD_TYPE==1:
                    reward += food_reward_v1(self.master_snake.get_health(), self.HEALTH_CUTOFF)
                elif self.FOOD_REWARD_TYPE==2:
                    reward += food_reward_v2(self.master_snake.get_health(), self.M)
                elif self.FOOD_REWARD_TYPE==3:
                    reward += food_reward_v3(self.master_snake.get_health(), self.HEALTH_CUTOFF)
                self.master_snake.set_full_health()
                self.master_snake.set_just_eaten(True)
            else: 
                if self.REWARD_TYPE == 0:
                    reward += DRF_v0(self.master_snake.get_length(), D_neck, D_head)
                elif self.REWARD_TYPE == 1:
                    reward += DRF_v1(self.master_snake.get_length(), D_neck, D_head,
                                     self.master_snake.get_health(), self.M)
                elif self.REWARD_TYPE == 2:
                    reward += DRF_v2(self.master_snake.get_length(), D_neck, D_head,
                                     self.master_snake.get_health(), self.M)
                elif self.REWARD_TYPE == 3:
                    reward += DRF_v3(self.master_snake.get_length(), D_neck, D_head,
                                     self.master_snake.get_health(), self.M, self.HEALTH_CUTOFF,
                                    self.DRF_MC_HIGH, self.DRF_MC_LOW)
                elif self.REWARD_TYPE == 4:
                    reward += DRF_v4(self.master_snake.get_length(), D_neck, D_head,
                                     self.master_snake.get_health(), self.M)
            # If out of health then die
            if self.master_snake.get_health() == 0:
                if self.IMMORTAL:
                    self.master_snake.set_just_eaten(True)
                    self.master_snake.set_full_health()
                    self._state[0][self.food_coord[0]][self.food_coord[1]] = self.NB
                    self._state[1][0] = self.master_snake.get_health()/self.M
                else:
                    reward+=self.DEATH_REWARD
                    self._episode_ended = True
        
        # Clip Reward to (-1,1) interval
        reward = np.clip(reward, a_min=-1, a_max=1)
        
        # If Episode Ended
        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            reward += self.STEP_REWARD
            return ts.transition(self._state, reward=reward, discount=1.0) 