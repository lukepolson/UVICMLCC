from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
import numpy as np
from numpy.random import exponential
import random
from collections import deque

class Snake():
    # Initialize
    def __init__(self, BOARD_SIZE, MAX_HEALTH):  
        # Set Initial Position
        self.path_X = deque()
        self.path_X.append(np.random.randint(low=2, high=BOARD_SIZE-2))
        self.path_X.append(self.path_X[0])
        self.path_X.append(self.path_X[0])
        self.path_Y = deque()
        self.path_Y.append(np.random.randint(low=2, high=BOARD_SIZE-2))
        self.path_Y.append(self.path_Y[0]+1)
        self.path_Y.append(self.path_Y[0]+2)
        
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
        return self.path_X[1], self.path_Y[1]
    
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
        self.master_snake = Snake(self.BOARD_SIZE, self.MAX_HEALTH)
        
        # Create Board
        self._state = {'board': 0, 'health': 0}
        self._state['board'] = [([[0, 0, 0]]*self.BOARD_SIZE) for i in range(self.BOARD_SIZE)]
        # Set To Max Health
        self._state['health'] = 1
        self._episode_ended = False 
        master_x_coords, master_y_coords = self.master_snake.get_path_coords()
        for i, (x, y) in enumerate(zip(master_x_coords, master_y_coords)):
                if i == 0:
                    self._state['board'][y][x] = [0, 1, 0]
                else:
                    self._state['board'][y][x] = [1, 0, 0]
        
        # Set food timer
        self.food_spawn_arr = np.ceil(exponential(5, size=100))
        self.current_food_spawn = 0 #current index of above array
        self.food_timer = 0
        # Maybe store other snakes healths in an array?

        # Set if snake ate during previous turn
        self.just_eaten = False          
    
    def __init__(self, FOOD_REWARD, STEP_REWARD, BOARD_SIZE, MAX_HEALTH):
        
        self.FOOD_REWARD = FOOD_REWARD
        self.STEP_REWARD = STEP_REWARD
        self.BOARD_SIZE = BOARD_SIZE
        self.MAX_HEALTH = MAX_HEALTH
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = {'board': array_spec.BoundedArraySpec(
            shape=(BOARD_SIZE,BOARD_SIZE, 3), dtype=np.int32, maximum=2, minimum=-1),
                                  'health': array_spec.BoundedArraySpec(
            shape=(), dtype=np.float32)}
                                  
        self.reset_board()
   
    def action_spec(self):
        return self._action_spec
    
    def get_board(self):
        return np.abs(np.array(self._state['board']))

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.reset_board()
        return ts.restart({'board':np.array(self._state['board'], dtype=np.int32),
                          'health': np.array(self._state['health'], dtype=np.float32)})
                  
    def foodspawn_assist(self):
        coords = np.argwhere(np.all(np.array(self._state['board']) == [0, 0, 0], axis=-1))
        coord = random.choice(coords)
        self._state['board'][coord[0]][coord[1]] = [0, 0, 1]
        return None

    def _step(self, action, data=None):
       
        reward = 0
        
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        
        # Reduce Health
        self.master_snake.reduce_health()
        self._state['health'] = self.master_snake.get_health()/self.MAX_HEALTH
        
        # Get Previous tail coordinates
        master_tail_x, master_tail_y = self.master_snake.get_tail_coords()
        
        # Apply Action
        try: self.master_snake.move(action)
        except ValueError as error:
            raise ValueError('`action` should be 0 or 1 or 2 or 3.')
        
        # Get  New Head Coords
        master_head_x, master_head_y = self.master_snake.get_head_coords()
        master_neck_x, master_neck_y = self.master_snake.get_neck_coords()
        
        # Spawn Food
        self.food_timer +=1
        if (self.food_spawn_arr[self.current_food_spawn]==self.food_timer):
            # Spawn Food
            self.foodspawn_assist()
            # Reset Timer
            self.food_timer = 0
            self.current_food_spawn = (self.current_food_spawn+1)%len(self.food_spawn_arr)
        
        # Check to see if out of bounds or hit itself
        if (master_head_x == self.BOARD_SIZE or master_head_y == self.BOARD_SIZE \
            or master_head_x == -1 or master_head_y == -1 \
            or self._state['board'][master_head_y][master_head_x] == [1, 0, 0]):
            self._episode_ended = True
        
        # Else perform snake step
        else:
            # Check What block snake has landed on. If -1 then
            # it landed on food
            block = self._state['board'][master_head_y][master_head_x]

            # Set Head Value to 2
            self._state['board'][master_head_y][master_head_x] = [0,1,0]
            # Set Neck to 1
            self._state['board'][master_neck_y][master_neck_x] = [1,0,0]
            # If snake has not just eaten then set tail location to zero
            if not(self.master_snake.get_just_eaten()):
                self._state['board'][master_tail_y][master_tail_x] = [0,0,0]
            self.master_snake.set_just_eaten(False)

            # Check if snake consumed food
            if(block==[0,0,1]):
                self.master_snake.set_full_health()
                self.master_snake.set_just_eaten(True)
                reward += self.FOOD_REWARD

        # If out of health then die
        if self.master_snake.get_health() == 0:
            self._episode_ended = True
        
        # If Episode Ended
        if self._episode_ended:
            reward = 0
            return ts.termination({'board':np.array(self._state['board'], dtype=np.int32),
                                  'health': np.array(self._state['health'], dtype=np.float32)},
                                  reward)
        else:
            reward += self.STEP_REWARD
            return ts.transition({'board':np.array(self._state['board'], dtype=np.int32),
                                  'health': np.array(self._state['health'], dtype=np.float32)},
                                 reward=reward, discount=1.0) 