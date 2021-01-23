#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:03:26 2021

@author: Karl Meisner-Jensen

Used as the environment for a DQN agent
"""

 
import pygame
import numpy as np
import random
import os

# Define some colors
black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)

# Set height and width for pygame window
screenheight = 600
screenwidth = 200

# Default frame rate
fast = True

frame_fast = 960
frame_slow = 30

# Environment rules
num_balls = 1
paddle_step = 10
paddle_size = 40

# Seed for every random func
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)


# Unused, intended to divide environment into quadrants for NN input
def make_quandrants(xz, yz,dead_zone=0):
    zones = []
    if dead_zone != 0:
        zones.append([0,0])
    
    z_x = screenwidth / xz
    z_y = (screenheight - dead_zone) / yz
    current_x = 0
    current_y = 0
    
    for i in range(xz):
        current_y = dead_zone
        for r in range(yz):
            zone = [current_x, current_y]
            zones.append(zone)
            current_y += z_y
        current_x += z_x
    return np.array(zones), z_x, z_y


class Ball(pygame.sprite.Sprite):
      
    width = 10
    height = 10

    def __init__(self):
        super().__init__()
        
        # Random start position for the ball, not too unfair though
        self.x = random.randrange(0, screenwidth)
        self.y = random.randrange(0, screenheight - 200)
        
        # Direction is skewed by default to make dumb implementation of random bounce work
        self.direction = np.array([.49,.51])
        self.speed = 10
        
        # Create the image of the ball
        self.image = pygame.Surface([self.width, self.height])
 
        # Color the ball
        self.image.fill(white)
 
        # Get a rectangle object that shows where our image is
        self.rect = self.image.get_rect()
                        
    def get_quadrant(self, zones, z_x, z_y, dead_zone):
        """
        Parameters
        ----------
        zones : np array
            Two-Dimensional array with pairs of x/y values denoting the starting
            position of the various quadrants
        z_x : Float
            Length of quadrants.
        z_y : Float
            Height of quadrants.
        dead_zone : Bool
            Whether or not the 0'th index is a dead zone.

        Returns
        -------
        np array
            Index of the current quadrant the ball exists in.
            
        Unused
        """
        
        if dead_zone == False:
            x_intersects = np.where(np.logical_and(zones[0:,0] <= self.x, zones[0:,0] + z_x > self.x))
            y_intersects = np.where(np.logical_and(zones[0:,1] <= self.y, zones[0:,1] + z_y > self.y))
            return np.intersect1d(x_intersects,y_intersects)
        
        # WE GET A SLICE HERE, THAT IS WHY WE ADD + 1 TO RETURN
        x_intersects = np.where(np.logical_and(zones[1:,0] <= self.x, zones[1:,0] + z_x > self.x))
        y_intersects = np.where(np.logical_and(zones[1:,1] <= self.y, zones[1:,1] + z_y > self.y))
        
        if len(np.intersect1d(x_intersects,y_intersects)) == 0:
            return [0]
    
        return np.intersect1d(x_intersects,y_intersects) + 1
    
    def bounce(self, paddle=False, random_bounce = False):
        """Bounces the ball off a vertical surface.
           If paddle is set to True, changes y position to avoid multiple collision
           If random_bounce is set to True, changes direction slightly every bounce"""
        random_var = random.uniform(0,0.3)
        
        # Set height above paddle so ball won't get 'stuck' when bouncing
        if paddle == True:
            self.y = screenheight - 30
        
        # Change a ball velocity randomly by max of .3 every bounce
        # Speed remains constant, however
        if random_bounce:
            biggest_dir = np.where(abs(self.direction) == abs(self.direction).max())
            smallest_dir = np.where(abs(self.direction) == abs(self.direction).min())
            
            if self.direction[biggest_dir] > 0:
                self.direction[biggest_dir] -= random_var
            else:
                self.direction[biggest_dir] += random_var
            
            if self.direction[smallest_dir] > 0:
                self.direction[smallest_dir] += random_var
            else:
                self.direction[smallest_dir] -= random_var

        self.direction[1] *= -1

    def update(self):
        """ Update the position of the ball.
            Returns true if the ball has fallen below the screen"""
        
        self.x += self.speed * self.direction[0] 
        self.y += self.speed * self.direction[1]
 
        self.rect.x = self.x
        self.rect.y = self.y
 
        # Dooes ball bounce off the top of the screen?
        if self.y <= 0:
            self.bounce(False, True)
            self.y = 1
 
        # Does ball bounce off the left of the screen?
        if self.x <= 0:
            self.direction[0] = self.direction[0] * -1
            self.x = 1
 
        # Does ball bounce of the right side of the screen?
        if self.x > screenwidth - self.width:
            self.direction[0] = self.direction[0] * -1
            self.x = screenwidth - self.width - 1
 
        # Does ball ball fall off the bottom edge of the screen?
        if self.y > screenheight:
            return True
        else:
            return False
 
 
class Player(pygame.sprite.Sprite):
    def __init__(self):
        # Call the parent's constructor
        super().__init__()
 
        self.width = paddle_size
        self.height = 15
 
        self.image = pygame.Surface([self.width, self.height])
        
        # Color the player
        self.image.fill((white))

        # Get a rectangle object that shows where our image is
        self.rect = self.image.get_rect()
        
        self.screenheight = screenheight
        self.screenwidth = screenwidth
        
        # Set starting position of player
        self.pos = 100
        
        # Set move speed of player
        self.speed = paddle_step
        
        self.rect.x = self.pos
        self.rect.y = self.screenheight-self.height
        
    def move(self, direction):
        """
        Updates player position by either moving right, left or not moving
        """
        if direction == 0:
            self.pos += self.speed
        elif direction == 1:
            self.pos -= self.speed
        elif direction == 2:
            pass
        
        if self.pos > screenwidth - self.width:
            self.pos = screenwidth - self.width
        if self.pos < 0:
            self.pos = 0
        
        self.rect.x = self.pos
    
    
    def get_player_quadrant(self, zones, z_x_p):
        """
        Unused, gets quadrant(s) player exists in for NN input
        """
        x_intersects = np.where(np.logical_and(zones[0:,0] <= self.pos, zones[0:,0] + z_x_p > self.pos))
        x_intersects_end = np.where(np.logical_and(zones[0:,0] < self.pos + paddle_size, zones[0:,0] + z_x_p >= self.pos + paddle_size))
        
        populated_zones = np.append(x_intersects, x_intersects_end)
        
        return populated_zones
   
        

class breakOut():
    
    # Colors redefined because I'm scared
    black = (0,0,0)
    white = (255,255,255)
    grey = (125,125,125)
    
    def __init__(self):
        pygame.init()
        pygame.font.init()

        self.clock = pygame.time.Clock()
        self.rendering = False
        self.reset()
        self.fast = True
        self.reward_font = pygame.font.SysFont('Comic Sans MS', 30)    
        
        # Rewards changed from experiemnt to experiment
        # Currently set to control experiment (no move rewards)
        self.move_penalty = 0
        self.stop_reward = 0
        self.hit_reward = 1
        self.lose_penalty = -5
        self.win_reward = 10
        
        # Time to win, game ends after this number of frames
        self.time_to_win = 2500
        
    def reset(self):
        """
        Resets environment
        Should be called before first game as well
        Returns the current state
        """
        
        self.player = Player()
        
        # Program supports multiple balls
        self.all_balls = [Ball() for i in range(num_balls)]
        
        # Create pygame groups for collision and rendering
        self.balls = pygame.sprite.Group()
        self.players = pygame.sprite.Group()
        
        self.balls.add(self.all_balls)
        self.players.add(self.player)
        
        self.won = False
        self.done = False
                
        self.reward = 0
        self.score = 0
        self.time_played = 0
        
        # To count amount of moves and stops for output data
        self.moves = 0
        self.stops = 0
        
        # Very complicated way of not exactly normalizing data
        x_obs = np.array([i.x for i in self.all_balls])
        y_obs = np.array([i.y for i in self.all_balls])
        
        x_obs = x_obs/screenwidth
        y_obs = y_obs/screenheight
        
        player_postiion = self.player.pos/screenwidth

        observations = np.append(x_obs, y_obs)
        observations = np.append(observations, player_postiion)
        return observations
        
    
    def init_render(self):
        """
        Only called when first rendering
        """
        self.disp = pygame.display.set_mode([screenwidth,screenheight])
        pygame.display.set_caption('Breakout')
        self.background = pygame.Surface(self.disp.get_size())
        self.rendering = True
        
    def render(self):
        """
        For rendering environment if you wanna watch
        """
        if not self.rendering:
            self.init_render()
            
        # Limit fps
        if self.fast == True:
            self.clock.tick(frame_fast)

        else:
            self.clock.tick(frame_slow)
                        
        # Fill display with grey color
        self.disp.fill(self.grey)
        
        # Draw all sprites
        self.players.draw(self.disp)
        self.balls.draw(self.disp)
        
        pygame.display.flip()
        
    def step(self, action):
        """
        Steps the environment one 'frame'
        Returns state, reward, done and a dummy value for more compatability...
        ... with openAI gym environments
        """

        self.time_played += 1
        
        # Reset reward every frame
        self.reward = 0
        
        # Check if game is over and update ball positions
        if not self.done:
            for i, r in enumerate(self.all_balls):
                if r.update() == True:
                    del self.all_balls[i]
                    self.reward += self.lose_penalty
            
            # Step the player if they have any action
            
            # If action is a move action, induce penalty
            if action == 2:
                self.player.move(action)
                self.reward += self.stop_reward
                self.stops += 1
            
            # If not a move action, give reward
            if not action == 2:
                self.player.move(action)
                self.reward += self.move_penalty
                self.moves += 1
                
        # Check ball collision with player paddle
        for i in self.all_balls:
            if pygame.sprite.collide_rect(self.player, i):
                i.bounce(True, True)
                self.reward += 1
       
        
        # Game ends if all the balls are gone
        if len(self.all_balls) == 0:
            self.done = True
        
        # Or if the time has run out
        elif self.time_played >= self.time_to_win:
            self.reward += self.win_reward
            self.done = True
            self.won = True
        
        # Increment the score by the collected reward
        self.score += self.reward
        
        # Dummy value for compatability with gym envs
        dummy = False
        
        # Still complicated way of normalizing data
        x_obs = np.array([i.x for i in self.all_balls])
        y_obs = np.array([i.y for i in self.all_balls])
        
        x_obs = x_obs/screenwidth
        y_obs = y_obs/screenheight
        
        player_postiion = self.player.pos/screenwidth

        observations = np.append(x_obs, y_obs)
        observations = np.append(observations, player_postiion)
        return observations, self.reward, self.done, dummy
    
    def close(self):
        """
        Unused ;_;
        """
        pygame.quit()
