#!/usr/bin/env python2
"""
Neural q-learning demonstration with a simple game.

"""
# Dependencies
from __future__ import division
import os; os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
from numpy import array, clip, sin, cos, arctan2, pi as PI
from numpy.random import uniform
import pygame
#import kalmann

################################################## GAME

class PaddleBoi(object):
    """
    |P A D D L E B O I|
    Slide your paddleboi left / right with the arrow keys (or 'A' / 'D').
    A wild ball bounces around. Help it hit green zones to improve your life.

    """
    #### GLOBALS
    # Hashmap from valid decisions to corresponding action values
    CHOICES = {"left": -1, "stall": 0, "right": 1}
    ####

    def __init__(self, width=500, height=500,
                 paddle_size=60, paddle_speed=5,
                 ball_size=15, ball_speed=5):
        """
        PaddleBoi constructor.

        width: extent of the domain's x-dimension (pix)
        height: extent of the domain's y-dimension (pix)
        paddle_size: extent of paddleboi's +x-aligned hitbox (pix)
        paddle_speed: max distance covered by paddleboi per update (pix/step)
        ball_size: radius of the ball's circular hitbox (pix)
        ball_speed: max distance covered by the ball per update (pix/step)

        """
        # Store provided parameters
        self.width = int(width)
        self.height = int(height)
        self.paddle_size = int(paddle_size)
        self.paddle_speed = int(paddle_speed)
        self.ball_size = int(ball_size)
        self.ball_speed = int(ball_speed)
        # Establish initial game data
        self.reset()

    def reset(self):
        """
        Sets game data to default initial conditions.

        """
        # Set paddle position to center of domain
        self.paddle_x = (self.width - self.paddle_size) // 2
        self.paddle_y = self.height // 2
        # Set random but valid ball position and direction
        self.ball_x = int(uniform(self.ball_size, self.width - self.ball_size))
        self.ball_y = int(uniform(self.ball_size, self.height - self.ball_size))
        self.ball_dx = cos(uniform(0.0, 2*PI))
        self.ball_dy = 1.0 - self.ball_dx**2
        # Set life-improvement score to minimum
        self.score = 0

    def get_reward(self):
        """
        Returns a float encoding the overall quality of your life.

        """
        return float(self.score)

    def get_state(self):
        """
        Returns an array of floats that encode your current situation.

        """
        return array((self.paddle_x,
                      self.ball_x,
                      self.ball_y,
                      arctan2(self.ball_dy, self.ball_dx)), dtype=float)

    def update(self, decision):
        """
        Advances the game forward by one step using the provided decision.

        decision: any valid member of CHOICES, it's up to you

        """
        # Turn symbolic decision into an action value, if valid
        try:
            action = PaddleBoi.CHOICES[decision]
        except KeyError:
            raise KeyError("Invalid decision provided to update PaddleBoi. "
                           +"Choose from: {0}.".format(PaddleBoi.CHOICES.keys()))
        # Update paddle position based on speed parameter and decided action
        paddle_velocity = action*self.paddle_speed
        self.paddle_x += paddle_velocity
        # Prevent paddle from leaving the domain
        self.paddle_x = clip(self.paddle_x, 0, self.width - self.paddle_size)
        # Update ball position
        self.ball_x = int(clip(self.ball_x + self.ball_speed*self.ball_dx, self.ball_size, self.width - self.ball_size))
        self.ball_y = int(clip(self.ball_y + self.ball_speed*self.ball_dy, self.ball_size, self.height - self.ball_size))
        if (self.ball_x <= self.ball_size) or (self.ball_x >= self.width - self.ball_size):
            self.ball_dx *= -1.0
        if (self.ball_y <= self.ball_size) or (self.ball_y >= self.height - self.ball_size):
            self.ball_dy *= -1.0
        # Impose consequences
        pass # ???

    def display(self, screen):
        """
        Draws the current situation on the provided screen.

        screen: a Pygame.Surface you are okay with completely drawing over

        """
        # Set screen title to game name and current score
        pygame.display.set_caption("PADDLE BOI - {0}".format(self.score))
        # Wipe all pixels to black
        screen.fill((0, 0, 0))
        # Select a color representative of current score
        color = (int(255*(2.0**(-self.score/20))), int(255*(1.0 - 2.0**(-self.score/20))), 0)
        # Draw the paddle
        paddle_xywh = (self.paddle_x, self.paddle_y-self.paddle_size//8,
                       self.paddle_size, self.paddle_size//4)
        pygame.draw.rect(screen, color, paddle_xywh, 0)
        # Draw the ball
        pygame.draw.circle(screen, color, (self.ball_x, self.ball_y), self.ball_size, 0)
        # Draw the consequence zones
        pass # ???
        # Render the image
        pygame.display.update()

    def play(self, policy=None, rate=60):
        """
        Runs the game using the given policy function and at the specified update rate.
        If a policy is not given, then decisions are read live from human input.

        policy: function that takes a state vector and returns a valid decision
        rate: max number of updates (and display frames) per second of real time

        """
        # Create visualization of domain
        screen = pygame.display.set_mode((self.width, self.height))
        # Enter main loop
        playing = True
        while playing:
            # Record current real time (in milliseconds)
            start_time = pygame.time.get_ticks()
            # Draw current situation to the screen
            self.display(screen)
            # Use policy or human input to make decision
            if policy:
                decision = policy(self.get_state())
            else:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    decision = "right"
                elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    decision = "left"
                else:
                    decision = "stall"
            # Evolve the situation by one step with the chosen decision
            self.update(decision)
            # Check for quit or reset command events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    playing = False
                    break
                elif (event.type == pygame.KEYDOWN) and (event.key == pygame.K_r):
                    self.reset()
            # Rest for amount of real time alloted per loop less what was used
            remaining_time = (1000//rate) - (pygame.time.get_ticks() - start_time)
            if remaining_time > 0:
                pygame.time.wait(remaining_time)

################################################## MAIN

if __name__ == "__main__":

    game = PaddleBoi()
    game.play()

##################################################
