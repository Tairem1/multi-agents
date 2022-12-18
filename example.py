# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:25:58 2022

@author: lucac
"""

import numpy as np
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time
from scenes import Scenes
from traffic_controller import CarController

import pygame

pygame.init()
human_controller = False

dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
world = Scenes(dt, width = 120, height = 120, ppm = 5) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

world.load_scene("scene01")
# world.render()

if not human_controller:
    # Let's implement some simple scenario with all agents
    start = time.time()
    while True:
        world.tick() # This ticks the world for one time step (dt second)
        world.render()
        time.sleep(dt/4) # Let's watch it 4x
        
        # if world.collision_exists(): # Or we can check if there is any collision at all.
        #     print('Collision exists somewhere...')
    world.close()
