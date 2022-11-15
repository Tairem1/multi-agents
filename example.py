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

human_controller = False

dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
world = Scenes(dt, width = 120, height = 120, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

world.load_scene("scene01")
world.render()

controller1 = CarController(world.routes[2], 5.0, world.c1)

if not human_controller:
    # Let's implement some simple scenario with all agents
    world.p1.set_control(0, 0.22) # The pedestrian will have 0 steering and 0.22 throttle. So it will not change its direction.
    # world.c1.set_control(0, 0.35)
    world.c2.set_control(0, 0.05)
    world.p2.set_control(0, 0.22)
    for k in range(200):
        
        if world.c1 in world.dynamic_agents:
            steering, acc, end_of_route = controller1.stanely_controller()
            if not end_of_route:
                world.c1.set_control(steering, acc)
            else:
                world.pop(world.c1)
            
        # All movable objects will keep their control the same as long as we don't change it.
        if k == 100: # Let's say the first Car will release throttle (and start slowing down due to friction)
            # world.c1.set_control(0, 0)
            pass
        elif k == 200: # The first Car starts pushing the brake a little bit. The second Car starts turning right with some throttle.
            # world.c1.set_control(0, -0.02)
            pass
        elif k == 325:
            # world.c1.set_control(0, 0.8)
            world.c2.set_control(-0.45, 0.3)
        elif k == 367: # The second Car stops turning.
            world.c2.set_control(0, 0.1)
        world.tick() # This ticks the world for one time step (dt second)
        world.render()
        time.sleep(dt/4) # Let's watch it 4x

        if world.collision_exists(world.p1): # We can check if the Pedestrian is currently involved in a collision. We could also check c1 or c2.
            print('Pedestrian has died!')
        # elif world.collision_exists(): # Or we can check if there is any collision at all.
        #     print('Collision exists somewhere...')
    world.close()
