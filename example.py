# -*- coding: utf-8 -*-
"""oo
Created on Fri Nov 11 11:25:58 2022

@author: lucac
"""
# import time
# from scenes import Scene

# dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
# world = Scene(dt, width = 120, height = 120, ppm = 5,
#               render=True) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

# world.load_scene("scene01")


# start = time.time()
# while True:
#     world.tick() # This ticks the world for one time step (dt second)
#     world.render()
#     # time.sleep(dt/4) # Let's watch it 4x
    
#     # if world.collision_exists(): # Or we can check if there is any collision at all.
#     #     print('Collision exists somewhere...')
# world.close()
import numpy as np
import matplotlib.pyplot as plt


def plot_comb(lam, pu):
    s = np.random.poisson(lam, 10_000)
    u = np.random.choice(np.arange(0, 6, dtype=np.int32))
    count, bins, ignored = plt.hist(s, 20, density=True)
    plt.show()
    

plot_comb(5.0, )