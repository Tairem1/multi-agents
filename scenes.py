# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:18:32 2022

@author: lucac
"""
from world import World
from agents import Painting, RectangleBuilding, Car, Pedestrian
from geometry import Point
import numpy as np

class Scenes(World):
    def __init__(self, dt: float, width: float, height: float, ppm: float = 8):
        super().__init__(dt, width, height, ppm)
        self.scene_name = None
        self.routes = []
        
    def draw_route(self, route, color='green'):
        for i in range(len(route)):
            self.add(Painting(Point(route[i, 0], route[i, 1]), Point(0.5, 0.5), color))
        
    def load_scene(self, scene_name):
        self.close()
        
        if self.scene_name is not None:
            self.close_scene()
        self.scene_name = scene_name
        
        if scene_name == "scene01":
            ####################
            # CREATE BUILDINGS #
            ####################
            p_s = 2 # pavement size
            road1_width = 6
            road2_width = 12
            
            sx, sy = ((self.width_m - road1_width)/2, 
                      (self.height_m - road2_width)/2)
            b1_x, b1_y = ((self.width_m - road1_width)/4, 
                          (self.height_m - sy/2.0))
            bsx = sx - 4.0
            bsy = sy - 4.0
            self.add(Painting(Point(b1_x, b1_y), Point(sx, sy), 'gray80')) # We build a sidewalk.
            self.add(RectangleBuilding(Point(b1_x, b1_y), Point(bsx, bsy))) # The RectangleBuilding is then on top of the sidewalk, with some margin.
            
            b3_x, b3_y = ((self.width_m - road1_width)/4, 
                          (self.height_m - road2_width)/4)
            self.add(Painting(Point(b3_x, b3_y), Point(sx, sy), 'gray80'))
            self.add(RectangleBuilding(Point(b3_x, b3_y), Point(bsx, bsy)))
    
            b4_x, b4_y = ((self.width_m - sx/2.0), 
                          (self.height_m - road2_width)/4)
            self.add(Painting(Point(b4_x, b4_y), Point(sx, sy), 'gray80'))
            self.add(RectangleBuilding(Point(b4_x, b4_y), Point(bsx, bsy)))
            
            # Let's repeat this for 4 different RectangleBuildings.
            b2_x, b2_y = b4_x, b1_y
            self.add(Painting(Point(b2_x, b2_y), Point(sx, sy), 'gray80'))
            self.add(RectangleBuilding(Point(b2_x, b2_y), Point(bsx, bsy)))
    
            # # Let's also add some zebra crossings, because why not.
            # self.add(Painting(Point(18, 81), Point(0.5, 2), 'white'))
            # self.add(Painting(Point(19, 81), Point(0.5, 2), 'white'))
            # self.add(Painting(Point(20, 81), Point(0.5, 2), 'white'))
            # self.add(Painting(Point(21, 81), Point(0.5, 2), 'white'))
            # self.add(Painting(Point(22, 81), Point(0.5, 2), 'white'))
            
            #####################
            # DEFINE CAR ROUTES #
            #####################
            N_points = 100
            r1 = np.linspace([self.width_m, b3_y + sy/2.0 + road2_width/4.0], 
                             [0, b3_y + sy/2.0 + road2_width/4.0], 
                             N_points)
            r2 = np.linspace([self.width_m, b3_y + sy/2.0 + 3.0*road2_width/4.0], 
                             [0, b3_y + sy/2.0 + 3.0*road2_width/4.0], 
                             N_points)
            r3 = np.linspace([b3_x + sx/2.0 + road1_width/2.0, 0], 
                             [b3_x + sx/2.0 + road1_width/2.0, self.height_m], 
                             N_points)
            self.routes.append(r1)
            self.routes.append(r2)
            self.routes.append(r3)
            
            self.draw_route(self.routes[0], 'green')
            self.draw_route(self.routes[1], 'white')
            self.draw_route(self.routes[2], 'red')
            
            
            # A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
            self.c1 = Car(Point(20,20), np.pi/2)
            self.c1.velocity = Point(0.0, 2.0)
            self.add(self.c1)
            
    
            self.c2 = Car(Point(118,90), np.pi, 'blue')
            self.c2.velocity = Point(3.0,0) # We can also specify an initial velocity just like this.
            self.add(self.c2)
    
            # Pedestrian is almost the same as Car. It is a "circle" object rather than a rectangle.
            self.p1 = Pedestrian(Point(28,81), np.pi)
            self.p1.max_speed = 10.0 # We can specify min_speed and max_speed of a Pedestrian (and of a Car). This is 10 m/s, almost Usain Bolt.
            self.add(self.p1)
    
            self.p2 = Pedestrian(Point(30, 90), np.pi/2)
            self.p2.max_speed = 5.0
            self.add(self.p2)
            
            
    def close_scene(self):
        if self.scene_name == "scene01":
            pass

        
    
