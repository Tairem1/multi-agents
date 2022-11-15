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
        
    def load_scene(self, scene_name):
        self.close()
        
        if self.scene_name is not None:
            self.close_scene()
        self.scene_name = scene_name
        
        if scene_name == "scene01":
            # Let's add some sidewalks and RectangleBuildings.
            # A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks.
            # A RectangleBuilding object is also static -- it does not move. But as opposed to Painting, it can be collided with.
            # For both of these objects, we give the center point and the size.
            self.add(Painting(Point(71.5, 106.5), Point(97, 27), 'gray80')) # We build a sidewalk.
            self.add(RectangleBuilding(Point(72.5, 107.5), Point(95, 25))) # The RectangleBuilding is then on top of the sidewalk, with some margin.
    
            # Let's repeat this for 4 different RectangleBuildings.
            self.add(Painting(Point(8.5, 106.5), Point(17, 27), 'gray80'))
            self.add(RectangleBuilding(Point(7.5, 107.5), Point(15, 25)))
    
            self.add(Painting(Point(8.5, 41), Point(17, 82), 'gray80'))
            self.add(RectangleBuilding(Point(7.5, 40), Point(15, 80)))
    
            self.add(Painting(Point(71.5, 41), Point(97, 82), 'gray80'))
            self.add(RectangleBuilding(Point(72.5, 40), Point(95, 80)))
    
            # Let's also add some zebra crossings, because why not.
            self.add(Painting(Point(18, 81), Point(0.5, 2), 'white'))
            self.add(Painting(Point(19, 81), Point(0.5, 2), 'white'))
            self.add(Painting(Point(20, 81), Point(0.5, 2), 'white'))
            self.add(Painting(Point(21, 81), Point(0.5, 2), 'white'))
            self.add(Painting(Point(22, 81), Point(0.5, 2), 'white'))
            
            # A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
            self.c1 = Car(Point(20,20), np.pi/2)
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

        
    
