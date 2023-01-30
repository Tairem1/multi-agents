# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:18:32 2022

@author: lucac
"""
from world import World
from agents import Painting, RectangleBuilding, Car, Pedestrian
from geometry import Point
import numpy as np
from traffic_controller import TrafficController

class Scenes(World):
    def __init__(self, dt: float, width: float, height: float, ppm: float = 8):
        super().__init__(dt, width, height, ppm)
        self.scene_name = None
        self.traffic_controller = None
        self.routes = []
        
        print('*'*50)
        print("Available commands: ")
        print("\t- Right click: display traffic status")
        print('*'*50)
        
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
            
            #####################
            # DEFINE CAR ROUTES #
            #####################
            N_points = 100
            r1 = np.linspace([0, b3_y + sy/2.0 + road2_width/4.0], 
                             [self.width_m, b3_y + sy/2.0 + road2_width/4.0], 
                             N_points)
            r2 = np.linspace([self.width_m, b3_y + sy/2.0 + 3.0*road2_width/4.0], 
                             [0, b3_y + sy/2.0 + 3.0*road2_width/4.0], 
                             N_points)
            r3 = np.linspace([b3_x + sx/2.0 + road1_width/2.0, 0], 
                             [b3_x + sx/2.0 + road1_width/2.0, self.height_m], 
                             N_points)
            
            
            # Create route 4 connecting route 3 to 1
            r31 = 5.0
            cx = b3_x + sx/2.0 + road1_width/2.0 + r31
            cy = b3_y + sy/2.0 + road2_width/4.0 - r31
            l1 = np.linspace([b3_x + sx/2.0 + road1_width/2.0, 0],
                             [b3_x + sx/2.0 + road1_width/2.0, cy],
                             20)                             
            l2 = np.linspace([cx, b3_y + sy/2.0 + road2_width/4.0],
                             [self.width_m, b3_y + sy/2.0 + road2_width/4.0],
                             20)
            c1 = np.array([[cx - r31*np.cos(t), cy + r31*np.sin(t)] for t in np.linspace(np.pi/8, 0.45*np.pi, 5)])
            r4 = np.array([*l1, *c1, *l2])
            
            # Create route 5 connecting route 1 and 3
            r13 = 5.0
            cx = b3_x + sx/2.0 + road1_width/2.0 - r31
            cy = b3_y + sy/2.0 + road2_width/4.0 + r31
            l1 = np.linspace([0, b3_y + sy/2.0 + road2_width/4.0],
                             [cx, b3_y + sy/2.0 + road2_width/4.0],
                             20)                             
            l2 = np.linspace([b3_x + sx/2.0 + road1_width/2.0, cy], 
                             [b3_x + sx/2.0 + road1_width/2.0, self.height_m], 
                             20)
            c1 = np.array([[cx + r31*np.sin(t), cy - r31*np.cos(t)] for t in np.linspace(np.pi/8, 0.45*np.pi, 5)])
            r5 = np.array([*l1, *c1, *l2])
            
            
            
            # Create route 5 connecting route 1 and 3
            r13 = 5.0
            cx = b3_x + sx/2.0 + road1_width/2.0 - r31
            cy = b3_y + sy/2.0 + road2_width/4.0 + r31
            l1 = np.linspace([0, b3_y + sy/2.0 + road2_width/4.0],
                             [cx, b3_y + sy/2.0 + road2_width/4.0],
                             20)                             
            l2 = np.linspace([b3_x + sx/2.0 + road1_width/2.0, cy], 
                             [b3_x + sx/2.0 + road1_width/2.0, self.height_m], 
                             20)
            c1 = np.array([[cx + r31*np.sin(t), cy - r31*np.cos(t)] for t in np.linspace(np.pi/8, 0.45*np.pi, 5)])
            r5 = np.array([*l1, *c1, *l2])

            
            self.routes.append(r1)
            self.routes.append(r2)
            self.routes.append(r3)
            self.routes.append(r4)
            self.routes.append(r5)
            self.non_ego_routes = [0, 1]
            
            self.draw_route(self.routes[0], 'green')
            self.draw_route(self.routes[1], 'white')
            self.draw_route(self.routes[2], 'red')
            self.draw_route(self.routes[3], 'orange')
            self.draw_route(self.routes[4], 'purple')
            
            self.traffic_controller = TrafficController(self, N_cars=10)
    
            # Pedestrian is almost the same as Car. It is a "circle" object rather than a rectangle.
            # self.p1 = Pedestrian(Point(28,81), np.pi)
            # self.p1.max_speed = 10.0 # We can specify min_speed and max_speed of a Pedestrian (and of a Car). This is 10 m/s, almost Usain Bolt.
            # self.add(self.p1)
    
            # self.p2 = Pedestrian(Point(30, 90), np.pi/2)
            # self.p2.max_speed = 5.0
            # self.add(self.p2)
            
            
    def close_scene(self):
        if self.scene_name == "scene01":
            pass
        
    def tick(self):
        self.traffic_controller.tick()
        super().tick()
        
    def tick_on_click(self, event):
        self.tick()
        
    def print_traffic(self, event):
        print(self.traffic_controller.traffic[['id', 'route', 'waypoint', 'front_vehicle_id']])
        
    def render(self):
        super().render()
        self.visualizer.win.bind("<Button-3>", self.print_traffic)
        self.visualizer.win.bind("<Button-1>", self.tick_on_click)
        
    
