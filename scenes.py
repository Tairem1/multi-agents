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

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import torch_geometric
import networkx as nx

from traffic_controller import CarController

from util.timer import Timer

t = Timer()


class Scene(World):
    ACTION_BRAKE = 0
    ACTION_NONE = 1
    ACTION_ACCELERATE = 2
    
    def __init__(self, dt: float, width: float, 
                 height: float, 
                 reward_fn = None,
                 ppm: float = 8,
                 render = False,
                 discrete_actions = True):
        super().__init__(dt, width, height, ppm)
        self.scene_name = None
        self.traffic_controller = None
        self.routes = []
        self.reward_fn = self._test_reward_fn if reward_fn is None else reward_fn
        self._render = render
        
        self._discrete_actions = discrete_actions
        
        print('*'*50)
        print("Available commands: ")
        print("\t- Left click: tick on click")
        print("\t- Right click: display traffic status")
        print("\t- WASD: control ego-vehicle")
        print("\t- O: print observation for DRL")
        print("\t- R: reset the environment")
        print("\t- F: forward tick")
        print('*'*50)
        self.a = 0.0
        self.s = 0.0
        
        # Graph building parameters
        self.detection_radius = 30.0
        self.adjecency_threshold = 15.0
        
    def draw_route(self, route, color='green'):
        for i in range(len(route)):
            self.add(Painting(Point(route[i, 0], route[i, 1]), Point(0.5, 0.5), color))
            
    def load_scene(self, scene_name):
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
            
            N_cars = np.random.randint(0, 11)
            self.traffic_controller = TrafficController(self, N_cars=N_cars)
    
            # Pedestrian is almost the same as Car. It is a "circle" object rather than a rectangle.
            # self.p1 = Pedestrian(Point(28,81), np.pi)
            # self.p1.max_speed = 10.0 # We can specify min_speed and max_speed of a Pedestrian (and of a Car). This is 10 m/s, almost Usain Bolt.
            # self.add(self.p1)
    
            # self.p2 = Pedestrian(Point(30, 90), np.pi/2)
            # self.p2.max_speed = 5.0
            # self.add(self.p2)
           
    def render(self):
        if self._render:
            super().render()
            self.visualizer.win.bind("<Button-3>", self.print_traffic)
            self.visualizer.win.bind("<Button-1>", self.tick_on_click)
            self.visualizer.win.bind("<KeyPress>", self.key_press)
            self.visualizer.win.bind("<KeyRelease>", self.key_release)
            self.visualizer.win.focus_set()
            
    def close_scene(self):
        if self.scene_name == "scene01":
            pass
        
    def tick(self):
        self.traffic_controller.ego_vehicle.set_control(self.s, self.a)
        self.traffic_controller.tick()
        super().tick()
        
    def tick_on_click(self, event):
        self.tick()
        
    def print_traffic(self, event):
        print(self.traffic_controller.traffic[['id', 'route', 'waypoint', 'front_vehicle_id']])
            
    def key_press(self, event):
        if event.char == "w":
            if self.traffic_controller.ego_vehicle.speed < 10.0:
                self.a = 1.0
            else:
                self.a = 0.0
        elif event.char == "s":
            if self.traffic_controller.ego_vehicle.speed > 0:
                self.a = -4.0
            else:
                self.a = 0.0
        elif event.char == "a":
            self.s = 1.0
        elif event.char == "d":
            self.s = -1.0
        elif event.char == "r":
            self.reset()
        elif event.char == "o":
            self.print_observation()
        elif event.char == "f":
            self.tick()
        else:
            pass
        
    def print_observation(self):
        obs = self._get_observation()
        print(f"Nodes: {obs.x}")
        print(f"Edges: {obs.edge_index.t()}")
        print(f"A: {to_dense_adj(obs.edge_index)}")
        
        
    def key_release(self, event):
        if event.char == "w":
            self.a = 0.0
        elif event.char == "s":
           self.a = 0.0
        elif event.char == "a":
            self.s = 0.0
        elif event.char == "d":
            self.s = 0.0
        else:
            pass
        
    def _get_observation(self):
        ego_vehicle = self.traffic_controller.ego_vehicle
        p_ego = np.array([ego_vehicle.x, ego_vehicle.y])
        
        nodes_list = []
        edge_list = []
        
        nearby_agents = []
        
        for a in self.dynamic_agents:
            p_a = np.array([a.x, a.y])
            if np.linalg.norm(p_a - p_ego) < self.detection_radius:
                nearby_agents.append(a)
                
        for i, a in enumerate(nearby_agents):
            p_a = np.array([a.x, a.y])
            if isinstance(a, Car):
                node = [1.0, 
                        0.0,
                        a.x - ego_vehicle.x, 
                        a.y - ego_vehicle.y
                        ]
            else:
                node = [0.0, 
                        1.0,
                        a.x - ego_vehicle.x,
                        a.y - ego_vehicle.y
                        ]
            nodes_list.append(node)
                    
            for j, b in enumerate(nearby_agents):
                if (i != j):
                    p_b = np.array([b.x, b.y])
                    d_ab = np.linalg.norm(p_a - p_b)
                    if d_ab < self.adjecency_threshold:
                        edge_list.append([i, j])
        nodes = torch.tensor(nodes_list, dtype=torch.float)
        edges = torch.tensor(edge_list, dtype=torch.long).t().contiguous().view(2, -1)
        
        obs = Data(x=nodes, edge_index=edges)
        return obs
    
    def _action_discrete_to_continuous(self, action):
        if action == Scene.ACTION_ACCELERATE:
            steer = 0.0
            accelerate = 0.5
        elif action == Scene.ACTION_BRAKE:
            steer = 0.0
            accelerate = -0.5
        elif action == Scene.ACTION_NONE:
            steer = 0.0
            accelerate = 0.0
        else:
            raise Exception("Unexpected action: {action}")
        return steer, accelerate
    
    @property
    def ego_vehicle(self):
        return self.traffic_controller.ego_vehicle
    
    def _goal_reached(self):
        return self.traffic_controller.ego_controller.goal_reached
    
    def _test_reward_fn(self):
        info = {'end_reason': None}
        
        goal_reached = self._goal_reached()
        collision = self.collision_exists(self.ego_vehicle)
        
        if self.t > 40.0:
            done = True
            reward = -0.25
            info['end_reason'] = 'timeout'
        elif goal_reached:
            done = True
            reward = 1.0
            info['end_reason'] = 'goal_reached'
        elif collision:
            done = True
            reward = -1.0
            info['end_reason'] = 'collision_found'
        else:
            reward = -self.dt/10.0
            done = False
            info['end_reason'] = None
        return reward, done, info
      
    def reset(self, seed=None):
        self.t = 0
        self.dynamic_agents = []
        self.static_agents = []
        self.load_scene(self.scene_name)
        self.episode_reward = 0.0
        return self._get_observation(), {}
        
    def step(self, action):
        self.render()
        
        if self._discrete_actions:
            _, self.a = self._action_discrete_to_continuous(action)
            self.s, _, _ = self.traffic_controller.ego_controller.stanely_controller()
        else:
            self.s, self.a = action
            
        self.tick()
        
        reward, done, info = self.reward_fn()
        obs = self._get_observation()
        
        self.episode_reward += reward
        return obs, reward, done, False, info
        
        
        
    
    
    
