# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:18:32 2022

@author: lucac
"""
from world import World
from agents import Painting, RectangleBuilding, Car, Pedestrian, EgoVehicle, CirclePainting
from geometry import Point
import numpy as np
import random
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
    ACTION_SIZE = 3
    OBS_SIZE = 5
    
    def __init__(self, dt: float, width: float, 
                 height: float, 
                 reward_fn = None,
                 ppm: float = 8,
                 render = False,
                 testing=False,
                 discrete_actions = True,
                 window_name="CARLO",
                 seed=0,
                 obs_type='gcn',
                 adjacency_norm='ones',
                 reward_configuration=None):
        super().__init__(dt, width, height, ppm, window_name=window_name)
        self.scene_name = None
        self.traffic_controller = None
        self.routes = []
        self.reward_fn = self._test_reward_fn if reward_fn is None else reward_fn
        self._render = render
        
        self._discrete_actions = discrete_actions
        self.testing = testing
        self.reward_configuration = reward_configuration
        # print('*'*50)
        # print("Available commands: ")
        # print("\t- Left click: tick on click")
        # print("\t- Right click: display traffic status")
        # print("\t- WASD: control ego-vehicle")
        # print("\t- O: print observation for DRL")
        # print("\t- R: reset the environment")
        # print("\t- F: forward tick")
        # print('*'*50)
        self.acceleration = 0.0
        self.steering = 0.0
        self.adjacency_norm = adjacency_norm
        
        # Graph building parameters
        self.detection_radius = 40.0
        self.adjecency_threshold = 30.0
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        if self.testing:
            self.timeout = 400.0
        else:
            self.timeout = 400.0
            
        if obs_type == 'gcn' or obs_type == 'gcn_speed' or obs_type == 'gcn_speed_route':
            self.obs_type = obs_type
        else:
            raise Exception(f"Unexpected observation type: {obs_type}, expected 'gcn' or 'gcn_speed', or 'gcn_speed_route'")
        
    def draw_route(self, route, color='green'):
        for i in range(len(route)):
            self.add(Painting(Point(route[i, 0], route[i, 1]), Point(0.5, 0.5), color))
            
    def load_scene(self, scene_name):
        self.scene_name = scene_name

        if scene_name == "scene01":
            ####################
            # CREATE BUILDINGS #
            ####################
            self.speed_normalization_factor = 10.0
            p_s = 2 # pavement size
            road1_width = 6
            road2_width = 12
            self.speed_limit = 50 / 3.6
            
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
            
            # N_cars = self.rng.randint(0, 10)
            N_cars = self.rng.poisson(5.0 )
            
            # Define ego vehicle
            initial_conditions = [(3, 18, 25), 
                                  (2, 40, 60),
                                  (4, 16, 28)]
            ego_route, initial_waypoint, goal_waypoint = initial_conditions[self.rng.choice(range(len(initial_conditions)))]
            
            # Draw the goal point
            xg, yg, h = self.get_transform(ego_route, goal_waypoint)
            goal = CirclePainting(Point(xg, yg), 1.0, color='pink')
            self.add(goal)
            
            x, y, heading = self.get_transform(ego_route, initial_waypoint)
            v0 = self.rng.uniform(0, 5.0)
            ego_vehicle = EgoVehicle(   Point(x, y), 
                                        heading, 
                                        color='blue', 
                                        velocity=Point(v0*np.cos(heading), 
                                                        v0*np.sin(heading)), 
                                        # velocity=Point(0,0), 
                                        ego_route_index=ego_route,
                                        initial_waypoint=initial_waypoint,
                                        goal_waypoint=goal_waypoint)
            self.traffic_controller = TrafficController(self, ego_vehicle,  
                                                        rng=self.rng,
                                                        N_cars=N_cars)
    
            # Pedestrian is almost the same as Car. It is a "circle" object rather than a rectangle.
            # self.p1 = Pedestrian(Point(28,81), np.pi)
            # self.p1.max_speed = 10.0 # We can specify min_speed and max_speed of a Pedestrian (and of a Car). This is 10 m/s, almost Usain Bolt.
            # self.add(self.p1)
    
            # self.p2 = Pedestrian(Point(30, 90), np.pi/2)
            # self.p2.max_speed = 5.0
            # self.add(self.p2)
            return N_cars
           
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
        self.traffic_controller.ego_vehicle.set_control(self.steering, self.acceleration)
        self.traffic_controller.tick()
        super().tick()
        
    def tick_on_click(self, event):
        self.tick()
        
    def print_traffic(self, event):
        print(self.traffic_controller.traffic[['id', 'route', 'waypoint', 'front_vehicle_id']])
            
    def key_press(self, event):
        if event.char == "w":
            if self.traffic_controller.ego_vehicle.speed < 10.0:
                self.acceleration = 1.0
            else:
                self.acceleration = 0.0
        elif event.char == "s":
            if self.traffic_controller.ego_vehicle.speed > 0:
                self.acceleration = -4.0
            else:
                self.acceleration = 0.0
        elif event.char == "a":
            self.steering = 1.0
        elif event.char == "d":
            self.steering = -1.0
        elif event.char == "r":
            self.reset()
        elif event.char == "o":
            self.print_observation()
        elif event.char == "f":
            self.tick()
        else:
            pass
        
    def key_release(self, event):
        if event.char == "w":
            self.acceleration = 0.0
        elif event.char == "s":
           self.acceleration = 0.0
        elif event.char == "a":
            self.steering = 0.0
        elif event.char == "d":
            self.steering = 0.0
        else:
            pass
        
    def _get_observation(self):
        ego_vehicle = self.traffic_controller.ego_vehicle
        p_ego = np.array([ego_vehicle.x, ego_vehicle.y])
        
        nodes_list = []
        edge_list = []
        edge_weight = []
        
        nearby_agents = []
        
        self.closest_vehicle_distance = self.detection_radius
        
        self.graph_plot = {}
        self.graph_plot['nodes'] = []
        self.graph_plot['edges'] = []
        
        for a in self.dynamic_agents:
            p_a = np.array([a.x, a.y])
            d_a_ego = np.linalg.norm(p_a - p_ego)
            if d_a_ego < self.detection_radius and a.collidable:
                if a is not ego_vehicle:
                    self.closest_vehicle_distance = d_a_ego
                nearby_agents.append(a)
                
        for i, a in enumerate(nearby_agents):
            p_a = np.array([a.x, a.y])
            self.graph_plot['nodes'].append(CirclePainting(Point(a.x, a.y), radius=0.5, color='green'))
            
            # Create nodes
            if isinstance(a, Car):
                vx = a.speed * np.cos(a.heading) / self.speed_normalization_factor
                vy = a.speed * np.sin(a.heading) / self.speed_normalization_factor
                node = [#1.0, 
                        #0.0,
                        a.x/self.width_m, 
                        a.y/self.height_m,
                        # a.heading/(2*np.pi),
                        # a.speed/self.speed_normalization_factor,
                        vx,
                        vy,
                        1.0, # distance to closest agent
                        ]
            else:
                node = [#0.0, 
                        #1.0,
                        a.x/self.width_m, 
                        a.y/self.height_m,
                        a.heading/(2*np.pi),
                        a.speed/self.speed_normalization_factor,
                        1.0, # distance to closest agent
                        ]
                    
            # Create edges and find closest vehicle
            for j, b in enumerate(nearby_agents):
                min_d = np.inf
                if (i != j):
                    p_b = np.array([b.x, b.y])
                    d_ab = np.linalg.norm(p_a - p_b)
                    
                    # Connect the vehicles if their distance is below adjacency threshold
                    if d_ab < self.adjecency_threshold:
                        edge_list.append([i, j])
                        
                        # center = (p_b + p_a) / 2.0
                        # center = Point(center[0], center[1])
                        # size = Point(np.linalg.norm(p_b - p_a), 0.02)
                        # y = p_b[1] - p_a[1]
                        # x = p_b[0] - p_a[0]
                        # heading = np.arctan2(y, x)
                        # self.graph_plot['edges'].append(Painting(center, size, color='green', heading=heading))
                        
                        if self.adjacency_norm.lower() == 'l2':
                            edge_weight.append(1.0/d_ab)
                        elif self.adjacency_norm.lower() == "ones":
                            edge_weight.append(1.0)
                        else:
                            raise Exception(f"Unexpected adjacency matrix norm {self.adjacency_norm}")
                        
                    # Find the closest vehicle
                    if d_ab < min_d:
                        node[4] = d_ab / self.max_d
                        min_d = d_ab
                        
            nodes_list.append(node)
            
        nodes = torch.tensor(nodes_list, dtype=torch.float)
        edges = torch.tensor(edge_list, dtype=torch.long).t().contiguous().view(2, -1)
        
        
        if self.adjacency_norm.lower() == 'l2':
            edge_weight = torch.tensor(edge_weight, dtype=torch.long).t().contiguous().view(-1, 1)
            self.graph = Data(x=nodes, edge_index=edges, edge_weight=edge_weight)
        elif self.adjacency_norm.lower() == "ones":
            self.graph = Data(x=nodes, edge_index=edges)
        else:
            raise Exception(f"Unexpected adjacency matrix norm {self.adjacency_norm}")
        

        if self.obs_type == 'gcn':
            obs = self.graph
        elif self.obs_type == 'gcn_speed':
            obs = (self.graph, ego_vehicle.speed)
        elif self.obs_type == 'gcn_speed_route':
            wp = self.traffic_controller.ego_controller.current_waypoint_index
            route = self.routes[ego_vehicle.ego_route_index][wp:wp+10]
            obs = (self.graph, ego_vehicle.speed, route)
        else:
            raise Exception('Unexpected observation type')
        
        return obs
    
    def _action_discrete_to_continuous(self, action):
        if action == Scene.ACTION_ACCELERATE:
            steer = 0.0
            accelerate = 3.0
        elif action == Scene.ACTION_BRAKE:
            steer = 0.0
            accelerate = -3.0
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
        
        if self.t > self.timeout:
            done = True
            reward = self.reward_configuration['timeout']
            info['end_reason'] = 'timeout'
        elif goal_reached:
            done = True
            reward = self.reward_configuration['goal_reached']
            info['end_reason'] = 'goal_reached'
        elif collision:
            done = True
            reward = self.reward_configuration['collision']
            info['end_reason'] = 'collision'
        else:
            done = False
            
            v = self.ego_vehicle.speed
            if v < 0.8 * self.speed_limit:
                r_velocity = 1.25 * (v / self.speed_limit)
            elif v >= 0.8 * self.speed_limit and v < self.speed_limit:
                r_velocity = 1.0
            else:
                r_velocity = 6.0 - 5.0 * (v / self.speed_limit)
            
            r_idle = -1.0 if (v < 5.0/3.6) else 0.0
            r_action = -np.abs(self.acceleration)
            r_proximity = -1.0 + self.closest_vehicle_distance/self.detection_radius
            
            r_velocity *= self.reward_configuration['velocity']
            r_action *= self.reward_configuration['action']
            r_idle *= self.reward_configuration['idle']
            r_proximity *= self.reward_configuration['proximity']
            
            # print(f"Reward: v {r_velocity:.4f}, a {r_action:.4f}, idle: {r_idle:.4f}, proximity: {r_proximity:.4f}")
            
            reward = r_velocity + r_action + r_idle + r_proximity
            
        return reward, done, info
      
    def reset(self, seed=None):
        self.t = 0
        self.dynamic_agents = []
        self.static_agents = []
        N_cars = self.load_scene(self.scene_name)
        self.episode_reward = 0.0
        return self._get_observation(), {'N_cars': N_cars}
        
    def step(self, action):
        for x in self.graph_plot['nodes']:
            self.add(x)
        for x in self.graph_plot['edges']:
            self.add(x)
        self.render()
        for x in self.graph_plot['nodes']:
            self.pop(x)
        for x in self.graph_plot['edges']:
            self.pop(x)
        
        
        if self._discrete_actions:
            _, self.acceleration = self._action_discrete_to_continuous(action)
            self.steering, _, _ = self.traffic_controller.ego_controller.stanely_controller()
        else:
            self.steering, self.acceleration = action
            
        self.tick()
        
        obs = self._get_observation()
        reward, done, info = self.reward_fn()
        
        self.episode_reward += reward
        return obs, reward, done, False, info
    
    def get_transform(self, route_index, point=0):
        route = self.routes[route_index]
        x, y = route[point]
        
        forward_vector = route[point+1] - route[point]
        heading = np.arctan2(forward_vector[1], forward_vector[0]) % (2*np.pi)
        return x, y, heading
    
    def reset_rng(self, seed=None):
        if self.testing:
            if seed is None:
                self.rng = np.random.RandomState(self.seed)
            else:
                self.rng = np.random.RandomState(seed)
        
        
        
    
    
    
