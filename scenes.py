# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:18:32 2022

@author: lucac
"""
from world import World
from agents import Painting, RectangleBuilding, Car, Pedestrian, EgoVehicle, CirclePainting, EgoPedestrian
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
from policy import init_agent
from util.timer import Timer

t = Timer()


def prepare_state_for_nn(state):
    if isinstance(state, tuple):
        if len(state) == 2:
            s = (state[0], torch.tensor([[state[1]]]))
        elif len(state) == 3:
            s = (state[0], 
                 torch.tensor([[state[1]]], dtype=torch.float32),
                 torch.tensor([state[2]], dtype=torch.float32))
    else:
        raise Exception("Unsupported state type")
    return s

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
                 reward_configuration=None,
                 svo = 0, # expressed in degrees
                 agent='vehicle',
                 vehicle_level='L0',
                 pedestrian_level='L0',
                 hidden_features=0, 
                 gcn_conv_layer=0,
                 n_conv_layers=0,
                 path_to_vehicle_agent=None,
                 path_to_pedestrian_agent=None):
        super().__init__(dt, width, height, ppm, window_name=window_name)
        self.scene_name = None
        self.traffic_controller = None
        self.routes = []
        self.pedestrian_routes = []
        self.reward_fn = self._test_reward_fn if reward_fn is None else reward_fn
        self.svo = svo * np.pi/180.0
        self._render = render
        
        self._discrete_actions = discrete_actions
        self.testing = testing
        self.reward_configuration = reward_configuration
        
        self.acceleration_v = 0.0
        self.steering_v = 0.0
        self.acceleration_p = 0.0
        self.steering_p = 0.0
        
        self.adjacency_norm = adjacency_norm
        
        self.ped_ai_cross = -1
        
        # Graph building parameters
        self.detection_radius = 40.0
        self.adjecency_threshold = 30.0
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.agent = agent # Pedestrian or Vehicle
        
        if self.testing:
            self.timeout = 400.0
        else:
            self.timeout = 400.0
            
        if obs_type == 'gcn' or obs_type == 'gcn_speed' or obs_type == 'gcn_speed_route':
            self.obs_type = obs_type
        else:
            raise Exception(f"Unexpected observation type: {obs_type}, expected 'gcn' or 'gcn_speed', or 'gcn_speed_route'")
        
        if self.agent.lower() == 'pedestrian':
            self.vehicle_level = vehicle_level
            if vehicle_level != 'L0':
                # device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print("Loading vehicle agent...")
                self.vehicle_agent = init_agent(Scene.ACTION_SIZE, 
                                                Scene.OBS_SIZE, 
                                                hidden_features=hidden_features,
                                                obs_type=obs_type,
                                                gcn_conv_layer=gcn_conv_layer,
                                                n_conv_layers=n_conv_layers
                                                )#.to(device)
                self.vehicle_agent.eval()
                self.vehicle_agent.load_state_dict(torch.load(path_to_vehicle_agent))
                print("Loaded!")
        elif self.agent.lower() == 'vehicle':
            self.pedestrian_level = pedestrian_level
            if pedestrian_level != 'L0':
                # device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print("Loading pedestrian agent...")
                self.pedestrian_agent = init_agent(Scene.ACTION_SIZE, 
                                                Scene.OBS_SIZE, 
                                                hidden_features=hidden_features,
                                                obs_type=obs_type,
                                                gcn_conv_layer=gcn_conv_layer,
                                                n_conv_layers=n_conv_layers
                                                )#.to(device)
                self.pedestrian_agent.eval()
                self.pedestrian_agent.load_state_dict(torch.load(path_to_pedestrian_agent))
                print("Loaded!")
            
            
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
            self.ped_speed_limit = 2.0
            
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
    
            # Pedestrian is almost the same as Car. It is a "circle" object rather than a rectangle.
            
            # Draw pedestrian goal point
            N_points = 20
            initial_waypoint = 0
            goal_waypoint = N_points - 2
            N_extra = 15
            
            rp1 = np.linspace([b3_x + sx/2.0 - p_s/2.0, b3_y + sy/2.0 - p_s], 
                              [b4_x - sx/2.0 + p_s/2.0, b3_y + sy/2.0 - p_s], 
                              N_points)
            rp1 = np.vstack((rp1, np.tile(rp1[-1,:], (N_extra, 1))))
            rp_1 = np.linspace([b4_x - sx/2.0 + p_s/2.0, b3_y + sy/2.0 - p_s], 
                               [b3_x + sx/2.0 - p_s/2.0, b3_y + sy/2.0 - p_s],
                              N_points)
            rp_1 = np.vstack((rp_1, np.tile(rp_1[-1,:], (N_extra, 1))))
            
            rp3 = np.linspace([b3_x + sx/2.0 - p_s/2.0, b1_y - sy/2.0 + p_s/2.0], 
                              [b4_x - sx/2.0 + p_s/2.0, b1_y - sy/2.0 + p_s/2.0], 
                              N_points)
            rp3 = np.vstack((rp3, np.tile(rp3[-1,:], (N_extra, 1))))
            rp_3 = np.linspace([b4_x - sx/2.0 + p_s/2.0, b1_y - sy/2.0 + p_s/2.0], 
                               [b3_x + sx/2.0 - p_s/2.0, b1_y - sy/2.0 + p_s/2.0],
                              N_points)
            rp_3 = np.vstack((rp_3, np.tile(rp_3[-1,:], (N_extra, 1))))
            
            rp2 = np.linspace([b3_x + sx/2.0 - p_s/2.0, b3_y + sy/2.0 - p_s/2.0], 
                              [b3_x + sx/2.0 - p_s/2.0, b1_y - sy/2.0 + p_s], 
                              N_points)
            rp2 = np.vstack((rp2, np.tile(rp2[-1,:], (N_extra, 1))))
            rp_2 = np.linspace([b3_x + sx/2.0 - p_s/2.0, b1_y - sy/2.0 + p_s/2.0], 
                               [b3_x + sx/2.0 - p_s/2.0, b3_y + sy/2.0 - p_s], 
                              N_points)
            rp_2 = np.vstack((rp_2, np.tile(rp_2[-1,:], (N_extra, 1))))
            
            rp4 = np.linspace([b4_x - sx/2.0 + p_s/2.0, b3_y + sy/2.0 - p_s/2.0], 
                              [b4_x - sx/2.0 + p_s/2.0, b1_y - sy/2.0 + p_s], 
                              N_points)
            rp4 = np.vstack((rp4, np.tile(rp4[-1,:], (N_extra, 1))))
            rp_4 = np.linspace([b4_x - sx/2.0 + p_s/2.0, b1_y - sy/2.0 + p_s/2.0], 
                               [b4_x - sx/2.0 + p_s/2.0, b3_y + sy/2.0 - p_s], 
                              N_points)
            rp_4 = np.vstack((rp_4, np.tile(rp_4[-1,:], (N_extra, 1))))
            
            self.pedestrian_routes.append(rp1)
            self.pedestrian_routes.append(rp_1)
            self.pedestrian_routes.append(rp3)
            self.pedestrian_routes.append(rp_3)
            self.pedestrian_routes.append(rp2)
            self.pedestrian_routes.append(rp_2)
            self.pedestrian_routes.append(rp4)
            self.pedestrian_routes.append(rp_4)
            route_index = self.rng.randint(0, len(self.pedestrian_routes))

            xg, yg, h = self.get_transform(route_index, goal_waypoint, pedestrian=True)
            goal_p = CirclePainting(Point(xg, yg), 1.0, color='green')
            self.add(goal_p)
            
            xs, ys, h = self.get_transform(route_index, initial_waypoint,  pedestrian=True)
            start_p = CirclePainting(Point(xs, ys), 1.0, color='red')
            self.add(start_p)
            
            ego_pedestrian = EgoPedestrian(Point(xs, ys), h, velocity=Point(0.0, 0.0),
                                            ego_route_index=route_index, 
                                            initial_waypoint = initial_waypoint,
                                            goal_waypoint=goal_waypoint)
    
            self.traffic_controller = TrafficController(self, 
                                                        ego_vehicle,
                                                        ego_pedestrian,
                                                        rng=self.rng,
                                                        N_cars=N_cars)
            return N_cars
           
    def render(self):
        if self._render:
            super().render()
            self.visualizer.win.bind("<Button-1>", self.print_traffic)
            self.visualizer.win.focus_set()
        
    def tick(self):
        self.traffic_controller.ego_vehicle.set_control(self.steering_v, self.acceleration_v)
        self.traffic_controller.ego_pedestrian.set_control(self.steering_p, self.acceleration_p)
        self.traffic_controller.tick()
        super().tick()
        
    def _goal_reached(self):
        if self.agent.lower() == 'vehicle':
            return self._vehicle_goal_reached()
        elif self.agent.lower() == 'pedestrian':
            return self._pedestrian_goal_reached()
        
    def _vehicle_goal_reached(self):
        return self.traffic_controller.ego_controller.vehicle_goal_reached
   
    def _pedestrian_goal_reached(self):
        ego_p = self.traffic_controller.ego_pedestrian
        xg, yg, _ = self.get_transform(ego_p.ego_route_index, 
                                       ego_p.goal_waypoint, 
                                       pedestrian=True)
        p_ego = np.array([ego_p.x, ego_p.y])
        p_goal = np.array([xg, yg])
        
        if np.linalg.norm(p_ego - p_goal) < 0.3:
            return True
        else:
            return False
        
        
    def _get_observation(self, agent=None):
        ag = agent if agent is not None else self.agent.lower()
        if ag == 'vehicle':
            agent = self.traffic_controller.ego_vehicle
            routes = self.routes
            wp = self.traffic_controller.ego_vehicle_current_waypoint
        elif ag == 'pedestrian':
            agent = self.traffic_controller.ego_pedestrian
            routes = self.pedestrian_routes
            wp = self.traffic_controller.current_ped_waypoint
        else:
            raise Exception('Unexptected agent')
        
        p_ego = np.array([agent.x, agent.y])
        
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
                if a is not agent:
                    self.closest_vehicle_distance = d_a_ego
                nearby_agents.append(a)
                
        for i, a in enumerate(nearby_agents):
            p_a = np.array([a.x, a.y])
            self.graph_plot['nodes'].append(CirclePainting(Point(a.x, a.y), radius=0.5, color='green'))
            
            # Create nodes
            if isinstance(a, Car):
                vx = a.speed * np.cos(a.heading) / self.speed_normalization_factor
                vy = a.speed * np.sin(a.heading) / self.speed_normalization_factor
                node = [#0.0,
                        a.x/self.width_m, 
                        a.y/self.height_m,
                        vx,
                        vy,
                        1.0, # distance to closest agent
                        ]
            else:
                vx = a.speed * np.cos(a.heading) / self.speed_normalization_factor
                vy = a.speed * np.sin(a.heading) / self.speed_normalization_factor
                node = [#1.0,
                        a.x/self.width_m, 
                        a.y/self.height_m,
                        vx,
                        vy,
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
            obs = (self.graph, agent.speed)
        elif self.obs_type == 'gcn_speed_route':
            route = routes[agent.ego_route_index][wp:wp+10]
            obs = (self.graph, agent.speed, route)
        else:
            raise Exception('Unexpected observation type')
        
        return obs
    
    def _reward_others(self):
        if self.agent.lower() == 'vehicle':
            agent = self.traffic_controller.ego_vehicle
        elif self.agent.lower() == 'pedestrian':
            agent = self.traffic_controller.ego_pedestrian
        else:
            raise Exception('Unexptected agent')
        
        p_ego = np.array([agent.x, agent.y])
        
        nearby_agents = []
        for a in self.dynamic_agents:
            p_a = np.array([a.x, a.y])
            d_a_ego = np.linalg.norm(p_a - p_ego)
            if d_a_ego < self.detection_radius and a.collidable:
                if a is not agent:
                    self.closest_vehicle_distance = d_a_ego
                    nearby_agents.append(a)
                
        r_others = 0.0
        for a in nearby_agents:
            p_a = np.array([a.x, a.y])
            d_a_ego = np.linalg.norm(p_a - p_ego)
            r_others += a.speed/a.max_speed * (1.0/d_a_ego)
        return r_others
            
    
    def _test_reward_fn(self):
        info = {'end_reason': None}
        
        goal_reached = self._goal_reached()
        
        if self.agent.lower() == 'vehicle':
            collision = self.collision_exists(self.ego_vehicle)
            v = self.ego_vehicle.speed
            speed_norm_factor = self.speed_limit
        elif self.agent.lower() == 'pedestrian':
            collision = self.collision_exists(self.ego_pedestrian)
            v = self.ego_pedestrian.speed
            speed_norm_factor = self.ped_speed_limit
        else:
            raise Exception(f'Unexpected agent {self.agent}')
        
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
            
            r_others = self._reward_others()
            
            if v < 0.8 * speed_norm_factor:
                r_velocity = 1.25 * (v / speed_norm_factor)
            elif v >= 0.8 * speed_norm_factor and v < speed_norm_factor:
                r_velocity = 1.0
            else:
                r_velocity = 6.0 - 5.0 * (v / speed_norm_factor)
            
            r_idle = -1.0 if (v < 5.0/3.6) else 0.0
            r_action = -np.abs(self.acceleration_v)
            r_proximity = -1.0 + self.closest_vehicle_distance/self.detection_radius
            
            r_velocity *= self.reward_configuration['velocity']
            r_action *= self.reward_configuration['action']
            r_idle *= self.reward_configuration['idle']
            r_proximity *= self.reward_configuration['proximity']
            
            # print(f"Reward: v {r_velocity:.4f}, a {r_action:.4f}, idle: {r_idle:.4f}, proximity: {r_proximity:.4f}")
            
            r_self = r_velocity + r_action + r_idle + r_proximity
            reward = np.cos(self.svo) * r_self + np.sin(self.svo) * r_others
            # print(f"reward: {reward:.4f}, r_self: {r_self:.4f}, r_others: {r_others:.4f}")
            
        return reward, done, info
      
    def reset(self, seed=None):
        self.t = 0
        self.dynamic_agents = []
        self.static_agents = []
        N_cars = self.load_scene(self.scene_name)
        self.episode_reward = 0.0
        self.ped_ai_cross = -1 # for level 0 ped ai crossing
        
        return self._get_observation(), {'N_cars': N_cars}
        
    def step(self, action):
        # Plot graph
        for x in self.graph_plot['nodes']:
            self.add(x)
        for x in self.graph_plot['edges']:
            self.add(x)
            
        self.render()
        
        # Plot graph
        for x in self.graph_plot['nodes']:
            self.pop(x)
        for x in self.graph_plot['edges']:
            self.pop(x)
        
        if self.agent.lower() == 'vehicle':
            _, self.acceleration_v = self._action_discrete_to_continuous(action)
            self.steering_v, _, _ = self.traffic_controller.ego_controller.stanely_controller()
            
            _, self.acceleration_p = self.pedestrian_AI()
            self.steering_p = 0.0
            
        elif self.agent.lower() == 'pedestrian':
            _, self.acceleration_p = self._action_discrete_to_continuous(action)
            self.steering_p = 0.0
            
            _, self.acceleration_v = self.vehicle_AI()
            self.steering_v, _, _ = self.traffic_controller.ego_controller.stanely_controller()
        
        else:
            raise Exception('Unexpected agent type')
            
        self.tick()
        
        obs = self._get_observation()
        reward, done, info = self.reward_fn()
        
        self.episode_reward += reward
        return obs, reward, done, False, info
    
    def vehicle_AI(self):
        if not self._vehicle_goal_reached():
            if self.vehicle_level == 'L0':
                return (None, 1.0)
            else:
                s = prepare_state_for_nn(self._get_observation(agent='vehicle'))
                action = int(torch.argmax(self.vehicle_agent(s)))
                return self._action_discrete_to_continuous(action, agent='vehicle')
        else:
            # self.traffic_controller.ego_vehicle.velocity = Point(0,0)
            return (None, 0.0)
        
    def pedestrian_AI(self):
        if not self._pedestrian_goal_reached():
            if self.pedestrian_level == 'L0':
                return self.level_0_pedestrian_AI()
            else:
                s = prepare_state_for_nn(self._get_observation(agent='pedestrian'))
                action = int(torch.argmax(self.pedestrian_agent(s)))
                return self._action_discrete_to_continuous(action, agent='pedestrian')
        else:
            return (None, 0.0)
        
    def level_0_pedestrian_AI(self):
        if not self._pedestrian_goal_reached():
            agent = self.traffic_controller.ego_pedestrian
            p_ego = np.array([agent.x, agent.y])
            
            nearby_agents = []
            
            self.closest_vehicle_distance = self.detection_radius
            
            self.graph_plot = {}
            self.graph_plot['nodes'] = []
            self.graph_plot['edges'] = []
            
            # If there are close agents: don't move, else cross
            if self.ped_ai_cross == -1:
                for a in self.dynamic_agents:
                    p_a = np.array([a.x, a.y])
                    d_a_ego = np.linalg.norm(p_a - p_ego)
                    if d_a_ego < 10.0 and a.collidable:
                        nearby_agents.append(a)
                if len(nearby_agents) > 1:
                    self.ped_ai_cross = 1.0
                else:
                    self.ped_ai_cross = 0.0
            return (None, self.ped_ai_cross)
        else:
            # Stop pedestrian if they reach their goal
            # self.traffic_controller.ego_pedestrian.velocity = Point(0,0)
            return (None, 0.0)
    
    ###############################################  
    #           AUXILIARY METHODS                 #
    ###############################################
    def draw_route(self, route, color='green'):
        for i in range(len(route)):
            self.add(Painting(Point(route[i, 0], route[i, 1]), Point(0.5, 0.5), color))
        
    def print_traffic(self, event):
        print(self.traffic_controller.traffic[['id', 'route', 'waypoint', 'front_vehicle_id']])
             
    def reset_rng(self, seed=None):
        if self.testing:
            if seed is None:
                self.rng = np.random.RandomState(self.seed)
            else:
                self.rng = np.random.RandomState(seed)
    
    @property
    def ego_vehicle(self):
        return self.traffic_controller.ego_vehicle
    
    @property
    def ego_pedestrian(self):
        return self.traffic_controller.ego_pedestrian
    
    def get_transform(self, route_index, point=0, pedestrian=False):
        if not pedestrian:
            route = self.routes[route_index]
        else:
            route = self.pedestrian_routes[route_index]
        x, y = route[point]
        
        forward_vector = route[point+1] - route[point]
        heading = np.arctan2(forward_vector[1], forward_vector[0]) % (2*np.pi)
        return x, y, heading

    def _action_discrete_to_continuous(self, action, agent=None):
        ag = agent if agent is not None else self.agent.lower()
        if ag.lower() == 'vehicle':
            MAX_A = 3.0
        elif ag.lower() == 'pedestrian':
            MAX_A = 1.0
        else:
            raise Exception(f'Unpexpected agent type: {self.agent.lower()}')
        
        if action == Scene.ACTION_ACCELERATE:
            steer = 0.0
            accelerate = MAX_A
        elif action == Scene.ACTION_BRAKE:
            steer = 0.0
            accelerate = -MAX_A
        elif action == Scene.ACTION_NONE:
            steer = 0.0
            accelerate = 0.0
        else:
            raise Exception("Unexpected action: {action}")
        return steer, accelerate                