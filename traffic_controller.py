# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:46:33 2022

@author: lucac
"""
import numpy as np
import pandas as pd
import random
from agents import Car
from geometry import Point

def computeAngleDifference(a1, a2):
    a1 = np.mod(a1, 2*np.pi)
    a2 = np.mod(a2, 2*np.pi)
    return np.mod((a1 - a2) - np.pi, 2*np.pi) - np.pi

class Proportional:
    def __init__(self, vehicle, k=2.0):
        self.k = k        
        self.desired_speed = max(np.random.normal(5.0, 1.0), 1.0)
        self.vehicle = vehicle
        
    def __call__(self, x):
        control = self.k*(self.desired_speed - self.vehicle.speed)
        return control
    
    
class IntelligentDriverModel:
    def __init__(self, vehicle):
        self.desired_speed = np.random.uniform(2.0, 8.0)
        self.minimum_spacing = 4.0
        self.desired_time_headway = np.random.uniform(2.0, 6.0)
        self.max_acceleration = np.random.uniform(3.0, 5.0)
        self.comfortable_braking_deceleration = np.random.uniform(3.0, 5.0)
        self.delta = 4.0
        self.vehicle = vehicle
        
    def __call__(self, front_vehicle):
        vehicle_position = np.array([self.vehicle.center.x, 
                                     self.vehicle.center.y])
        
        if front_vehicle is not None:
            front_vehicle_position = np.array([front_vehicle.center.x, 
                                         front_vehicle.center.y])
            s_alpha = np.linalg.norm(front_vehicle_position - vehicle_position) \
                                    - self.vehicle.size.x
            s0 = self.minimum_spacing
            v_alpha = self.vehicle.speed
            d_v_alpha = self.vehicle.speed - front_vehicle.speed
            s_star = s0 + v_alpha * self.desired_time_headway \
                + v_alpha * d_v_alpha / (2*np.sqrt(self.max_acceleration \
                                    * self.comfortable_braking_deceleration))
             
            # print(f"s_alpha: {s_alpha}")
            # print(f"s0: {s0}, v_alpha {v_alpha}, d_v_alpha: {d_v_alpha}")
            # print(f"s_star: {s_star}, t1: {v_alpha * self.desired_time_headway}, t2: {v_alpha * d_v_alpha / (2*np.sqrt(self.max_acceleration  * self.comfortable_braking_deceleration))}")
                    
            a_other = - (s_star / s_alpha)**2
        else:
            a_other = 0.0
        
        a_self = (1.0 - (self.vehicle.speed / self.desired_speed)**self.delta)
        return self.max_acceleration * (a_other + a_self)
            


class CarController:
    def __init__(self, route, 
                 vehicle, 
                 initial_waypoint=0,
                 acceleration_controller='Proportional'):
        self.route = route                  # The route we want to follow
        
        self.vehicle = vehicle              # the vehicle we are controlling
        self.current_waypoint_index = initial_waypoint
        
        self._steering = 0.0
        self._acceleration = 0.0
        
        
        if acceleration_controller == 'Proportional':
            self.acceleration_controller = Proportional(self.vehicle)
        elif acceleration_controller == 'IDM':
            self.acceleration_controller = IntelligentDriverModel(self.vehicle)
        else:
            raise Exception(f'Unexpected acceleration controller: {self.accelerationController}')
    
    def update_waypoint(self):
        # Returns False if the route is over, True otherwise
        vehicle_position = np.array([self.vehicle.center.x, 
                                     self.vehicle.center.y])
        next_waypoint_index = int(self.current_waypoint_index + 1)
        forward_vector = self.route[next_waypoint_index] \
                        - self.route[self.current_waypoint_index]
        if np.dot(forward_vector, 
                  vehicle_position - self.route[next_waypoint_index]) > 0:
            self.current_waypoint_index = next_waypoint_index
            
        if self.current_waypoint_index == len(self.route) - 1:
            return False
        else:
            return True
        
    def stanely_controller(self, front_vehicle=None):
        if self.update_waypoint():
            kv = 1.0
            
            # Returnwos desired acceleration and steering
            front_axle = self.vehicle.front_axle
            forward_vector = self.route[self.current_waypoint_index + 1] - \
                            self.route[self.current_waypoint_index]
            tangent_angle = np.arctan2(forward_vector[1], forward_vector[0]) % (2*np.pi)
            
            if np.linalg.norm(forward_vector) > 0.1:
                theta_p = computeAngleDifference(self.vehicle.heading, tangent_angle)
                
                orthogonal_angle = np.arctan2(forward_vector[0], 
                                              -forward_vector[1])
                orthogonal_unit_v = np.array([np.cos(orthogonal_angle),
                                              np.sin(orthogonal_angle)])
                df = np.dot(orthogonal_unit_v, 
                            front_axle - self.route[self.current_waypoint_index])
                
                ld = max(self.vehicle.speed / kv, 1.0)
                
                self._steering = -(theta_p + np.arctan(df / ld))
                self._acceleration = self.acceleration_controller(front_vehicle)
            end_of_route = False
        else:
            end_of_route = True
        
        return self._steering, self._acceleration, end_of_route
    
    
    
"""
The traffic controller should be responsible for:
    XXX spawning agents on their routes
    XXX adding new agents as previous ones disappear
    XXX applying control to follow the route
     avoid collisions with other agents in the scene
"""
    
class TrafficController:
    def __init__(self, world, N_cars : int = 3):
        """
        Parameters
        ----------
        world : TYPE
            DESCRIPTION.
        N_agents : int, optional
            DESCRIPTION. The default is 3.

        Returns
        -------
        None.

        """
        self.world = world
        self.N_cars = N_cars
        self.traffic = pd.DataFrame(columns=['id', 'route', 'vehicle', 'controller'])
        
        for _ in range(N_cars):
            route_index = np.random.choice(self.world.non_ego_routes)
            for _ in range(300):
                if self.spawn_car(route_index, random_point=True):
                    break
                else:
                    continue
        print(self.traffic)
        
                    
    def tick(self):
        for index, row in self.traffic.iterrows():
            steering, acceleration, end_of_route = row['controller'].stanely_controller(row['front_vehicle'])
            self.traffic.loc[index, 'vehicle'].set_control(steering, acceleration)
            self.traffic.loc[index, 'waypoint'] = int(row['controller'].current_waypoint_index)
            
            if end_of_route:
                # Remove the agent from the world
                self.world.pop(row['vehicle'])
                
                # And delete the associated traffic entry
                self.traffic.drop(index, inplace=True) 
        

        # Respawn agents if they disappeared   
        N_spawn = self.N_cars - len(self.traffic)
        for _ in range(N_spawn):
            route_index = np.random.choice(self.world.non_ego_routes)
            self.spawn_car(route_index)
            
        # Update ego_vehicle route
        ego_vehicle = self.traffic[self.traffic['id'] == -1].iloc[0]['vehicle']
        y = ego_vehicle.center.y
        if np.abs(y - self.world.routes[0]) < 0.50:
            self.traffic[self.traffic['id'] == -1]['route'] = 0
        elif np.abs(y - self.world.routes[1]) < 0.50:
            self.traffic[self.traffic['id'] == -1]['route'] = 0
        else:
            self.traffic[self.traffic['id'] == -1]['route'] = None
        ######################################################################
            
        self.traffic.reset_index()
        self.update_front_vehicles()
            
            
    def spawn_car(self, route_index, random_point=False):
        if random_point:
            i = np.random.randint(0, len(self.world.routes[route_index])-1)
        else: 
            i = 0
        route = self.world.routes[route_index]
        x, y = route[i]
        
        forward_vector = route[i+1] - route[i]
        heading = np.arctan2(forward_vector[1], forward_vector[0]) % (2*np.pi)
        initial_velocity = np.random.uniform(3.0, 5.0) * np.array([
            np.cos(heading), np.sin(heading)])
        
        car = Car(Point(x, y), heading, 
                  velocity=Point(initial_velocity[0], initial_velocity[1]))
        
        if not self.world.collision_exists(car):
            self.world.add(car)
            controller = CarController(route, car, 
                                       initial_waypoint=i,
                                       acceleration_controller='IDM')
            traffic_agent = {   'id': [car.id],
                                'route': [route_index],
                                'vehicle': [car],
                                'controller': [controller],
                                'waypoint': [0],
                                'front_vehicle': [None],
                                'front_vehicle_id': [None]}
            da = pd.DataFrame(traffic_agent)
            
            # Add agent to the traffic list
            self.traffic = pd.concat((self.traffic, da),
                                     ignore_index=True)
            return True
        else:
            print("Tried to spawn car agent but collision existed")
            return False
        
    
    def add_ego_vehicle(self, ego_vehicle, ego_route=-1):
        if not self.world.collision_exists(ego_vehicle):
            traffic_agent = {
                    'id': [ego_vehicle.id],
                    'route': [ego_route],
                    'vehicle': [ego_vehicle],
                    'controller': [None],
                    'waypoint': [None],
                    'front_vehicle': [None],
                    'front_vehicle_id': [None]
                }
            da = pd.DataFrame(traffic_agent)
            
            # Add agent to the traffic list
            self.traffic = pd.concat((self.traffic, da),
                                     ignore_index=True)
        else:
            raise Exception("Tried to spawn car agent but collision existed")
    
        
    def update_front_vehicles(self, max_range=30):
        for index, row in self.traffic.iterrows():
            df = self.traffic[(self.traffic['route'] == row['route']) & 
                              (self.traffic['waypoint'] >= row['waypoint']) &
                              (self.traffic['id'] != row['id'])].sort_values('waypoint')
            
            if len(df) >= 1:
                self.traffic.loc[index, 'front_vehicle'] = df.iloc[0]['vehicle']
                self.traffic.loc[index, 'front_vehicle_id'] = self.traffic.loc[index, 'front_vehicle'].id
            else:
                self.traffic.loc[index, 'front_vehicle'] = None
                self.traffic.loc[index, 'front_vehicle_id'] = None
                
        
        
        
            
            
            
                
        
        
        
            
    