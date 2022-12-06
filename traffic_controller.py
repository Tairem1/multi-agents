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

class CarController:
    def __init__(self, route, desired_speed, vehicle, 
                 initial_waypoint=0):
        self.route = route                  # The route we want to follow
        self.desired_speed = desired_speed  # The car desired speed
        self.vehicle = vehicle              # the vehicle we are controlling
        self.current_waypoint_index = initial_waypoint
        
        self._steering = 0.0
        self._acceleration = 0.0
        
        
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
        
    def stanely_controller(self):
        if self.update_waypoint():
            kv = 1.0
            ka = 2.0
            
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
                self._acceleration = ka*(self.desired_speed - self.vehicle.speed)
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
        self.car_counter = 0
        self.N_cars = N_cars
        self.traffic = pd.DataFrame(columns=['id', 'route', 'vehicle', 'controller'])
        
        for _ in range(N_cars):
            route_index = np.random.randint(len(self.world.routes))
            for _ in range(300):
                if self.spawn_car(route_index, random_point=True):
                    break
                else:
                    continue
        print(self.traffic)
        
                    
    def tick(self):
        self.traffic.reset_index()
        
        for index, row in self.traffic.iterrows():
            steering, acceleration, end_of_route = row['controller'].stanely_controller()
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
            route_index = np.random.randint(len(self.world.routes))
            self.spawn_car(route_index)
            
            
    def spawn_car(self, route_index, random_point=False):
        if random_point:
            i = np.random.randint(0, len(self.world.routes[route_index])-1)
        else: 
            i = 0
        route = self.world.routes[route_index]
        x, y = route[i]
        
        forward_vector = route[i+1] - route[i]
        heading = np.arctan2(forward_vector[1], forward_vector[0]) % (2*np.pi)
        
        car = Car(Point(x, y), heading)
        
        if not self.world.collision_exists(car):
            self.world.add(car)
            desired_speed = max(np.random.normal(5.0, 1.0), 1.0)
            controller = CarController(route, desired_speed, 
                                       car, initial_waypoint=i)
            traffic_agent = {   'id': [self.car_counter],
                                'route': [route_index],
                                'vehicle': [car],
                                'controller': [controller],
                                'waypoint': [0]}
            da = pd.DataFrame(traffic_agent)
            
            # Add agent to the traffic list
            self.traffic = pd.concat((self.traffic, da),
                                     ignore_index=True)
            self.car_counter += 1
            return True
        else:
            print("Tried to spawn car agent but collision existed")
            return False
        
        
        
        
            
            
            
                
        
        
        
            
    