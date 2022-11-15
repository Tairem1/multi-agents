# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:46:33 2022

@author: lucac
"""

import numpy as np

def computeAngleDifference(a1, a2):
    a1 = np.mod(a1, 2*np.pi)
    a2 = np.mod(a2, 2*np.pi)
    return np.mod((a1 - a2) - np.pi, 2*np.pi) - np.pi

class CarController:
    def __init__(self, route, desired_speed, vehicle):
        self.route = route                  # The route we want to follow
        self.desired_speed = desired_speed  # The car desired speed
        self.vehicle = vehicle              # the vehicle we are controlling
        
        self.current_waypoint_index = 0
        
    def update_waypoint(self):
        # Returns False if the route is over, True otherwise
        
        if self.current_waypoint_index == len(self.route) - 5:
            return False
        else:
            vehicle_position = np.array([self.vehicle.center.x, 
                                         self.vehicle.center.y])
            next_waypoint_index = self.current_waypoint_index + 1
            forward_vector = self.route[next_waypoint_index] \
                                - self.route[self.current_waypoint_index]
            if np.dot(forward_vector, 
                      vehicle_position - self.route[next_waypoint_index]) > 0:
                self.current_waypoint_index = next_waypoint_index
            return True
        
    def stanely_controller(self):
        if self.update_waypoint():
            kv = 1.0
            ka = 2.0
            
            # Returns desired acceleration and steering
            front_axle = self.vehicle.front_axle
            
            forward_vector = self.route[self.current_waypoint_index + 1] - \
                            self.route[self.current_waypoint_index]
            tangent_angle = np.arctan2(forward_vector[1], forward_vector[0]) % (2*np.pi)
            
            
            theta_p = computeAngleDifference(self.vehicle.heading, tangent_angle)
            
            orthogonal_angle = np.arctan2(forward_vector[0], 
                                          -forward_vector[1])
            orthogonal_unit_v = np.array([np.cos(orthogonal_angle),
                                          np.sin(orthogonal_angle)])
            
            df = np.dot(orthogonal_unit_v, 
                        front_axle - self.route[self.current_waypoint_index])
            
            ld = max(self.vehicle.speed / kv, 1.0)
            
            steering = -(theta_p + np.arctan(df / ld))
            # steering = -theta_p
            acceleration = ka*(self.desired_speed - self.vehicle.speed)
            end_of_route = False
        else:
            steering = 0.0
            acceleration = 0.0
            end_of_route = True
        
        return steering, acceleration, end_of_route
    
    
"""
The traffic controller should be responsible for:
    spawning agents on their routes
    applying control to follow the route
    avoid collisions with other agents in the scene
"""
    
# class TrafficController:
    
    
#     def __init__(self, world):
#         self.traffic = {}
        
#         for agent in world.agents:
            
        
        
        
        
        
        
            
    