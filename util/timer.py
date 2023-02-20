# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 08:45:11 2023

@author: AdminS
"""

import time
class Timer:
    def __init__(self):
        pass
    
    def start(self):
        self.start_time = time.time()
        
    def stop(self, function_name):
        self.stop_time = time.time()
        print(f"TIMER\t{function_name} {self.stop_time - self.start_time}")