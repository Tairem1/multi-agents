from entities import RectangleEntity, CircleEntity, RingEntity
from geometry import Point
import numpy as np

# For colors, we use tkinter colors. See http://www.science.smith.edu/dftwiki/index.php/Color_Charts_for_TKinter

class Car(RectangleEntity):
    def __init__(self, center: Point, heading: float, color: str = 'red', velocity=Point(0,0)):
        size = Point(4., 2.)
        movable = True
        friction = 0.06
        super(Car, self).__init__(center, heading, size, movable, friction, velocity)
        self.color = color
        self.collidable = True
    
    @property
    def front_axle(self):
        x = self.center.x + np.cos(self.heading)*self.size.x/2.0
        y = self.center.y + np.sin(self.heading)*self.size.x/2.0
        return np.array([x, y])
        
class Pedestrian(CircleEntity):
    def __init__(self, center: Point, heading: float, color: str = 'LightSalmon3'): # after careful consideration, I decided my color is the same as a salmon, so here we go.
        radius = 0.5
        movable = True
        friction = 0.2
        super(Pedestrian, self).__init__(center, heading, radius, movable, friction)
        self.color = color
        self.collidable = True
        
class RectangleBuilding(RectangleEntity):
    def __init__(self, center: Point, size: Point, color: str = 'gray26'):
        heading = 0.
        movable = False
        friction = 0.
        super(RectangleBuilding, self).__init__(center, heading, size, movable, friction)
        self.color = color
        self.collidable = True
        
class CircleBuilding(CircleEntity):
    def __init__(self, center: Point, radius: float, color: str = 'gray26'):
        heading = 0.
        movable = False
        friction = 0.
        super(CircleBuilding, self).__init__(center, heading, radius, movable, friction)
        self.color = color
        self.collidable = True

class RingBuilding(RingEntity):
    def __init__(self, center: Point, inner_radius: float, outer_radius: float, color: str = 'gray26'):
        heading = 0.
        movable = False
        friction = 0.
        super(RingBuilding, self).__init__(center, heading, inner_radius, outer_radius, movable, friction)
        self.color = color
        self.collidable = True

class Painting(RectangleEntity):
    def __init__(self, center: Point, size: Point, color: str = 'gray26', heading: float = 0.):
        movable = False
        friction = 0.
        super(Painting, self).__init__(center, heading, size, movable, friction)
        self.color = color
        self.collidable = False
